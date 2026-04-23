"""Codex CLI provider - shells out to OpenAI Codex CLI."""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger

from providers.base import BaseProvider, ProviderConfig
from providers.common import (
    HeuristicToolParser,
    SSEBuilder,
    ThinkTagParser,
    append_request_id,
    get_user_facing_error_message,
    map_stop_reason,
)


class CodexCLIProvider(BaseProvider):
    """Provider that shells out to OpenAI Codex CLI."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        model: str = "gpt-5.4",
    ):
        super().__init__(config)
        self._model = model
        self._provider_name = "codex_cli"

    async def cleanup(self) -> None:
        """No resources to clean up for CLI provider."""
        pass

    def _build_codex_messages(self, request: Any) -> list[dict]:
        """Convert Anthropic messages to Codex format."""
        messages = []

        # Get messages from request
        request_messages = getattr(request, "messages", [])
        if not request_messages:
            return messages

        for msg in request_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle content as string or list of blocks
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Convert Anthropic content blocks to text
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        if block_type == "text":
                            text_parts.append(block.get("text", ""))
                        elif block_type == "image":
                            # Skip images for now
                            pass
                        elif block_type == "thinking":
                            # Skip thinking blocks
                            pass
                if text_parts:
                    messages.append({"role": role, "content": "\n".join(text_parts)})

        return messages

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format by shelling out to Codex CLI."""
        tag = self._provider_name
        message_id = f"msg_{uuid.uuid4()}"
        sse = SSEBuilder(message_id, self._model, input_tokens)

        req_tag = f" request_id={request_id}" if request_id else ""
        logger.info(
            "{}_STREAM:{} model={} msgs={}",
            tag,
            req_tag,
            self._model,
            len(getattr(request, "messages", [])),
        )

        yield sse.message_start()

        think_parser = ThinkTagParser()
        heuristic_parser = HeuristicToolParser()
        thinking_enabled = self._is_thinking_enabled(request)

        finish_reason = None
        error_occurred = False
        error_message = ""

        try:
            # Build the prompt from messages
            messages = self._build_codex_messages(request)
            if not messages:
                raise ValueError("No messages in request")

            # Combine all messages into a single prompt
            prompt_parts = []
            for msg in messages:
                if msg["role"] == "user":
                    prompt_parts.append(msg["content"])
                elif msg["role"] == "assistant":
                    prompt_parts.append(f"Assistant: {msg['content']}")

            prompt = "\n\n".join(prompt_parts)

            # Run codex CLI with JSON output
            cmd = ["codex", "exec", "--json", "--model", self._model]

            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Send prompt to stdin
            process.stdin.write(prompt.encode())
            await process.stdin.drain()
            process.stdin.close()

            # Read JSONL output
            full_text = ""
            async for line in process.stdout:
                line = line.decode().strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                    event_type = event.get("type")

                    if event_type == "item.completed":
                        item = event.get("item", {})
                        if item.get("type") == "agent_message":
                            text = item.get("text", "")
                            if text:
                                full_text += text
                                # Process text through parsers
                                for part in think_parser.feed(text):
                                    if part.type == "think":
                                        if not thinking_enabled:
                                            continue
                                        for event in sse.ensure_thinking_block():
                                            yield event
                                        yield sse.emit_thinking_delta(part.content)
                                    else:
                                        filtered_text, detected_tools = heuristic_parser.feed(part.content)

                                        if filtered_text:
                                            for event in sse.ensure_text_block():
                                                yield event
                                            yield sse.emit_text_delta(filtered_text)

                                        for tool_use in detected_tools:
                                            for event in sse.close_content_blocks():
                                                yield event

                                            block_idx = sse.blocks.allocate_index()
                                            if tool_use.get("name") == "Task" and isinstance(
                                                tool_use.get("input"), dict
                                            ):
                                                tool_use["input"]["run_in_background"] = False
                                            yield sse.content_block_start(
                                                block_idx,
                                                "tool_use",
                                                id=tool_use["id"],
                                                name=tool_use["name"],
                                            )
                                            yield sse.content_block_delta(
                                                block_idx,
                                                "input_json_delta",
                                                json.dumps(tool_use["input"]),
                                            )
                                            yield sse.content_block_stop(block_idx)

                    elif event_type == "turn.completed":
                        finish_reason = "stop"
                        usage = event.get("usage", {})
                        logger.debug(
                            "{} usage: input={} output={}",
                            tag,
                            usage.get("input_tokens", 0),
                            usage.get("output_tokens", 0),
                        )

                    elif event_type == "error":
                        error_occurred = True
                        error_message = event.get("message", "Unknown error")
                        logger.error("{} codex error: {}", tag, error_message)

                    elif event_type == "turn.failed":
                        error_occurred = True
                        error_msg = event.get("error", {})
                        if isinstance(error_msg, dict):
                            error_message = error_msg.get("message", "Unknown error")
                        else:
                            error_message = str(error_msg)
                        logger.error("{} codex turn failed: {}", tag, error_message)

                except json.JSONDecodeError:
                    logger.warning("{} failed to parse JSONL line: {}", tag, line)

            # Wait for process to complete
            await process.wait()

            if process.returncode != 0 and not error_occurred:
                stderr = await process.stderr.read()
                stderr_text = stderr.decode().strip()
                if stderr_text:
                    logger.error("{} codex stderr: {}", tag, stderr_text)
                error_occurred = True
                error_message = f"Codex CLI exited with code {process.returncode}"

        except Exception as e:
            logger.error("{}_ERROR:{} {}: {}", tag, req_tag, type(e).__name__, e)
            error_occurred = True
            error_message = get_user_facing_error_message(e)

        if error_occurred:
            error_message = append_request_id(error_message, request_id)
            logger.info(
                "{}_STREAM: Emitting SSE error event for {}{}",
                tag,
                type(e).__name__,
                req_tag,
            )
            for event in sse.close_content_blocks():
                yield event
            for event in sse.emit_error(error_message):
                yield event

        # Flush remaining content
        remaining = think_parser.flush()
        if remaining:
            if remaining.type == "think":
                if not thinking_enabled:
                    remaining = None
                else:
                    for event in sse.ensure_thinking_block():
                        yield event
                    yield sse.emit_thinking_delta(remaining.content)
            if remaining and remaining.type == "text":
                for event in sse.ensure_text_block():
                    yield event
                yield sse.emit_text_delta(remaining.content)

        for tool_use in heuristic_parser.flush():
            for event in sse.close_content_blocks():
                yield event

            block_idx = sse.blocks.allocate_index()
            yield sse.content_block_start(
                block_idx,
                "tool_use",
                id=tool_use["id"],
                name=tool_use["name"],
            )
            if tool_use.get("name") == "Task" and isinstance(
                tool_use.get("input"), dict
            ):
                tool_use["input"]["run_in_background"] = False
            yield sse.content_block_delta(
                block_idx,
                "input_json_delta",
                json.dumps(tool_use["input"]),
            )
            yield sse.content_block_stop(block_idx)

        if (
            not error_occurred
            and sse.blocks.text_index == -1
            and not sse.blocks.tool_states
        ):
            for event in sse.ensure_text_block():
                yield event
            yield sse.emit_text_delta(" ")

        for event in sse.close_all_blocks():
            yield event

        output_tokens = sse.estimate_output_tokens()
        yield sse.message_delta(map_stop_reason(finish_reason), output_tokens)
        yield sse.message_stop()
