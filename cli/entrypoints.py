"""CLI entry points for the installed package."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def _get_lock_file_path() -> Path:
    """Get the path to the lock file for the proxy server."""
    config_dir = Path.home() / ".config" / "free-claude-code"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "proxy.lock"


def _is_proxy_running(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)  # Send signal 0 to check if process exists
        return True
    except OSError:
        return False


def _acquire_lock() -> bool:
    """Try to acquire the lock file for the proxy server.

    Returns True if lock was acquired (or proxy already running), False otherwise.
    """
    lock_file = _get_lock_file_path()

    if lock_file.exists():
        try:
            existing_pid = int(lock_file.read_text().strip())
            if _is_proxy_running(existing_pid):
                return True  # Proxy already running
        except ValueError, OSError:
            pass  # Lock file is corrupted, continue

    # Write our PID to the lock file
    try:
        lock_file.write_text(str(os.getpid()), encoding="utf-8")
        return True
    except OSError:
        return False


def _release_lock() -> None:
    """Release the lock file if it belongs to this process."""
    lock_file = _get_lock_file_path()

    if lock_file.exists():
        try:
            existing_pid = int(lock_file.read_text().strip())
            if existing_pid == os.getpid():
                lock_file.unlink()
        except ValueError, OSError:
            pass  # Lock file is corrupted or doesn't exist


def serve() -> None:
    """Start the FastAPI server (registered as `free-claude-code` script)."""
    import uvicorn

    from cli.process_registry import kill_all_best_effort
    from config.settings import get_settings

    settings = get_settings()

    # Acquire lock before starting
    if not _acquire_lock():
        print("✗ Failed to acquire lock. Another instance may be starting.")
        sys.exit(1)

    try:
        uvicorn.run(
            "api.app:app",
            host=settings.host,
            port=settings.port,
            log_level="debug",
            timeout_graceful_shutdown=5,
        )
    finally:
        _release_lock()
        kill_all_best_effort()


def init() -> None:
    """Scaffold config at ~/.config/free-claude-code/.env (registered as `fcc-init`)."""
    import importlib.resources

    config_dir = Path.home() / ".config" / "free-claude-code"
    env_file = config_dir / ".env"

    if env_file.exists():
        print(f"Config already exists at {env_file}")
        print("Delete it first if you want to reset to defaults.")
        return

    config_dir.mkdir(parents=True, exist_ok=True)
    template = (
        importlib.resources.files("config").joinpath("env.example").read_text("utf-8")
    )
    env_file.write_text(template, encoding="utf-8")
    print(f"Config created at {env_file}")
    print(
        "Edit it to set your API keys and model preferences, then run: free-claude-code"
    )


def run() -> None:
    """Start proxy in background and launch Claude Code (registered as `fcc-run`).

    Usage:
        fcc-run                    # Launch Claude with default model
        fcc-run --model opus       # Launch Claude with Opus model
        fcc-run --model sonnet     # Launch Claude with Sonnet model
        fcc-run --model haiku      # Launch Claude with Haiku model
        fcc-run --model custom     # Launch Claude with custom model (set MODEL in .env)
    """
    from config.settings import get_settings

    settings = get_settings()

    # Check if proxy is already running
    proxy_url = f"http://{settings.host}:{settings.port}"
    lock_file = _get_lock_file_path()

    proxy_already_running = False
    if lock_file.exists():
        try:
            existing_pid = int(lock_file.read_text().strip())
            if _is_proxy_running(existing_pid):
                proxy_already_running = True
                print(f"✓ Proxy already running (PID: {existing_pid}) at {proxy_url}")
        except ValueError, OSError:
            pass  # Lock file is corrupted, continue

    if not proxy_already_running:
        # Start proxy in background
        print(f"Starting proxy server at {proxy_url}...")
        _proxy_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api.app:app",
                "--host",
                settings.host,
                "--port",
                str(settings.port),
                "--log-level",
                "warning",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Wait for proxy to be ready
        max_wait = 10
        for i in range(max_wait):
            try:
                import httpx

                with httpx.Client(timeout=1.0) as client:
                    response = client.get(f"{proxy_url}/v1/models")
                    if response.status_code == 200:
                        print(f"✓ Proxy started at {proxy_url}")
                        break
            except Exception:
                if i < max_wait - 1:
                    time.sleep(0.5)
                else:
                    print("✗ Failed to start proxy server")
                    sys.exit(1)

    # Parse model argument
    model_map = {
        "opus": settings.model_opus,
        "sonnet": settings.model_sonnet,
        "haiku": settings.model_haiku,
    }

    model = None
    model_display_name = None
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model_arg = sys.argv[idx + 1].lower()
            model = model_map.get(model_arg, settings.model)
            if model_arg not in model_map and model_arg != "custom":
                print(f"Unknown model '{model_arg}', using default")
                model = settings.model
            model_display_name = model_arg
        else:
            model = settings.model
            model_display_name = "default"
    else:
        model = settings.model
        model_display_name = "default"

    # Extract the actual model name for display (e.g., "z-ai/glm4.7" from "nvidia_nim/z-ai/glm4.7")
    if model and "/" in model:
        actual_model_name = model.split("/", 1)[1]
    else:
        actual_model_name = model or "unknown"

    # Get auth token from .env if configured
    auth_token = settings.anthropic_auth_token or "freecc"

    # Set environment variables for Claude Code
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = proxy_url
    env["ANTHROPIC_AUTH_TOKEN"] = f"{auth_token}:{model}" if model else auth_token

    print(f"Model mapping: {model_display_name} → {actual_model_name}")
    print(f"Launching Claude Code with model: {actual_model_name}")
    print("Press Ctrl+C to exit")

    # Launch Claude Code
    try:
        subprocess.run(["claude"], env=env)
    except KeyboardInterrupt:
        print("\nExiting...")
    except FileNotFoundError:
        print(
            "✗ Claude Code CLI not found. Install it from: https://github.com/anthropics/claude-code"
        )
        sys.exit(1)
