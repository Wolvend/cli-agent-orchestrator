"""Constants for CLI Agent Orchestrator application."""

import os
from pathlib import Path
import tempfile

from cli_agent_orchestrator.models.provider import ProviderType

# Session configuration
SESSION_PREFIX = "cao-"

# Available providers (derived from enum)
PROVIDERS = [p.value for p in ProviderType]
DEFAULT_PROVIDER = ProviderType.Q_CLI.value

# Tmux capture limits
TMUX_HISTORY_LINES = 200

# Application directories
_DEFAULT_CAO_HOME_DIR = Path.home() / ".aws" / "cli-agent-orchestrator"
# Override CAO state/log location when running in sandboxed environments.
# If unset, CAO uses the user's home directory as usual (or falls back to /tmp if not writable).
_CAO_HOME_DIR_OVERRIDE = os.getenv("CAO_HOME_DIR")

def _ensure_writable_dir(path: Path) -> bool:
    """Ensure a directory exists and is writable.

    Note: In sandboxed environments, the directory may exist but still be non-writable. We validate
    by creating a temporary file under the directory.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="._cao_write_test_", dir=str(path), delete=True) as tmp:
            tmp.write(b"1")
            tmp.flush()
        return True
    except OSError:
        return False


if _CAO_HOME_DIR_OVERRIDE:
    CAO_HOME_DIR = Path(_CAO_HOME_DIR_OVERRIDE).expanduser()
    if not _ensure_writable_dir(CAO_HOME_DIR):
        raise RuntimeError(f"CAO_HOME_DIR is not writable: {CAO_HOME_DIR}")
else:
    if _ensure_writable_dir(_DEFAULT_CAO_HOME_DIR):
        CAO_HOME_DIR = _DEFAULT_CAO_HOME_DIR
    else:
        CAO_HOME_DIR = Path("/tmp/cli-agent-orchestrator")
        if not _ensure_writable_dir(CAO_HOME_DIR):
            raise RuntimeError(f"Fallback CAO_HOME_DIR is not writable: {CAO_HOME_DIR}")
DB_DIR = CAO_HOME_DIR / "db"
LOG_DIR = CAO_HOME_DIR / "logs"
TERMINAL_LOG_DIR = LOG_DIR / "terminal"
TERMINAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Terminal log configuration
INBOX_POLLING_INTERVAL = 5  # Seconds between polling for log file changes
INBOX_SERVICE_TAIL_LINES = 5  # Number of lines to check in get_status for inbox service

# Cleanup configuration
RETENTION_DAYS = 14  # Days to keep terminals, messages, and logs

AGENT_CONTEXT_DIR = CAO_HOME_DIR / "agent-context"

# Agent store directories
LOCAL_AGENT_STORE_DIR = CAO_HOME_DIR / "agent-store"

# Q CLI directories
Q_AGENTS_DIR = Path.home() / ".aws" / "amazonq" / "cli-agents"

# Kiro CLI directories
KIRO_AGENTS_DIR = Path.home() / ".kiro" / "agents"

# Database configuration
DATABASE_FILE = DB_DIR / "cli-agent-orchestrator.db"
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"

# Server configuration
SERVER_HOST = "localhost"
SERVER_PORT = 9889
SERVER_VERSION = "0.1.0"
API_BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
