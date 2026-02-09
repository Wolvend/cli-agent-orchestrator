"""Codex provider implementation.

This provider intentionally avoids driving the interactive Codex TUI (which is difficult to
reliably parse from tmux pane captures). Instead, it runs Codex in one-shot mode via
`codex exec` and uses a stable CAO marker line to detect readiness/completion.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from pathlib import Path
from typing import Optional

from cli_agent_orchestrator.clients.tmux import tmux_client
from cli_agent_orchestrator.constants import CAO_HOME_DIR
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.base import BaseProvider
from cli_agent_orchestrator.utils.agent_profiles import load_agent_profile
from cli_agent_orchestrator.utils.terminal import wait_for_shell, wait_until_status

logger = logging.getLogger(__name__)

ANSI_CODE_PATTERN = r"\x1b\[[0-9;]*m"

# Stable CAO markers appended by this provider (not emitted by Codex itself).
IDLE_MARKER = "CAO_CODEX_IDLE"
DONE_MARKER = "CAO_CODEX_DONE"
ERROR_MARKER = "CAO_CODEX_ERROR"

# Match markers as full lines.
IDLE_LINE_PATTERN = rf"^{re.escape(IDLE_MARKER)}$"
IDLE_AT_END_PATTERN = rf"(?:{IDLE_LINE_PATTERN})\s*\Z"
DONE_LINE_PATTERN = rf"^{re.escape(DONE_MARKER)}$"
ERROR_LINE_PATTERN = rf"^{re.escape(ERROR_MARKER)}$"

# Best-effort detection for interactive approval prompts if Codex blocks on stdin.
WAITING_PROMPT_PATTERN = r"Approve this command\?\s*\[y/n\]|(?:\by/n\b|\byes/no\b)"


class CodexProvider(BaseProvider):
    """Provider for Codex CLI tool integration (non-interactive `codex exec`)."""

    def __init__(
        self,
        terminal_id: str,
        session_name: str,
        window_name: str,
        agent_profile: Optional[str] = None,
    ):
        super().__init__(terminal_id, session_name, window_name)
        self._initialized = False
        self._agent_profile = agent_profile

        state_dir = CAO_HOME_DIR / "providers" / "codex" / terminal_id
        state_dir.mkdir(parents=True, exist_ok=True)
        self._state_dir = state_dir
        self._last_message_path = state_dir / "last_message.txt"

    def _resolve_model(self) -> Optional[str]:
        """Resolve Codex model from env or agent profile frontmatter."""
        model = os.getenv("CAO_CODEX_MODEL")
        if model:
            return model

        if not self._agent_profile:
            return None

        try:
            profile = load_agent_profile(self._agent_profile)
            return profile.model
        except Exception:
            return None

    def _extra_exec_args(self) -> list[str]:
        """Optional extra args for `codex exec` supplied via env."""
        raw = os.getenv("CAO_CODEX_EXEC_ARGS") or os.getenv("CAO_CODEX_ARGS")
        if not raw:
            return []
        try:
            return shlex.split(raw)
        except ValueError:
            # If parsing fails, ignore rather than breaking the provider.
            logger.warning("Invalid CAO_CODEX_EXEC_ARGS; ignoring")
            return []

    def initialize(self) -> bool:
        """Initialize Codex provider by preparing the shell environment.

        We disable the shell prompt (PS1) so the CAO idle marker can reliably be the last line,
        making log/pane parsing deterministic.
        """
        if not wait_for_shell(tmux_client, self.session_name, self.window_name, timeout=10.0):
            raise TimeoutError("Shell initialization timed out after 10 seconds")

        # Ensure deterministic output: remove prompt and print an idle marker.
        setup_cmd = "export PS1=''; export PROMPT_COMMAND=''; echo " + shlex.quote(IDLE_MARKER)
        tmux_client.send_keys(self.session_name, self.window_name, setup_cmd)

        if not wait_until_status(self, TerminalStatus.IDLE, timeout=10.0, polling_interval=0.5):
            raise TimeoutError("Codex shell initialization timed out after 10 seconds")

        self._initialized = True
        return True

    def format_input(self, message: str) -> str:
        """Wrap the message as a one-shot `codex exec` invocation.

        We write the final agent message to a deterministic file under CAO_HOME_DIR and then append
        CAO markers so `get_status()` can cheaply infer completion.
        """
        # Avoid embedding raw newlines into tmux send-keys; treat them as literal "\n".
        safe_message = message.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")

        # Clear any previous output so callers don't accidentally read stale data mid-run.
        out_path = self._last_message_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        command_parts: list[str] = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--color",
            "never",
            "-o",
            str(out_path),
        ]

        model = self._resolve_model()
        if model:
            command_parts.extend(["--model", model])

        command_parts.extend(self._extra_exec_args())

        # Provide the prompt as an argument (most reliable when driving via tmux send-keys).
        command_parts.append(safe_message)
        codex_cmd = f"rm -f {shlex.quote(str(out_path))} ; {shlex.join(command_parts)}"

        # Append deterministic markers (rc-aware) and end in IDLE_MARKER.
        # NOTE: Do not use `set -e`; we want the marker emission to run even if codex fails.
        marker_tail = (
            f" ; rc=$? ; "
            f'if [ "$rc" -eq 0 ]; then echo {shlex.quote(DONE_MARKER)} ; '
            f"else echo {shlex.quote(ERROR_MARKER)} ; fi ; "
            f"echo {shlex.quote(IDLE_MARKER)}"
        )

        return codex_cmd + marker_tail

    def get_status(self, tail_lines: Optional[int] = None) -> TerminalStatus:
        """Get Codex status by analyzing terminal output."""
        output = tmux_client.get_history(self.session_name, self.window_name, tail_lines=tail_lines)
        if not output:
            return TerminalStatus.ERROR

        clean_output = re.sub(ANSI_CODE_PATTERN, "", output)
        tail_output = "\n".join(clean_output.splitlines()[-40:])

        if re.search(WAITING_PROMPT_PATTERN, tail_output, re.IGNORECASE | re.MULTILINE):
            return TerminalStatus.WAITING_USER_ANSWER

        has_idle_at_end = bool(
            re.search(IDLE_AT_END_PATTERN, clean_output, re.MULTILINE | re.DOTALL)
        )
        if not has_idle_at_end:
            return TerminalStatus.PROCESSING

        # At an idle marker. Prefer explicit ERROR marker if present, otherwise DONE.
        if re.search(ERROR_LINE_PATTERN, clean_output, re.MULTILINE):
            return TerminalStatus.ERROR
        if re.search(DONE_LINE_PATTERN, clean_output, re.MULTILINE):
            return TerminalStatus.COMPLETED

        return TerminalStatus.IDLE

    def get_idle_pattern_for_log(self) -> str:
        """Return Codex IDLE marker pattern for log files."""
        return IDLE_MARKER

    def extract_last_message_from_script(self, script_output: str) -> str:
        """Return the last message from the `--output-last-message` file."""
        path = self._last_message_path
        if not path.exists():
            raise ValueError("No Codex response found - output file missing")

        data = path.read_text(encoding="utf-8", errors="replace").strip()
        if not data:
            raise ValueError("Empty Codex response - output file was empty")

        return data

    def exit_cli(self) -> str:
        """Exit the provider session."""
        # We run inside a shell; exiting closes the pane.
        return "exit"

    def cleanup(self) -> None:
        """Clean up Codex provider."""
        self._initialized = False
