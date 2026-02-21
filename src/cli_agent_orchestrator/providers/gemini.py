"""Gemini CLI provider implementation.

This provider avoids a long-lived interactive Gemini session. Instead, it executes
one-shot Gemini invocations per input (similar to Codex/Claude providers) and emits
stable CAO markers for deterministic status parsing.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import uuid
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

# Stable CAO markers appended by this provider (not emitted by Gemini itself).
IDLE_MARKER = "CAO_GEMINI_IDLE"
DONE_MARKER = "CAO_GEMINI_DONE"
ERROR_MARKER = "CAO_GEMINI_ERROR"

IDLE_LINE_PATTERN = rf"^{re.escape(IDLE_MARKER)}$"
IDLE_AT_END_PATTERN = rf"(?:{IDLE_LINE_PATTERN})\s*\Z"
DONE_LINE_PATTERN = rf"^{re.escape(DONE_MARKER)}$"
ERROR_LINE_PATTERN = rf"^{re.escape(ERROR_MARKER)}$"

# OAuth/API-key/model-selection flows that require interactive user action.
WAITING_PROMPT_PATTERN = (
    r"(?:Please visit the following URL to authorize the application:|"
    r"Enter the authorization code:|"
    r"API\s*key|Gemini\s*API\s*Key|Paste\s+your\s+API\s+key|Select\s+Model)"
)

# Legacy interactive patterns kept as fallback for compatibility with existing fixtures/logs.
USER_PREFIX_PATTERN = r"^User:\s+"
ASSISTANT_PREFIX_PATTERN = r"^Model:\s+"
ERROR_PREFIX_PATTERN = r"^(?:✕\s+|Error:)"
PROCESSING_PATTERN = r"\b(?:responding|loading)\b"
IDLE_PROMPT_TEXT_PATTERN = r"Type your message"
IDLE_PROMPT_AT_END_PATTERN = rf"(?:{IDLE_PROMPT_TEXT_PATTERN}.*)\s*\Z"

VALID_APPROVAL_MODES = {"default", "auto_edit", "yolo", "plan"}


class GeminiProvider(BaseProvider):
    """Provider for Gemini CLI tool integration."""

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

        state_dir = CAO_HOME_DIR / "providers" / "gemini" / terminal_id
        state_dir.mkdir(parents=True, exist_ok=True)
        self._state_dir = state_dir
        self._last_message_path = state_dir / "last_message.txt"

    def _resolve_model(self) -> Optional[str]:
        """Resolve Gemini model name from env or agent profile frontmatter."""
        model = os.getenv("CAO_GEMINI_MODEL")
        if model:
            return model

        if not self._agent_profile:
            return None

        try:
            profile = load_agent_profile(self._agent_profile)
            return profile.model
        except Exception:
            return None

    def _gemini_home_dir(self) -> Path:
        """Return a writable HOME for Gemini CLI."""
        return CAO_HOME_DIR / "providers" / "gemini-home"

    def _resolve_approval_mode(self) -> Optional[str]:
        raw = (
            os.getenv("CAO_GEMINI_APPROVAL_MODE")
            or os.getenv("MODELJUMP_GEMINI_APPROVAL_MODE")
            or os.getenv("TASKFORK_GEMINI_APPROVAL_MODE")
            or "default"
        )
        mode = raw.strip()
        return mode if mode in VALID_APPROVAL_MODES else None

    def _extra_args(self) -> list[str]:
        raw = os.getenv("CAO_GEMINI_ARGS")
        if not raw:
            return []
        try:
            return shlex.split(raw)
        except ValueError:
            logger.warning("Invalid CAO_GEMINI_ARGS; ignoring")
            return []

    def initialize(self) -> bool:
        """Initialize provider shell (Gemini process is started per input)."""
        if not wait_for_shell(tmux_client, self.session_name, self.window_name, timeout=10.0):
            raise TimeoutError("Shell initialization timed out after 10 seconds")

        setup_cmd = "export PS1=''; export PROMPT_COMMAND=''; echo " + shlex.quote(IDLE_MARKER)
        tmux_client.send_keys(self.session_name, self.window_name, setup_cmd)
        if not wait_until_status(self, TerminalStatus.IDLE, timeout=10.0, polling_interval=0.5):
            raise TimeoutError("Gemini shell initialization timed out after 10 seconds")

        self._initialized = True
        return True

    def format_input(self, message: str) -> str:
        """Wrap message as one-shot Gemini CLI invocation."""
        out_path = self._last_message_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        prompt_path = self._state_dir / f"prompt_{uuid.uuid4().hex}.txt"
        prompt_path.write_text(message, encoding="utf-8", errors="strict")

        gemini_home = self._gemini_home_dir()
        gemini_home.mkdir(parents=True, exist_ok=True)

        parts: list[str] = [
            "gemini",
            "--output-format",
            "text",
            "--prompt",
            "",
        ]

        approval_mode = self._resolve_approval_mode()
        if approval_mode:
            parts.extend(["--approval-mode", approval_mode])

        model = self._resolve_model()
        if model:
            parts.extend(["--model", model])

        parts.extend(self._extra_args())

        env_prefix = f"HOME={shlex.quote(str(gemini_home))}"
        run_cmd = (
            f"rm -f {shlex.quote(str(out_path))} ; "
            f"cat {shlex.quote(str(prompt_path))} | {env_prefix} {shlex.join(parts)} "
            f"> {shlex.quote(str(out_path))} 2>&1"
        )

        marker_tail = (
            " ; rc=$? ; "
            f"rm -f {shlex.quote(str(prompt_path))} ; "
            f'if [ "$rc" -eq 0 ]; then echo {shlex.quote(DONE_MARKER)} ; '
            f"else echo {shlex.quote(ERROR_MARKER)} ; fi ; "
            f"echo {shlex.quote(IDLE_MARKER)}"
        )

        return run_cmd + marker_tail

    def _legacy_status_from_output(self, clean_output: str) -> TerminalStatus:
        """Legacy interactive Gemini status parsing kept for compatibility."""
        tail_output = "\n".join(clean_output.splitlines()[-25:])

        if re.search(WAITING_PROMPT_PATTERN, clean_output, re.IGNORECASE):
            return TerminalStatus.WAITING_USER_ANSWER

        last_user = None
        for match in re.finditer(USER_PREFIX_PATTERN, clean_output, re.MULTILINE):
            last_user = match

        output_after_last_user = clean_output[last_user.start() :] if last_user else clean_output
        assistant_after_last_user = bool(
            last_user
            and re.search(
                ASSISTANT_PREFIX_PATTERN, output_after_last_user, re.MULTILINE | re.DOTALL
            )
        )
        has_idle_prompt_at_end = bool(
            re.search(IDLE_PROMPT_AT_END_PATTERN, clean_output, re.IGNORECASE | re.DOTALL)
        )

        if last_user is not None:
            if re.search(ERROR_PREFIX_PATTERN, output_after_last_user, re.MULTILINE):
                return TerminalStatus.ERROR
            if re.search(PROCESSING_PATTERN, output_after_last_user, re.IGNORECASE):
                return TerminalStatus.PROCESSING
        else:
            if re.search(ERROR_PREFIX_PATTERN, tail_output, re.MULTILINE):
                return TerminalStatus.ERROR
            if re.search(PROCESSING_PATTERN, tail_output, re.IGNORECASE):
                return TerminalStatus.PROCESSING

        if has_idle_prompt_at_end:
            if last_user is not None and assistant_after_last_user:
                return TerminalStatus.COMPLETED
            return TerminalStatus.IDLE

        return TerminalStatus.PROCESSING

    def get_status(self, tail_lines: Optional[int] = None) -> TerminalStatus:
        """Get Gemini status by analyzing terminal output."""
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
        if has_idle_at_end:
            if re.search(ERROR_LINE_PATTERN, clean_output, re.MULTILINE):
                try:
                    if self._last_message_path.exists():
                        err_text = self._last_message_path.read_text(
                            encoding="utf-8", errors="replace"
                        )
                        err_text = re.sub(ANSI_CODE_PATTERN, "", err_text)
                        if re.search(
                            WAITING_PROMPT_PATTERN, err_text, re.IGNORECASE | re.MULTILINE
                        ):
                            return TerminalStatus.WAITING_USER_ANSWER
                except Exception:
                    pass
                return TerminalStatus.ERROR
            if re.search(DONE_LINE_PATTERN, clean_output, re.MULTILINE):
                return TerminalStatus.COMPLETED
            return TerminalStatus.IDLE

        # Backward-compatible parser for legacy interactive mode output.
        return self._legacy_status_from_output(clean_output)

    def get_idle_pattern_for_log(self) -> str:
        """Return Gemini IDLE marker pattern for log files."""
        return IDLE_MARKER

    def extract_last_message_from_script(self, script_output: str) -> str:
        """Extract Gemini final response from provider output file."""
        path = self._last_message_path
        if not path.exists():
            raise ValueError("No Gemini response found - output file missing")

        data = path.read_text(encoding="utf-8", errors="replace")
        data = re.sub(ANSI_CODE_PATTERN, "", data).replace("\u200b", "").strip()
        if not data:
            raise ValueError("Empty Gemini response - no content found")

        model_lines = re.findall(r"^Model:[ \t]*(.*)$", data, re.MULTILINE)
        if model_lines:
            final_answer = model_lines[-1].strip()
            if final_answer:
                return final_answer

        lines = [line.strip() for line in data.splitlines() if line.strip()]
        for line in reversed(lines):
            if line.startswith("Loaded cached credentials."):
                continue
            if line.startswith("Skill conflict detected:"):
                continue
            if re.match(r"^Type your message\b", line):
                continue
            if re.match(r"^User:\s*", line):
                continue
            if re.match(r"^Model:\s*$", line):
                continue
            return line

        raise ValueError("Empty Gemini response - no usable content found")

    def exit_cli(self) -> str:
        """Exit the provider session."""
        return "exit"

    def cleanup(self) -> None:
        """Clean up Gemini provider."""
        self._initialized = False
