"""Claude Code provider implementation.

This provider intentionally avoids driving the interactive Claude Code TUI. In practice, the
interactive `claude` command frequently enters setup flows (workspace trust, model selection,
etc.) that block automation. Instead, we run Claude Code in one-shot print mode (`claude -p`)
and use stable CAO marker lines to detect readiness/completion.
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

# Stable CAO markers appended by this provider (not emitted by Claude itself).
IDLE_MARKER = "CAO_CLAUDE_CODE_IDLE"
DONE_MARKER = "CAO_CLAUDE_CODE_DONE"
ERROR_MARKER = "CAO_CLAUDE_CODE_ERROR"

# Match markers as full lines.
IDLE_LINE_PATTERN = rf"^{re.escape(IDLE_MARKER)}$"
IDLE_AT_END_PATTERN = rf"(?:{IDLE_LINE_PATTERN})\s*\Z"
DONE_LINE_PATTERN = rf"^{re.escape(DONE_MARKER)}$"
ERROR_LINE_PATTERN = rf"^{re.escape(ERROR_MARKER)}$"

# Best-effort detection when the CLI cannot proceed without user intervention.
# This includes both "needs interactive auth" (login/setup-token) and "account blocked" states
# (quota/credits/subscription). For smoke tests we treat these as actionable user states rather
# than provider failures.
WAITING_PROMPT_PATTERN = (
    r"(?:setup-token|log\s*in|login|authenticate|subscription|billing|"
    r"hit\s+your\s+limit|insufficient\s+credits|no\s+credits|quota)"
)


class ClaudeCodeProvider(BaseProvider):
    """Provider for Claude Code CLI tool integration (non-interactive `claude -p`)."""

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

        state_dir = CAO_HOME_DIR / "providers" / "claude_code" / terminal_id
        state_dir.mkdir(parents=True, exist_ok=True)
        self._state_dir = state_dir
        self._last_message_path = state_dir / "last_message.txt"
        self._system_prompt_path: Optional[Path] = None
        self._mcp_config_path: Optional[Path] = None

        # Load agent profile once and materialize any large config payloads to files.
        # This keeps the tmux command line short, which matters because tmux send-keys
        # intentionally throttles long strings.
        if self._agent_profile is not None:
            profile = load_agent_profile(self._agent_profile)

            if profile.system_prompt:
                p = state_dir / "system_prompt.txt"
                p.write_text(profile.system_prompt, encoding="utf-8")
                self._system_prompt_path = p

            if profile.mcpServers:
                p = state_dir / "mcp_config.json"
                p.write_text(profile.model_dump_json(include={"mcpServers"}), encoding="utf-8")
                self._mcp_config_path = p

    def _resolve_model(self) -> Optional[str]:
        """Resolve Claude model name from env or agent profile frontmatter."""
        model = os.getenv("CAO_CLAUDE_CODE_MODEL")
        if model:
            return model

        if not self._agent_profile:
            return None

        try:
            profile = load_agent_profile(self._agent_profile)
            return profile.model
        except Exception:
            return None

    def _extra_print_args(self) -> list[str]:
        """Optional extra args for `claude -p` supplied via env."""
        raw = os.getenv("CAO_CLAUDE_CODE_PRINT_ARGS") or os.getenv("CAO_CLAUDE_CODE_ARGS")
        if not raw:
            return []
        try:
            return shlex.split(raw)
        except ValueError:
            logger.warning("Invalid CAO_CLAUDE_CODE_PRINT_ARGS; ignoring")
            return []

    def _build_claude_print_command(self) -> str:
        """Build a shell command for `claude -p` without embedding large payloads inline."""
        parts: list[str] = [
            "claude",
            "-p",
            "--input-format",
            "text",
            "--output-format",
            "text",
            "--no-session-persistence",
        ]

        extra_args = self._extra_print_args()

        # Disable built-in tools by default to avoid permission prompts. Users can override via
        # CAO_CLAUDE_CODE_PRINT_ARGS="--tools default --permission-mode dontAsk", etc.
        if "--tools" not in extra_args:
            parts.extend(["--tools", ""])

        model = self._resolve_model()
        if model:
            parts.extend(["--model", model])

        # Add system prompt via command substitution to keep the typed command short.
        if self._system_prompt_path is not None:
            parts.append("--append-system-prompt")
            parts.append(f"\"$(cat {shlex.quote(str(self._system_prompt_path))})\"")

        # MCP config supports JSON files directly; prefer a file path over inline JSON.
        if self._mcp_config_path is not None:
            parts.extend(["--mcp-config", str(self._mcp_config_path)])

        # Append extra args as tokens; we quote them once at render time so users can't
        # smuggle shell syntax via env vars.
        if extra_args:
            parts.extend(extra_args)

        # Quote empty-string arg for --tools (and any other values that may need quoting).
        # We do this at the end because the system prompt token already includes quotes.
        rendered: list[str] = []
        for tok in parts:
            if tok.startswith("\"$(") and tok.endswith(")\""):
                rendered.append(tok)
                continue
            rendered.append(shlex.quote(tok))
        return " ".join(rendered)

    def initialize(self) -> bool:
        """Initialize Claude Code provider by preparing the shell environment."""
        if not wait_for_shell(tmux_client, self.session_name, self.window_name, timeout=10.0):
            raise TimeoutError("Shell initialization timed out after 10 seconds")

        # Ensure deterministic output: remove prompt and print an idle marker.
        setup_cmd = "export PS1=''; export PROMPT_COMMAND=''; echo " + shlex.quote(IDLE_MARKER)
        tmux_client.send_keys(self.session_name, self.window_name, setup_cmd)

        if not wait_until_status(self, TerminalStatus.IDLE, timeout=10.0, polling_interval=0.5):
            raise TimeoutError("Claude Code shell initialization timed out after 10 seconds")

        self._initialized = True
        return True

    def format_input(self, message: str) -> str:
        """Wrap the message as a one-shot `claude -p` invocation."""
        out_path = self._last_message_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the prompt to a temp file so the text doesn't appear in tmux logs/history.
        prompt_path = self._state_dir / f"prompt_{uuid.uuid4().hex}.txt"
        prompt_path.write_text(message, encoding="utf-8", errors="strict")

        claude_cmd = self._build_claude_print_command()

        # Feed prompt via stdin (claude -p supports pipes), capture output to file, and
        # append stable markers for status detection.
        run_cmd = (
            f"rm -f {shlex.quote(str(out_path))} ; "
            f"cat {shlex.quote(str(prompt_path))} | {claude_cmd} "
            f"> {shlex.quote(str(out_path))} 2>&1"
        )

        marker_tail = (
            " ; rc=$? ; "
            f"rm -f {shlex.quote(str(prompt_path))} ; "
            f"if [ \"$rc\" -eq 0 ]; then echo {shlex.quote(DONE_MARKER)} ; "
            f"else echo {shlex.quote(ERROR_MARKER)} ; fi ; "
            f"echo {shlex.quote(IDLE_MARKER)}"
        )

        return run_cmd + marker_tail

    def get_status(self, tail_lines: Optional[int] = None) -> TerminalStatus:
        """Get Claude Code status by analyzing terminal output."""
        output = tmux_client.get_history(self.session_name, self.window_name, tail_lines=tail_lines)
        if not output:
            return TerminalStatus.ERROR

        clean_output = re.sub(ANSI_CODE_PATTERN, "", output)
        tail_output = "\n".join(clean_output.splitlines()[-40:])

        # If we see obvious auth/setup blockers in the captured output, treat as waiting.
        if re.search(WAITING_PROMPT_PATTERN, tail_output, re.IGNORECASE | re.MULTILINE):
            return TerminalStatus.WAITING_USER_ANSWER

        has_idle_at_end = bool(
            re.search(IDLE_AT_END_PATTERN, clean_output, re.MULTILINE | re.DOTALL)
        )
        if not has_idle_at_end:
            return TerminalStatus.PROCESSING

        if re.search(ERROR_LINE_PATTERN, clean_output, re.MULTILINE):
            # If the one-shot run failed due to auth/setup, treat it as WAITING so callers can
            # attach to the tmux session and resolve (e.g. `/login`, `setup-token`, etc.).
            try:
                if self._last_message_path.exists():
                    err_text = self._last_message_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                    err_text = re.sub(ANSI_CODE_PATTERN, "", err_text)
                    if re.search(WAITING_PROMPT_PATTERN, err_text, re.IGNORECASE | re.MULTILINE):
                        return TerminalStatus.WAITING_USER_ANSWER
            except Exception:
                pass

            return TerminalStatus.ERROR
        if re.search(DONE_LINE_PATTERN, clean_output, re.MULTILINE):
            return TerminalStatus.COMPLETED

        return TerminalStatus.IDLE

    def get_idle_pattern_for_log(self) -> str:
        """Return Claude Code IDLE marker pattern for log files."""
        return IDLE_MARKER

    def extract_last_message_from_script(self, script_output: str) -> str:
        """Return the last message from the output file written by `claude -p`."""
        path = self._last_message_path
        if not path.exists():
            raise ValueError("No Claude Code response found - output file missing")

        data = path.read_text(encoding="utf-8", errors="replace")
        data = re.sub(ANSI_CODE_PATTERN, "", data).strip()
        if not data:
            raise ValueError("Empty Claude Code response - output file was empty")

        return data

    def exit_cli(self) -> str:
        """Exit the provider session (shell)."""
        return "exit"

    def cleanup(self) -> None:
        """Clean up Claude Code provider."""
        self._initialized = False
