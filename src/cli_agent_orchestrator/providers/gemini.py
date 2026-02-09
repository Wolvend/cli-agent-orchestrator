"""Gemini CLI provider implementation."""

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

# Gemini CLI (screen reader mode) markers.
ANSI_CODE_PATTERN = r"\x1b\[[0-9;]*m"
# In `--output-format text` mode, Gemini prints conversation lines as `User:` / `Model:`.
# Note: The interactive input placeholder also starts with `>`, so we avoid using that for
# "last user message" detection.
USER_PREFIX_PATTERN = r"^User:\s+"
ASSISTANT_PREFIX_PATTERN = r"^Model:\s+"
ERROR_PREFIX_PATTERN = r"^(?:✕\s+|Error:)"

# In screen reader mode Gemini renders "responding"/"loading" as text alternatives for the spinner.
PROCESSING_PATTERN = r"\b(?:responding|loading)\b"

# Gemini setup flows that require user input (API key, model selection, etc.).
WAITING_PROMPT_PATTERN = r"(?:API\s*key|Gemini\s*API\s*Key|Paste\s+your\s+API\s+key|Select\s+Model)"

# Input placeholder line shown when ready for input.
IDLE_PROMPT_TEXT_PATTERN = r"Type your message"
# Match prompt only if it appears at the end of the captured output.
IDLE_PROMPT_AT_END_PATTERN = rf"(?:{IDLE_PROMPT_TEXT_PATTERN}.*)\s*\Z"


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
            # Agent profile is optional for this provider.
            return None

    def _gemini_home_dir(self) -> Path:
        """Return a writable HOME for Gemini CLI so it doesn't write under /home in sandbox."""
        return CAO_HOME_DIR / "providers" / "gemini-home"

    def initialize(self) -> bool:
        """Initialize Gemini provider by starting gemini command."""
        if not wait_for_shell(tmux_client, self.session_name, self.window_name, timeout=10.0):
            raise TimeoutError("Shell initialization timed out after 10 seconds")

        gemini_home = self._gemini_home_dir()
        gemini_home.mkdir(parents=True, exist_ok=True)

        command_parts: list[str] = [
            "gemini",
            "--screen-reader",
            "--output-format",
            "text",
        ]

        model = self._resolve_model()
        if model:
            command_parts.extend(["--model", model])

        env_prefix = f"HOME={shlex.quote(str(gemini_home))}"
        command = f"{env_prefix} {shlex.join(command_parts)}"

        tmux_client.send_keys(self.session_name, self.window_name, command)

        # Gemini may require interactive setup (API key / model selection). Treat that as
        # successful initialization so the user can attach and complete setup.
        if not wait_until_status(self, TerminalStatus.IDLE, timeout=60.0, polling_interval=1.0):
            status = self.get_status()
            if status != TerminalStatus.WAITING_USER_ANSWER:
                raise TimeoutError("Gemini initialization timed out after 60 seconds")

        self._initialized = True
        return True

    def get_status(self, tail_lines: Optional[int] = None) -> TerminalStatus:
        """Get Gemini status by analyzing terminal output."""
        output = tmux_client.get_history(self.session_name, self.window_name, tail_lines=tail_lines)

        if not output:
            return TerminalStatus.ERROR

        clean_output = re.sub(ANSI_CODE_PATTERN, "", output)
        tail_output = "\n".join(clean_output.splitlines()[-25:])

        # Setup prompts can appear even before any user message.
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

        # Only consider errors/processing as actionable if they occur after the last user message.
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
            # Consider COMPLETED only if we see an assistant marker after the last user message.
            if last_user is not None and assistant_after_last_user:
                return TerminalStatus.COMPLETED
            return TerminalStatus.IDLE

        # If we're not at an idle prompt and we don't see explicit errors/setup prompts,
        # assume the CLI is still producing output.
        return TerminalStatus.PROCESSING

    def get_idle_pattern_for_log(self) -> str:
        """Return Gemini IDLE prompt pattern for log files."""
        return IDLE_PROMPT_TEXT_PATTERN

    def extract_last_message_from_script(self, script_output: str) -> str:
        """Extract Gemini's final response message from the last `Model:` line."""
        clean_output = re.sub(ANSI_CODE_PATTERN, "", script_output)

        # In screen-reader mode, Gemini refreshes the UI and repeats `Model:` lines; the last one
        # is the most reliable "final answer" signal.
        # IMPORTANT: Do not allow `\\s` here; it includes newlines and can accidentally capture the
        # next UI line when the model response is blank.
        model_lines = re.findall(r"^Model:[ \t]*(.*)$", clean_output, re.MULTILINE)
        if not model_lines:
            raise ValueError("No Gemini response found - no 'Model:' line detected")

        final_answer = model_lines[-1].strip()
        if not final_answer:
            raise ValueError("Empty Gemini response - no content found")

        return final_answer

    def exit_cli(self) -> str:
        """Get the command to exit Gemini CLI."""
        return "/quit"

    def cleanup(self) -> None:
        """Clean up Gemini CLI provider."""
        self._initialized = False
