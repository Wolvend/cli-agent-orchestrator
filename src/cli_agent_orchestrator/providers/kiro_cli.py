"""Kiro CLI provider implementation."""

import logging
import re
import time
from typing import Optional

from cli_agent_orchestrator.clients.tmux import tmux_client
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.base import BaseProvider
from cli_agent_orchestrator.utils.terminal import wait_for_shell

logger = logging.getLogger(__name__)

# Regex patterns for Kiro CLI output analysis (module-level constants)
GREEN_ARROW_PATTERN = r"^>\s*"  # Pattern for ANSI-cleaned output (start of line)
ANSI_CODE_PATTERN = r"\x1b\[[0-9;]*m"
ESCAPE_SEQUENCE_PATTERN = r"\[[?0-9;]*[a-zA-Z]"
CONTROL_CHAR_PATTERN = r"[\x00-\x1f\x7f-\x9f]"
BELL_CHAR = "\x07"
IDLE_PROMPT_PATTERN_LOG = r"\x1b\[38;5;\d+m\[.+?\].*\x1b\[38;5;\d+m>\s*\x1b\[\d*m"

# Error indicators
ERROR_INDICATORS = [
    "Kiro is having trouble responding right now",
    "dispatch failure",
]

# Device-code login prompt shown when the CLI needs the user to authenticate in a browser.
# NOTE: During auth the CLI can spam progress lines (e.g. "Logging in...") which may push the
# initial URL/code out of the last N captured tmux lines. Include a stable progress marker so
# we still surface WAITING_USER_ANSWER.
LOGIN_PROMPT_PATTERN = (
    r"(?:Confirm the following code in the browser|Open this URL:|user_code=|"
    r"view\.awsapps\.com/start/#/device|Logging in\.{3,}|"
    r"let's get you signed in|Press enter to continue to the browser|"
    r"press enter to continue|press enter|esc to cancel)"
)


def _redact_auth_output(text: str) -> str:
    """Best-effort redaction for device-code auth flows (avoid leaking live codes into logs)."""
    text = re.sub(r"(user_code=)[A-Za-z0-9-]+", r"\1REDACTED", text)
    text = re.sub(
        r"(Open this URL:\s*)https?://\\S+",
        r"\1REDACTED_URL",
        text,
        flags=re.IGNORECASE,
    )
    return text


class KiroCliProvider(BaseProvider):
    """Provider for Kiro CLI tool integration."""

    def __init__(self, terminal_id: str, session_name: str, window_name: str, agent_profile: str):
        super().__init__(terminal_id, session_name, window_name)
        self._initialized = False
        self._agent_profile = agent_profile
        # Create dynamic prompt pattern based on agent profile (ANSI-free)
        # Matches: [agent] !> or [agent] > or [agent] X% > or [agent] λ > or [agent] X% λ >
        # after ANSI codes are stripped
        # Also matches with trailing text like "How can I help?"
        self._idle_prompt_pattern = (
            rf"\[{re.escape(self._agent_profile)}\]\s*(?:\d+%\s*)?(?:\u03bb\s*)?!?>\s*"
        )
        self._permission_prompt_pattern = r"Allow this action\?.*?\[.*?y.*?/.*?n.*?/.*?t.*?\]:"

    def initialize(self) -> bool:
        """Initialize Kiro CLI provider by starting kiro-cli chat command."""
        # Wait for shell to be ready first
        if not wait_for_shell(tmux_client, self.session_name, self.window_name, timeout=10.0):
            raise TimeoutError("Shell initialization timed out after 10 seconds")

        command = f"kiro-cli chat --agent {self._agent_profile}"
        tmux_client.send_keys(self.session_name, self.window_name, command)

        # Kiro CLI may enter interactive setup flows (auth, permission prompts, etc.).
        # Treat WAITING_USER_ANSWER as a successful init so the user can attach and proceed.
        deadline = time.time() + 30.0
        while time.time() < deadline:
            status = self.get_status(tail_lines=200)
            if status in (TerminalStatus.IDLE, TerminalStatus.WAITING_USER_ANSWER):
                self._initialized = True
                return True
            if status == TerminalStatus.ERROR:
                raise RuntimeError("Kiro CLI entered ERROR state during initialization")
            time.sleep(1.0)

        raw = tmux_client.get_history(self.session_name, self.window_name, tail_lines=200)
        clean = re.sub(ANSI_CODE_PATTERN, "", raw or "")
        clean = _redact_auth_output(clean)
        snippet_lines = [ln for ln in clean.splitlines() if ln.strip()]
        snippet = "\n".join(snippet_lines[-40:]) if snippet_lines else "(no output captured)"
        raise TimeoutError(
            f"Kiro CLI initialization timed out after 30 seconds\n--- last output ---\n{snippet}"
        )

        # Unreachable

    def get_status(self, tail_lines: Optional[int] = None) -> TerminalStatus:
        """Get Kiro CLI status by analyzing terminal output."""
        logger.debug(f"get_status: tail_lines={tail_lines}")
        output = tmux_client.get_history(self.session_name, self.window_name, tail_lines=tail_lines)

        if not output:
            return TerminalStatus.ERROR

        # Strip ANSI codes once for all pattern matching
        clean_output = re.sub(ANSI_CODE_PATTERN, "", output)

        # Check for error indicators
        if any(indicator.lower() in clean_output.lower() for indicator in ERROR_INDICATORS):
            return TerminalStatus.ERROR

        # Check for permission prompt — count lines with idle prompt after last [y/n/t]:
        # Active prompt: 0-1 lines with idle prompt (CLI renders prompt on next line)
        # Stale prompt: 2+ lines with idle prompt (user answered, agent continued)
        # Line-based counting handles \r redraws (same line, no \n) correctly
        perm_matches = list(re.finditer(self._permission_prompt_pattern, clean_output, re.DOTALL))
        if perm_matches:
            after_last_perm = clean_output[perm_matches[-1].end() :]
            lines_after = after_last_perm.split("\n")
            idle_lines = sum(
                1 for line in lines_after if re.search(self._idle_prompt_pattern, line)
            )
            if idle_lines <= 1:
                return TerminalStatus.WAITING_USER_ANSWER

        # Check for completed state (has response + agent prompt AFTER the response)
        green_arrows = list(re.finditer(GREEN_ARROW_PATTERN, clean_output, re.MULTILINE))
        if green_arrows:
            # Find if there's an idle prompt after the last green arrow
            last_arrow_pos = green_arrows[-1].end()
            idle_prompts = list(re.finditer(self._idle_prompt_pattern, clean_output))

            for prompt in idle_prompts:
                if prompt.start() > last_arrow_pos:
                    logger.debug(f"get_status: returning COMPLETED")
                    return TerminalStatus.COMPLETED

            # Has green arrow but no prompt after it - still processing
            return TerminalStatus.PROCESSING

        # No response arrow yet; if the prompt is visible the terminal is idle/ready.
        if re.search(self._idle_prompt_pattern, clean_output):
            return TerminalStatus.IDLE

        # Device-code login flow requires user to authenticate in a browser.
        if re.search(LOGIN_PROMPT_PATTERN, clean_output, re.IGNORECASE):
            return TerminalStatus.WAITING_USER_ANSWER

        return TerminalStatus.PROCESSING

    def extract_last_message_from_script(self, script_output: str) -> str:
        """Extract agent's final response message using green arrow indicator."""
        # Strip ANSI codes for pattern matching
        clean_output = re.sub(ANSI_CODE_PATTERN, "", script_output)

        # Find patterns in clean output
        green_arrows = list(re.finditer(GREEN_ARROW_PATTERN, clean_output, re.MULTILINE))
        idle_prompts = list(re.finditer(self._idle_prompt_pattern, clean_output))

        if not green_arrows:
            raise ValueError("No Kiro CLI response found - no green arrow pattern detected")

        if not idle_prompts:
            raise ValueError("Incomplete Kiro CLI response - no final prompt detected")

        # Find the last green arrow (response start)
        last_arrow_pos = green_arrows[-1].end()

        # Find idle prompt that comes AFTER the last green arrow
        final_prompt = None
        for prompt in idle_prompts:
            if prompt.start() > last_arrow_pos:
                final_prompt = prompt
                break

        if not final_prompt:
            raise ValueError(
                "Incomplete Kiro CLI response - no final prompt detected after response"
            )

        # Extract directly from clean output
        start_pos = last_arrow_pos
        end_pos = final_prompt.start()

        final_answer = clean_output[start_pos:end_pos].strip()

        if not final_answer:
            raise ValueError("Empty Kiro CLI response - no content found")

        # Clean up the message
        final_answer = re.sub(ANSI_CODE_PATTERN, "", final_answer)
        final_answer = re.sub(ESCAPE_SEQUENCE_PATTERN, "", final_answer)
        final_answer = re.sub(CONTROL_CHAR_PATTERN, "", final_answer)
        return final_answer.strip()

    def get_idle_pattern_for_log(self) -> str:
        """Return Kiro CLI IDLE prompt pattern for log files."""
        return IDLE_PROMPT_PATTERN_LOG

    def exit_cli(self) -> str:
        """Get the command to exit Kiro CLI."""
        return "/exit"

    def cleanup(self) -> None:
        """Clean up Kiro CLI provider."""
        self._initialized = False
