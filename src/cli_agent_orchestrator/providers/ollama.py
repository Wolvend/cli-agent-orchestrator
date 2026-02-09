"""Ollama CLI provider implementation."""

import logging
import os
import re
import shlex
import subprocess
import time
from typing import Optional

from cli_agent_orchestrator.clients.tmux import tmux_client
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.base import BaseProvider
from cli_agent_orchestrator.utils.agent_profiles import load_agent_profile
from cli_agent_orchestrator.utils.terminal import wait_for_shell

logger = logging.getLogger(__name__)

ANSI_CODE_PATTERN = r"\x1b\[[0-9;]*m"

# Interactive prompt for `ollama run` sessions.
# Newer Ollama versions render an initial help prompt on the same line as the
# prompt (e.g. `>>> Send a message (/? for help)`), while older versions render
# just `>>>`.
IDLE_PROMPT_PATTERN = r"^>>>\s*(?:Send a message\s*\(/\?\s+for help\))?\s*$"

# Experimental tool loop prompts / approvals (best-effort).
WAITING_PROMPT_PATTERN = r"^(?:Approve|Allow|Use tool)\b.*\b(?:y/n|yes/no|yes|no)\b"

ERROR_PATTERN = r"^Error:\s+"


class OllamaProvider(BaseProvider):
    """Provider for Ollama CLI tool integration."""

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

    def _resolve_model(self) -> str:
        """Resolve Ollama model name from env or agent profile frontmatter."""
        model = os.getenv("CAO_OLLAMA_MODEL")
        if model:
            return model

        if self._agent_profile:
            try:
                profile = load_agent_profile(self._agent_profile)
                if profile.model:
                    return profile.model
            except Exception:
                # Agent profile is optional for this provider.
                pass

        # Sensible default; user can override with CAO_OLLAMA_MODEL or profile.model.
        return "llama3.2"

    def initialize(self) -> bool:
        """Initialize Ollama provider by starting `ollama run`."""
        if not wait_for_shell(tmux_client, self.session_name, self.window_name, timeout=10.0):
            raise TimeoutError("Shell initialization timed out after 10 seconds")

        model = self._resolve_model()

        # Fail fast if the Ollama daemon isn't running (common on Linux/WSL where
        # users need to run `ollama serve` themselves).
        #
        # Using the CLI here is the most reliable way to respect OLLAMA_HOST and
        # other Ollama env vars without re-implementing URL parsing.
        health_timeout_s = float(os.getenv("CAO_OLLAMA_HEALTHCHECK_TIMEOUT", "5"))
        try:
            probe = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=health_timeout_s,
                check=False,
                env=os.environ.copy(),
            )
        except FileNotFoundError as e:
            raise RuntimeError("Ollama CLI not found in PATH (expected `ollama`)") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                "Ollama healthcheck timed out. Is the server running? "
                "Try starting it with `ollama serve` or set OLLAMA_HOST."
            ) from e

        if probe.returncode != 0:
            err = (probe.stderr or probe.stdout or "").strip()
            first = err.splitlines()[0] if err else "unknown error"
            raise RuntimeError(
                f"Ollama healthcheck failed: {first}\n"
                "Tip: on Linux/WSL you typically need to run `ollama serve` first."
            )

        # Avoid writing readline history under /home in sandboxed environments.
        env_prefix = "OLLAMA_NOHISTORY=1"
        command = f"{env_prefix} {shlex.join(['ollama', 'run', model])}"

        tmux_client.send_keys(self.session_name, self.window_name, command)

        init_timeout_s = float(os.getenv("CAO_OLLAMA_INIT_TIMEOUT", "180"))
        deadline = time.time() + init_timeout_s
        while time.time() < deadline:
            status = self.get_status(tail_lines=200)

            if status == TerminalStatus.IDLE:
                self._initialized = True
                return True

            if status == TerminalStatus.ERROR:
                # Pull a small tail and surface the last Error line.
                raw = tmux_client.get_history(self.session_name, self.window_name, tail_lines=80)
                clean = re.sub(ANSI_CODE_PATTERN, "", raw)
                err_lines = [
                    ln.strip() for ln in clean.splitlines() if ln.strip().startswith("Error:")
                ]
                err = err_lines[-1] if err_lines else "Error: (unknown)"
                raise RuntimeError(
                    f"Ollama CLI error during startup: {err}\n"
                    "Tip: ensure the Ollama server is reachable (e.g. `ollama serve` "
                    "on Linux/WSL, or set OLLAMA_HOST for a remote daemon)."
                )

            if status == TerminalStatus.WAITING_USER_ANSWER:
                # Consider the provider initialized if it is waiting for interactive input.
                self._initialized = True
                return True

            time.sleep(1.0)

        raise TimeoutError(
            f"Ollama initialization timed out after {init_timeout_s:g} seconds "
            f"(model={model}). Consider increasing CAO_OLLAMA_INIT_TIMEOUT."
        )

    def get_status(self, tail_lines: Optional[int] = None) -> TerminalStatus:
        """Get Ollama status by analyzing terminal output."""
        output = tmux_client.get_history(self.session_name, self.window_name, tail_lines=tail_lines)

        if not output:
            return TerminalStatus.ERROR

        clean_output = re.sub(ANSI_CODE_PATTERN, "", output)
        lines = clean_output.splitlines()
        if not lines:
            return TerminalStatus.ERROR

        # Locate prompt lines and classify them as "idle prompts" vs "user prompts".
        prompt_indices: list[int] = []
        user_prompt_indices: list[int] = []
        for idx, line in enumerate(lines):
            if not line.startswith(">>>"):
                continue
            prompt_indices.append(idx)
            if not re.match(IDLE_PROMPT_PATTERN, line):
                user_prompt_indices.append(idx)

        last_user_idx = user_prompt_indices[-1] if user_prompt_indices else None

        # Errors/approval prompts should be considered actionable only after the last user input.
        if last_user_idx is not None:
            output_after_last_user = "\n".join(lines[last_user_idx + 1 :])
            if re.search(ERROR_PATTERN, output_after_last_user, re.MULTILINE):
                return TerminalStatus.ERROR
            if re.search(
                WAITING_PROMPT_PATTERN, output_after_last_user, re.IGNORECASE | re.MULTILINE
            ):
                return TerminalStatus.WAITING_USER_ANSWER
        else:
            tail_output = "\n".join(lines[-25:])
            if re.search(ERROR_PATTERN, tail_output, re.MULTILINE):
                return TerminalStatus.ERROR
            if re.search(WAITING_PROMPT_PATTERN, tail_output, re.IGNORECASE | re.MULTILINE):
                return TerminalStatus.WAITING_USER_ANSWER

        # Determine whether we are at an idle prompt at the end.
        last_nonempty_idx = None
        for idx in range(len(lines) - 1, -1, -1):
            if lines[idx].strip():
                last_nonempty_idx = idx
                break

        if last_nonempty_idx is None:
            return TerminalStatus.ERROR

        last_line = lines[last_nonempty_idx]
        if not (last_line.startswith(">>>") and re.match(IDLE_PROMPT_PATTERN, last_line)):
            return TerminalStatus.PROCESSING

        # If idle prompt is present, decide whether we completed a request.
        if last_user_idx is None:
            return TerminalStatus.IDLE

        assistant_block = "\n".join(lines[last_user_idx + 1 : last_nonempty_idx])
        return TerminalStatus.COMPLETED if assistant_block.strip() else TerminalStatus.IDLE

    def get_idle_pattern_for_log(self) -> str:
        """Return Ollama IDLE prompt pattern for log files."""
        return ">>>"

    def extract_last_message_from_script(self, script_output: str) -> str:
        """Extract Ollama's final response message (text between last user prompt and final >>>)."""
        clean_output = re.sub(ANSI_CODE_PATTERN, "", script_output)
        lines = clean_output.splitlines()

        user_prompt_indices: list[int] = []
        for idx, line in enumerate(lines):
            if not line.startswith(">>>"):
                continue
            if not re.match(IDLE_PROMPT_PATTERN, line):
                user_prompt_indices.append(idx)

        if not user_prompt_indices:
            raise ValueError("No Ollama response found - no user prompt detected")

        last_user_idx = user_prompt_indices[-1]

        last_idle_idx = None
        for idx in range(len(lines) - 1, last_user_idx, -1):
            line = lines[idx]
            if line.startswith(">>>") and re.match(IDLE_PROMPT_PATTERN, line):
                last_idle_idx = idx
                break

        end_idx = last_idle_idx if last_idle_idx is not None else len(lines)
        final_answer = "\n".join(lines[last_user_idx + 1 : end_idx]).strip()
        if not final_answer:
            raise ValueError("Empty Ollama response - no content found")

        return final_answer

    def exit_cli(self) -> str:
        """Get the command to exit Ollama REPL."""
        # Best-effort; if this doesn't exit in a given Ollama version, send Ctrl+C manually.
        return "/bye"

    def cleanup(self) -> None:
        """Clean up Ollama CLI provider."""
        self._initialized = False
