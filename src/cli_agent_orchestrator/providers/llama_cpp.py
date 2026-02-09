"""llama.cpp (llama-cli) provider implementation."""

import logging
import os
import re
import shlex
import shutil
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

# llama-cli prompt is printed as "\n> " when waiting for input.
IDLE_PROMPT_PATTERN = r"^>\s*$"
IDLE_PROMPT_AT_END_PATTERN = rf"(?:{IDLE_PROMPT_PATTERN})\s*\Z"
USER_PROMPT_WITH_TEXT_PATTERN = r"^>\s+\S.+$"

ERROR_PATTERN = r"^(?:Error:|Failed to load the model|invalid argument:|terminate called|Segmentation fault)\b"


class LlamaCppProvider(BaseProvider):
    """Provider for llama.cpp `llama-cli`."""

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

    def _llama_home_dir(self) -> Path:
        """Return a writable HOME for llama.cpp cache/config."""
        return CAO_HOME_DIR / "providers" / "llama-cpp-home"

    def _resolve_llama_cli_bin(self) -> str:
        """Resolve llama-cli binary path from env, PATH, or common local checkout locations."""
        bin_path = os.getenv("CAO_LLAMA_CPP_BIN")
        if bin_path:
            return bin_path

        which = shutil.which("llama-cli")
        if which:
            return which

        # Best-effort: if the terminal was launched from a repo root, try common relative locations.
        pane_dir = tmux_client.get_pane_working_directory(self.session_name, self.window_name)
        candidates: list[Path] = []
        if pane_dir:
            base = Path(pane_dir)
            candidates.extend(
                [
                    base / "llama.cpp" / "build" / "bin" / "llama-cli",
                    base / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli",
                    base / "build" / "bin" / "llama-cli",
                    base / "build" / "bin" / "Release" / "llama-cli",
                ]
            )

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return str(candidate)

        raise FileNotFoundError(
            "Could not find llama-cli. Set CAO_LLAMA_CPP_BIN or add llama-cli to PATH."
        )

    def _resolve_model_spec(self) -> str:
        """Resolve local model path or HF repo spec."""
        model = os.getenv("CAO_LLAMA_CPP_MODEL")
        if model:
            return model

        if self._agent_profile:
            try:
                profile = load_agent_profile(self._agent_profile)
                if profile.model:
                    return profile.model
            except Exception:
                pass

        # Stable default from llama.cpp README. This may download on first run.
        return "ggml-org/gemma-3-1b-it-GGUF"

    def _build_command(self) -> str:
        llama_cli = self._resolve_llama_cli_bin()
        model_spec = self._resolve_model_spec()

        command_parts: list[str] = [
            llama_cli,
            "--simple-io",
            "--no-show-timings",
        ]

        model_path = Path(model_spec).expanduser()
        if model_path.exists():
            command_parts.extend(["-m", str(model_path)])
        else:
            command_parts.extend(["-hf", model_spec])

        # Allow users to pass through additional llama-cli flags (e.g. deterministic settings).
        # Example: CAO_LLAMA_CPP_ARGS="--temp 0 --seed 0"
        extra_args = os.getenv("CAO_LLAMA_CPP_ARGS", "").strip()
        if extra_args:
            try:
                command_parts.extend(shlex.split(extra_args))
            except ValueError as e:
                raise ValueError(f"Invalid CAO_LLAMA_CPP_ARGS: {e}") from e

        llama_home = self._llama_home_dir()
        llama_home.mkdir(parents=True, exist_ok=True)

        env_prefix = f"HOME={shlex.quote(str(llama_home))}"
        return f"{env_prefix} {shlex.join(command_parts)}"

    def initialize(self) -> bool:
        """Initialize llama.cpp provider by starting llama-cli."""
        if not wait_for_shell(tmux_client, self.session_name, self.window_name, timeout=10.0):
            raise TimeoutError("Shell initialization timed out after 10 seconds")

        command = self._build_command()
        tmux_client.send_keys(self.session_name, self.window_name, command)

        # Loading (and potentially downloading) can take a while. Allow override via env.
        init_timeout_s = float(os.getenv("CAO_LLAMA_CPP_INIT_TIMEOUT", "600"))
        if not wait_until_status(
            self, TerminalStatus.IDLE, timeout=init_timeout_s, polling_interval=2.0
        ):
            raise TimeoutError(f"llama-cli initialization timed out after {init_timeout_s:g} seconds")

        self._initialized = True
        return True

    def get_status(self, tail_lines: Optional[int] = None) -> TerminalStatus:
        """Get llama.cpp status by analyzing terminal output."""
        output = tmux_client.get_history(self.session_name, self.window_name, tail_lines=tail_lines)
        if not output:
            return TerminalStatus.ERROR

        clean_output = re.sub(ANSI_CODE_PATTERN, "", output)
        tail_output = "\n".join(clean_output.splitlines()[-40:])

        last_user = None
        for match in re.finditer(USER_PROMPT_WITH_TEXT_PATTERN, clean_output, re.MULTILINE):
            last_user = match

        output_after_last_user = clean_output[last_user.start() :] if last_user else clean_output

        if last_user is not None:
            if re.search(ERROR_PATTERN, output_after_last_user, re.IGNORECASE | re.MULTILINE):
                return TerminalStatus.ERROR
        else:
            if re.search(ERROR_PATTERN, tail_output, re.IGNORECASE | re.MULTILINE):
                return TerminalStatus.ERROR

        has_idle_prompt_at_end = bool(
            re.search(IDLE_PROMPT_AT_END_PATTERN, clean_output, re.MULTILINE | re.DOTALL)
        )
        if not has_idle_prompt_at_end:
            return TerminalStatus.PROCESSING

        if last_user is None:
            return TerminalStatus.IDLE

        idle_prompts = list(re.finditer(IDLE_PROMPT_PATTERN, clean_output, re.MULTILINE))
        last_idle_prompt = idle_prompts[-1] if idle_prompts else None
        end_pos = last_idle_prompt.start() if last_idle_prompt else len(clean_output)

        assistant_block = clean_output[last_user.end() : end_pos]
        if assistant_block.strip():
            return TerminalStatus.COMPLETED

        return TerminalStatus.IDLE

    def get_idle_pattern_for_log(self) -> str:
        """Return a cheap marker that likely appears when llama-cli is ready."""
        return r">"

    def extract_last_message_from_script(self, script_output: str) -> str:
        """Extract the final assistant response (text between last user prompt and the final >)."""
        clean_output = re.sub(ANSI_CODE_PATTERN, "", script_output)

        user_prompts = list(re.finditer(USER_PROMPT_WITH_TEXT_PATTERN, clean_output, re.MULTILINE))
        if not user_prompts:
            raise ValueError("No llama.cpp response found - no user prompt detected")

        last_user = user_prompts[-1]
        start_pos = last_user.end()

        idle_prompts = list(re.finditer(IDLE_PROMPT_PATTERN, clean_output, re.MULTILINE))
        last_idle_prompt = idle_prompts[-1] if idle_prompts else None
        end_pos = last_idle_prompt.start() if last_idle_prompt else len(clean_output)

        final_answer = clean_output[start_pos:end_pos].strip()
        if not final_answer:
            raise ValueError("Empty llama.cpp response - no content found")

        return final_answer

    def exit_cli(self) -> str:
        """Get the command to exit llama-cli."""
        return "/exit"

    def cleanup(self) -> None:
        """Clean up llama.cpp provider."""
        self._initialized = False
