"""Unit tests for Claude Code provider."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.claude_code import ClaudeCodeProvider

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(filename: str) -> str:
    with open(FIXTURES_DIR / filename, "r") as f:
        return f.read()


class TestClaudeCodeProviderInitialization:
    @patch("cli_agent_orchestrator.providers.claude_code.wait_until_status")
    @patch("cli_agent_orchestrator.providers.claude_code.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.claude_code.tmux_client")
    def test_initialize_success(self, mock_tmux, mock_wait_shell, mock_wait_status):
        mock_wait_shell.return_value = True
        mock_wait_status.return_value = True

        provider = ClaudeCodeProvider("test1234", "test-session", "window-0", None)
        result = provider.initialize()

        assert result is True
        mock_wait_shell.assert_called_once()
        assert mock_tmux.send_keys.call_count == 1
        (session, window, cmd) = mock_tmux.send_keys.call_args.args
        assert session == "test-session"
        assert window == "window-0"
        assert "CAO_CLAUDE_CODE_IDLE" in cmd
        mock_wait_status.assert_called_once()

    @patch("cli_agent_orchestrator.providers.claude_code.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.claude_code.tmux_client")
    def test_initialize_shell_timeout(self, mock_tmux, mock_wait_shell):
        mock_wait_shell.return_value = False

        provider = ClaudeCodeProvider("test1234", "test-session", "window-0", None)

        with pytest.raises(TimeoutError, match="Shell initialization timed out"):
            provider.initialize()

    @patch("cli_agent_orchestrator.providers.claude_code.wait_until_status")
    @patch("cli_agent_orchestrator.providers.claude_code.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.claude_code.tmux_client")
    def test_initialize_timeout(self, mock_tmux, mock_wait_shell, mock_wait_status):
        mock_wait_shell.return_value = True
        mock_wait_status.return_value = False

        provider = ClaudeCodeProvider("test1234", "test-session", "window-0", None)

        with pytest.raises(TimeoutError, match="Claude Code shell initialization timed out"):
            provider.initialize()


class TestClaudeCodeProviderStatusDetection:
    @patch("cli_agent_orchestrator.providers.claude_code.tmux_client")
    def test_get_status_idle(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("claude_code_idle_output.txt")

        provider = ClaudeCodeProvider("test1234", "test-session", "window-0")
        status = provider.get_status()

        assert status == TerminalStatus.IDLE

    @patch("cli_agent_orchestrator.providers.claude_code.tmux_client")
    def test_get_status_completed(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("claude_code_completed_output.txt")

        provider = ClaudeCodeProvider("test1234", "test-session", "window-0")
        status = provider.get_status()

        assert status == TerminalStatus.COMPLETED

    @patch("cli_agent_orchestrator.providers.claude_code.tmux_client")
    def test_get_status_processing(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("claude_code_processing_output.txt")

        provider = ClaudeCodeProvider("test1234", "test-session", "window-0")
        status = provider.get_status()

        assert status == TerminalStatus.PROCESSING

    @patch("cli_agent_orchestrator.providers.claude_code.tmux_client")
    def test_get_status_error(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("claude_code_error_output.txt")

        provider = ClaudeCodeProvider("test1234", "test-session", "window-0")
        status = provider.get_status()

        assert status == TerminalStatus.ERROR

    @patch("cli_agent_orchestrator.providers.claude_code.tmux_client")
    def test_get_status_waiting_user_answer(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("claude_code_waiting_output.txt")

        provider = ClaudeCodeProvider("test1234", "test-session", "window-0")
        status = provider.get_status()

        assert status == TerminalStatus.WAITING_USER_ANSWER

    @patch("cli_agent_orchestrator.providers.claude_code.tmux_client")
    def test_get_status_waiting_user_answer_from_output_file(self, mock_tmux, tmp_path):
        # Error marker present, but the underlying CLI error indicates auth/setup is required.
        mock_tmux.get_history.return_value = load_fixture("claude_code_error_output.txt")

        provider = ClaudeCodeProvider("test1234", "test-session", "window-0")
        provider._last_message_path = tmp_path / "last_message.txt"
        provider._last_message_path.write_text(
            "Your account does not have access to Claude Code. Please run /login.\n",
            encoding="utf-8",
        )

        status = provider.get_status()
        assert status == TerminalStatus.WAITING_USER_ANSWER

    @patch("cli_agent_orchestrator.providers.claude_code.tmux_client")
    def test_get_status_waiting_user_answer_from_output_file_quota(self, mock_tmux, tmp_path):
        # Error marker present, and the underlying CLI indicates a quota/credits block.
        mock_tmux.get_history.return_value = load_fixture("claude_code_error_output.txt")

        provider = ClaudeCodeProvider("test1234", "test-session", "window-0")
        provider._last_message_path = tmp_path / "last_message.txt"
        provider._last_message_path.write_text(
            "You've hit your limit · resets 7am (America/Los_Angeles)\n",
            encoding="utf-8",
        )

        status = provider.get_status()
        assert status == TerminalStatus.WAITING_USER_ANSWER

    @patch("cli_agent_orchestrator.providers.claude_code.tmux_client")
    def test_get_status_empty_output(self, mock_tmux):
        mock_tmux.get_history.return_value = ""

        provider = ClaudeCodeProvider("test1234", "test-session", "window-0")
        status = provider.get_status()

        assert status == TerminalStatus.ERROR

    @patch("cli_agent_orchestrator.providers.claude_code.tmux_client")
    def test_get_status_with_tail_lines(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("claude_code_idle_output.txt")

        provider = ClaudeCodeProvider("test1234", "test-session", "window-0")
        status = provider.get_status(tail_lines=50)

        assert status == TerminalStatus.IDLE
        mock_tmux.get_history.assert_called_once_with("test-session", "window-0", tail_lines=50)


class TestClaudeCodeProviderMessageExtraction:
    def test_extract_last_message_success(self, tmp_path):
        provider = ClaudeCodeProvider("test1234", "test-session", "window-0")
        provider._last_message_path = tmp_path / "last_message.txt"
        provider._last_message_path.write_text("\x1b[31m4\x1b[0m\n", encoding="utf-8")

        message = provider.extract_last_message_from_script("irrelevant")
        assert message == "4"

    def test_extract_message_missing_output_file(self, tmp_path):
        provider = ClaudeCodeProvider("test1234", "test-session", "window-0")
        provider._last_message_path = tmp_path / "missing.txt"

        with pytest.raises(ValueError, match="No Claude Code response found"):
            provider.extract_last_message_from_script("irrelevant")

    def test_extract_message_empty_output_file(self, tmp_path):
        provider = ClaudeCodeProvider("test1234", "test-session", "window-0")
        provider._last_message_path = tmp_path / "empty.txt"
        provider._last_message_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="Empty Claude Code response"):
            provider.extract_last_message_from_script("irrelevant")


class TestClaudeCodeProviderFormatInput:
    def test_format_input_writes_prompt_file_and_builds_command(self, tmp_path):
        provider = ClaudeCodeProvider("test1234", "test-session", "window-0")
        provider._state_dir = tmp_path
        provider._last_message_path = tmp_path / "last_message.txt"

        with patch(
            "cli_agent_orchestrator.providers.claude_code.uuid.uuid4",
            return_value=SimpleNamespace(hex="abc123"),
        ):
            cmd = provider.format_input("hello\nworld")

        prompt_path = tmp_path / "prompt_abc123.txt"
        assert prompt_path.exists()
        assert prompt_path.read_text(encoding="utf-8") == "hello\nworld"

        # Ensure the prompt is not in the shell command (only the file path).
        assert "hello" not in cmd
        assert "world" not in cmd
        assert f"cat {shlex_quote(str(prompt_path))}" in cmd
        assert str(provider._last_message_path) in cmd


def shlex_quote(s: str) -> str:
    # Local helper to match the provider's quoting behavior in assertions.
    import shlex

    return shlex.quote(s)


class TestClaudeCodeProviderMisc:
    def test_get_idle_pattern_for_log(self):
        provider = ClaudeCodeProvider("test1234", "test-session", "window-0")
        assert provider.get_idle_pattern_for_log() == "CAO_CLAUDE_CODE_IDLE"

    def test_exit_cli(self):
        provider = ClaudeCodeProvider("test1234", "test-session", "window-0")
        assert provider.exit_cli() == "exit"

    def test_cleanup(self):
        provider = ClaudeCodeProvider("test1234", "test-session", "window-0")
        provider._initialized = True
        provider.cleanup()
        assert provider._initialized is False
