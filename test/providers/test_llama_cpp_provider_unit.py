"""Unit tests for llama.cpp (llama-cli) provider."""

from pathlib import Path
from unittest.mock import patch

import pytest

from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.llama_cpp import LlamaCppProvider

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(filename: str) -> str:
    with open(FIXTURES_DIR / filename, "r") as f:
        return f.read()


class TestLlamaCppProviderInitialization:
    @patch.object(LlamaCppProvider, "_build_command", return_value="llama-cli --version")
    @patch("cli_agent_orchestrator.providers.llama_cpp.wait_until_status")
    @patch("cli_agent_orchestrator.providers.llama_cpp.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.llama_cpp.tmux_client")
    def test_initialize_success(self, mock_tmux, mock_wait_shell, mock_wait_status, _mock_cmd):
        mock_wait_shell.return_value = True
        mock_wait_status.return_value = True

        provider = LlamaCppProvider("test1234", "test-session", "window-0", None)
        result = provider.initialize()

        assert result is True
        mock_wait_shell.assert_called_once()
        assert mock_tmux.send_keys.call_count == 1
        mock_wait_status.assert_called_once()

    @patch.object(LlamaCppProvider, "_build_command", return_value="llama-cli --version")
    @patch("cli_agent_orchestrator.providers.llama_cpp.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.llama_cpp.tmux_client")
    def test_initialize_shell_timeout(self, mock_tmux, mock_wait_shell, _mock_cmd):
        mock_wait_shell.return_value = False

        provider = LlamaCppProvider("test1234", "test-session", "window-0", None)
        with pytest.raises(TimeoutError, match="Shell initialization timed out"):
            provider.initialize()

    @patch.object(LlamaCppProvider, "_build_command", return_value="llama-cli --version")
    @patch("cli_agent_orchestrator.providers.llama_cpp.wait_until_status")
    @patch("cli_agent_orchestrator.providers.llama_cpp.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.llama_cpp.tmux_client")
    def test_initialize_timeout_raises(
        self, mock_tmux, mock_wait_shell, mock_wait_status, _mock_cmd
    ):
        mock_wait_shell.return_value = True
        mock_wait_status.return_value = False

        provider = LlamaCppProvider("test1234", "test-session", "window-0", None)
        with pytest.raises(TimeoutError, match="llama-cli initialization timed out after"):
            provider.initialize()


class TestLlamaCppProviderStatusDetection:
    @patch("cli_agent_orchestrator.providers.llama_cpp.tmux_client")
    def test_get_status_idle(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("llama_cpp_idle_output.txt")

        provider = LlamaCppProvider("test1234", "test-session", "window-0", None)
        assert provider.get_status() == TerminalStatus.IDLE

    @patch("cli_agent_orchestrator.providers.llama_cpp.tmux_client")
    def test_get_status_completed(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("llama_cpp_completed_output.txt")

        provider = LlamaCppProvider("test1234", "test-session", "window-0", None)
        assert provider.get_status() == TerminalStatus.COMPLETED

    @patch("cli_agent_orchestrator.providers.llama_cpp.tmux_client")
    def test_get_status_processing(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("llama_cpp_processing_output.txt")

        provider = LlamaCppProvider("test1234", "test-session", "window-0", None)
        assert provider.get_status() == TerminalStatus.PROCESSING

    @patch("cli_agent_orchestrator.providers.llama_cpp.tmux_client")
    def test_get_status_error(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("llama_cpp_error_output.txt")

        provider = LlamaCppProvider("test1234", "test-session", "window-0", None)
        assert provider.get_status() == TerminalStatus.ERROR

    @patch("cli_agent_orchestrator.providers.llama_cpp.tmux_client")
    def test_get_status_empty_output(self, mock_tmux):
        mock_tmux.get_history.return_value = ""

        provider = LlamaCppProvider("test1234", "test-session", "window-0", None)
        assert provider.get_status() == TerminalStatus.ERROR


class TestLlamaCppProviderMessageExtraction:
    def test_extract_last_message_success(self):
        output = load_fixture("llama_cpp_completed_output.txt")

        provider = LlamaCppProvider("test1234", "test-session", "window-0", None)
        message = provider.extract_last_message_from_script(output)

        assert "2+2 is 4" in message

    def test_extract_last_message_no_prompt(self):
        provider = LlamaCppProvider("test1234", "test-session", "window-0", None)
        with pytest.raises(ValueError, match="No llama\\.cpp response found"):
            provider.extract_last_message_from_script("No prompts here")

    def test_extract_last_message_empty_response(self):
        provider = LlamaCppProvider("test1234", "test-session", "window-0", None)
        with pytest.raises(ValueError, match="Empty llama\\.cpp response"):
            provider.extract_last_message_from_script("> hello\n>\n")


class TestLlamaCppProviderMisc:
    def test_get_idle_pattern_for_log(self):
        provider = LlamaCppProvider("test1234", "test-session", "window-0", None)
        assert provider.get_idle_pattern_for_log() == r">"

    def test_exit_cli(self):
        provider = LlamaCppProvider("test1234", "test-session", "window-0", None)
        assert provider.exit_cli() == "/exit"

    def test_cleanup(self):
        provider = LlamaCppProvider("test1234", "test-session", "window-0", None)
        provider._initialized = True
        provider.cleanup()
        assert provider._initialized is False
