"""Unit tests for Gemini provider."""

from pathlib import Path
from unittest.mock import patch

import pytest

from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.gemini import GeminiProvider

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(filename: str) -> str:
    with open(FIXTURES_DIR / filename, "r") as f:
        return f.read()


class TestGeminiProviderInitialization:
    @patch("cli_agent_orchestrator.providers.gemini.wait_until_status")
    @patch("cli_agent_orchestrator.providers.gemini.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.gemini.tmux_client")
    def test_initialize_success(self, mock_tmux, mock_wait_shell, mock_wait_status, tmp_path):
        mock_wait_shell.return_value = True
        mock_wait_status.return_value = True

        provider = GeminiProvider("test1234", "test-session", "window-0", None)
        result = provider.initialize()

        assert result is True
        mock_wait_shell.assert_called_once()
        assert mock_tmux.send_keys.call_count == 1
        mock_wait_status.assert_called_once()

    @patch("cli_agent_orchestrator.providers.gemini.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.gemini.tmux_client")
    def test_initialize_shell_timeout(self, mock_tmux, mock_wait_shell):
        mock_wait_shell.return_value = False

        provider = GeminiProvider("test1234", "test-session", "window-0", None)

        with pytest.raises(TimeoutError, match="Shell initialization timed out"):
            provider.initialize()

    @patch("cli_agent_orchestrator.providers.gemini.wait_until_status")
    @patch("cli_agent_orchestrator.providers.gemini.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.gemini.tmux_client")
    def test_initialize_timeout_raises(self, mock_tmux, mock_wait_shell, mock_wait_status):
        mock_wait_shell.return_value = True
        mock_wait_status.return_value = False

        provider = GeminiProvider("test1234", "test-session", "window-0", None)

        with pytest.raises(TimeoutError, match="Gemini shell initialization timed out"):
            provider.initialize()

    @patch("cli_agent_orchestrator.providers.gemini.wait_until_status")
    @patch("cli_agent_orchestrator.providers.gemini.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.gemini.tmux_client")
    def test_initialize_uses_shell_idle_timeout(self, mock_tmux, mock_wait_shell, mock_wait_status):
        mock_wait_shell.return_value = True
        mock_wait_status.return_value = True

        provider = GeminiProvider("test1234", "test-session", "window-0", None)
        result = provider.initialize()

        assert result is True
        _, kwargs = mock_wait_status.call_args
        assert kwargs["timeout"] == 10.0

class TestGeminiProviderStatusDetection:
    @patch("cli_agent_orchestrator.providers.gemini.tmux_client")
    def test_get_status_idle(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("gemini_idle_output.txt")

        provider = GeminiProvider("test1234", "test-session", "window-0", None)
        status = provider.get_status()

        assert status == TerminalStatus.IDLE

    @patch("cli_agent_orchestrator.providers.gemini.tmux_client")
    def test_get_status_completed(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("gemini_completed_output.txt")

        provider = GeminiProvider("test1234", "test-session", "window-0", None)
        status = provider.get_status()

        assert status == TerminalStatus.COMPLETED

    @patch("cli_agent_orchestrator.providers.gemini.tmux_client")
    def test_get_status_processing(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("gemini_processing_output.txt")

        provider = GeminiProvider("test1234", "test-session", "window-0", None)
        status = provider.get_status()

        assert status == TerminalStatus.PROCESSING

    @patch("cli_agent_orchestrator.providers.gemini.tmux_client")
    def test_get_status_waiting_user_answer(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("gemini_waiting_output.txt")

        provider = GeminiProvider("test1234", "test-session", "window-0", None)
        status = provider.get_status()

        assert status == TerminalStatus.WAITING_USER_ANSWER

    @patch("cli_agent_orchestrator.providers.gemini.tmux_client")
    def test_get_status_error(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("gemini_error_output.txt")

        provider = GeminiProvider("test1234", "test-session", "window-0", None)
        status = provider.get_status()

        assert status == TerminalStatus.ERROR

    @patch("cli_agent_orchestrator.providers.gemini.tmux_client")
    def test_get_status_empty_output(self, mock_tmux):
        mock_tmux.get_history.return_value = ""

        provider = GeminiProvider("test1234", "test-session", "window-0", None)
        status = provider.get_status()

        assert status == TerminalStatus.ERROR


class TestGeminiProviderMessageExtraction:
    def test_extract_last_message_success(self):
        output = load_fixture("gemini_completed_output.txt")

        provider = GeminiProvider("test1234", "test-session", "window-0", None)
        provider._last_message_path.write_text(output, encoding="utf-8")
        message = provider.extract_last_message_from_script(output)

        assert "2+2 is 4" in message

    def test_extract_last_message_no_marker(self):
        output = "No assistant marker here"

        provider = GeminiProvider("test1234", "test-session", "window-0", None)
        provider._last_message_path.write_text(output, encoding="utf-8")

        assert provider.extract_last_message_from_script(output) == output

    def test_extract_last_message_empty_response(self):
        output = "Model: \nType your message or @path/to/file"

        provider = GeminiProvider("test1234", "test-session", "window-0", None)
        provider._last_message_path.write_text(output, encoding="utf-8")

        with pytest.raises(ValueError, match="Empty Gemini response"):
            provider.extract_last_message_from_script(output)


class TestGeminiProviderMisc:
    def test_get_idle_pattern_for_log(self):
        provider = GeminiProvider("test1234", "test-session", "window-0", None)
        assert provider.get_idle_pattern_for_log() == "CAO_GEMINI_IDLE"

    def test_exit_cli(self):
        provider = GeminiProvider("test1234", "test-session", "window-0", None)
        assert provider.exit_cli() == "exit"

    def test_cleanup(self):
        provider = GeminiProvider("test1234", "test-session", "window-0", None)
        provider._initialized = True
        provider.cleanup()
        assert provider._initialized is False
