"""Unit tests for Ollama provider."""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.ollama import OllamaProvider

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(filename: str) -> str:
    with open(FIXTURES_DIR / filename, "r") as f:
        return f.read()


class TestOllamaProviderInitialization:
    @patch("cli_agent_orchestrator.providers.ollama.subprocess.run")
    @patch("cli_agent_orchestrator.providers.ollama.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.ollama.tmux_client")
    def test_initialize_success(self, mock_tmux, mock_wait_shell, mock_run):
        mock_wait_shell.return_value = True
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ollama", "list"], returncode=0, stdout="", stderr=""
        )

        provider = OllamaProvider("test1234", "test-session", "window-0", None)
        with patch.object(
            OllamaProvider, "get_status", return_value=TerminalStatus.IDLE
        ) as mock_get_status:
            result = provider.initialize()

        assert result is True
        mock_wait_shell.assert_called_once()
        mock_run.assert_called_once()
        assert mock_tmux.send_keys.call_count == 1
        assert mock_get_status.call_count >= 1

    @patch("cli_agent_orchestrator.providers.ollama.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.ollama.tmux_client")
    def test_initialize_shell_timeout(self, mock_tmux, mock_wait_shell):
        mock_wait_shell.return_value = False

        provider = OllamaProvider("test1234", "test-session", "window-0", None)

        with pytest.raises(TimeoutError, match="Shell initialization timed out"):
            provider.initialize()

    @patch("cli_agent_orchestrator.providers.ollama.subprocess.run")
    @patch("cli_agent_orchestrator.providers.ollama.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.ollama.tmux_client")
    def test_initialize_timeout_raises(self, mock_tmux, mock_wait_shell, mock_run):
        mock_wait_shell.return_value = True
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ollama", "list"], returncode=0, stdout="", stderr=""
        )

        provider = OllamaProvider("test1234", "test-session", "window-0", None)

        with patch.dict(os.environ, {"CAO_OLLAMA_INIT_TIMEOUT": "0"}):
            with pytest.raises(TimeoutError, match="Ollama initialization timed out"):
                provider.initialize()


class TestOllamaProviderStatusDetection:
    @patch("cli_agent_orchestrator.providers.ollama.tmux_client")
    def test_get_status_idle(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("ollama_idle_output.txt")

        provider = OllamaProvider("test1234", "test-session", "window-0", None)
        status = provider.get_status()

        assert status == TerminalStatus.IDLE

    @patch("cli_agent_orchestrator.providers.ollama.tmux_client")
    def test_get_status_completed(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("ollama_completed_output.txt")

        provider = OllamaProvider("test1234", "test-session", "window-0", None)
        status = provider.get_status()

        assert status == TerminalStatus.COMPLETED

    @patch("cli_agent_orchestrator.providers.ollama.tmux_client")
    def test_get_status_processing(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("ollama_processing_output.txt")

        provider = OllamaProvider("test1234", "test-session", "window-0", None)
        status = provider.get_status()

        assert status == TerminalStatus.PROCESSING

    @patch("cli_agent_orchestrator.providers.ollama.tmux_client")
    def test_get_status_waiting_user_answer(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("ollama_waiting_output.txt")

        provider = OllamaProvider("test1234", "test-session", "window-0", None)
        status = provider.get_status()

        assert status == TerminalStatus.WAITING_USER_ANSWER

    @patch("cli_agent_orchestrator.providers.ollama.tmux_client")
    def test_get_status_error(self, mock_tmux):
        mock_tmux.get_history.return_value = load_fixture("ollama_error_output.txt")

        provider = OllamaProvider("test1234", "test-session", "window-0", None)
        status = provider.get_status()

        assert status == TerminalStatus.ERROR

    @patch("cli_agent_orchestrator.providers.ollama.tmux_client")
    def test_get_status_empty_output(self, mock_tmux):
        mock_tmux.get_history.return_value = ""

        provider = OllamaProvider("test1234", "test-session", "window-0", None)
        status = provider.get_status()

        assert status == TerminalStatus.ERROR


class TestOllamaProviderMessageExtraction:
    def test_extract_last_message_success(self):
        output = load_fixture("ollama_completed_output.txt")

        provider = OllamaProvider("test1234", "test-session", "window-0", None)
        message = provider.extract_last_message_from_script(output)

        assert "How can I help" in message

    def test_extract_last_message_no_prompt(self):
        output = "No prompts here"

        provider = OllamaProvider("test1234", "test-session", "window-0", None)

        with pytest.raises(ValueError, match="No Ollama response found"):
            provider.extract_last_message_from_script(output)

    def test_extract_last_message_empty_response(self):
        output = ">>> hello\n\n>>>"

        provider = OllamaProvider("test1234", "test-session", "window-0", None)

        with pytest.raises(ValueError, match="Empty Ollama response"):
            provider.extract_last_message_from_script(output)


class TestOllamaProviderMisc:
    def test_get_idle_pattern_for_log(self):
        provider = OllamaProvider("test1234", "test-session", "window-0", None)
        assert provider.get_idle_pattern_for_log() == ">>>"

    def test_exit_cli(self):
        provider = OllamaProvider("test1234", "test-session", "window-0", None)
        assert provider.exit_cli() == "/bye"

    def test_cleanup(self):
        provider = OllamaProvider("test1234", "test-session", "window-0", None)
        provider._initialized = True
        provider.cleanup()
        assert provider._initialized is False
