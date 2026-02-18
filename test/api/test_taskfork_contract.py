"""Route-level contract tests for TaskFork -> CAO integration.

These tests validate that CAO's HTTP API surface (paths + query param names + response shapes)
stays compatible with TaskFork's CAO HTTP client.

They must not require tmux or real provider CLIs; service calls are mocked.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from cli_agent_orchestrator.api.main import app
from cli_agent_orchestrator.models.terminal import Terminal

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app, raise_app_exceptions=True)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestTaskForkContract:
    async def test_health(self, client: httpx.AsyncClient):
        res = await client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["service"] == "cli-agent-orchestrator"

    async def test_create_session_param_names(self, client: httpx.AsyncClient):
        with patch("cli_agent_orchestrator.api.main.terminal_service") as mock_svc:
            mock_svc.create_terminal.return_value = Terminal(
                id="abcd1234",
                name="window-0",
                provider="codex",
                session_name="cao-tf_test",
                agent_profile="developer",
                status="idle",
            )

            res = await client.post(
                "/sessions",
                params={
                    "provider": "codex",
                    "agent_profile": "developer",
                    "session_name": "tf_test",
                    "working_directory": "/tmp",
                },
            )

            assert res.status_code == 201
            kwargs = mock_svc.create_terminal.call_args.kwargs
            assert kwargs["provider"] == "codex"
            assert kwargs["agent_profile"] == "developer"
            assert kwargs["session_name"] == "tf_test"
            assert kwargs["new_session"] is True
            assert kwargs["working_directory"] == "/tmp"

    async def test_create_terminal_in_session_param_names(self, client: httpx.AsyncClient):
        with patch("cli_agent_orchestrator.api.main.terminal_service") as mock_svc:
            mock_svc.create_terminal.return_value = Terminal(
                id="beefcafe",
                name="window-1",
                provider="codex",
                session_name="cao-tf_test",
                agent_profile="reviewer",
                status="idle",
            )

            res = await client.post(
                "/sessions/cao-tf_test/terminals",
                params={
                    "provider": "codex",
                    "agent_profile": "reviewer",
                    "working_directory": "/tmp/work",
                },
            )

            assert res.status_code == 201
            kwargs = mock_svc.create_terminal.call_args.kwargs
            assert kwargs["provider"] == "codex"
            assert kwargs["agent_profile"] == "reviewer"
            assert kwargs["session_name"] == "cao-tf_test"
            assert kwargs["new_session"] is False
            assert kwargs["working_directory"] == "/tmp/work"

    async def test_get_terminal(self, client: httpx.AsyncClient):
        with patch("cli_agent_orchestrator.api.main.terminal_service") as mock_svc:
            mock_svc.get_terminal.return_value = {
                "id": "abcd1234",
                "name": "window-0",
                "provider": "codex",
                "session_name": "cao-tf_test",
                "agent_profile": "developer",
                "status": "idle",
                "last_active": None,
            }

            res = await client.get("/terminals/abcd1234")
            assert res.status_code == 200
            data = res.json()
            assert data["id"] == "abcd1234"
            assert data["provider"] == "codex"
            assert data["session_name"] == "cao-tf_test"

    async def test_send_input(self, client: httpx.AsyncClient):
        with patch("cli_agent_orchestrator.api.main.terminal_service") as mock_svc:
            mock_svc.send_input.return_value = True

            res = await client.post(
                "/terminals/abcd1234/input",
                params={"message": "hello"},
            )
            assert res.status_code == 200
            assert res.json()["success"] is True
            mock_svc.send_input.assert_called_once_with("abcd1234", "hello")

    async def test_get_output_last(self, client: httpx.AsyncClient):
        with patch("cli_agent_orchestrator.api.main.terminal_service") as mock_svc:
            mock_svc.get_output.return_value = "last message"

            res = await client.get("/terminals/abcd1234/output", params={"mode": "last"})
            assert res.status_code == 200
            data = res.json()
            assert data["output"] == "last message"
            assert data["mode"] == "last"
            mock_svc.get_output.assert_called_once()

    async def test_exit_terminal(self, client: httpx.AsyncClient):
        provider = SimpleNamespace(exit_cli=lambda: "/exit")
        with patch("cli_agent_orchestrator.api.main.provider_manager") as mock_mgr, patch(
            "cli_agent_orchestrator.api.main.terminal_service"
        ) as mock_svc:
            mock_mgr.get_provider.return_value = provider
            mock_svc.send_input.return_value = True

            res = await client.post("/terminals/abcd1234/exit")
            assert res.status_code == 200
            assert res.json()["success"] is True
            mock_mgr.get_provider.assert_called_once_with("abcd1234")
            mock_svc.send_input.assert_called_once_with("abcd1234", "/exit")

    async def test_delete_session(self, client: httpx.AsyncClient):
        with patch("cli_agent_orchestrator.api.main.session_service") as mock_svc:
            mock_svc.delete_session.return_value = True
            res = await client.delete("/sessions/cao-tf_test")
            assert res.status_code == 200
            assert res.json()["success"] is True
            mock_svc.delete_session.assert_called_once_with("cao-tf_test")

    async def test_create_inbox_message(self, client: httpx.AsyncClient):
        inbox_msg = SimpleNamespace(
            id=123,
            sender_id="deadbeef",
            receiver_id="abcd1234",
            created_at=datetime(2026, 2, 18, 1, 0, 0, tzinfo=timezone.utc),
        )
        with patch("cli_agent_orchestrator.api.main.create_inbox_message") as mock_create, patch(
            "cli_agent_orchestrator.api.main.inbox_service"
        ) as mock_inbox:
            mock_create.return_value = inbox_msg

            res = await client.post(
                "/terminals/abcd1234/inbox/messages",
                params={"sender_id": "deadbeef", "message": "hi"},
            )

            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True
            assert data["message_id"] == 123
            assert data["sender_id"] == "deadbeef"
            assert data["receiver_id"] == "abcd1234"
            assert isinstance(data["created_at"], str) and data["created_at"]

            mock_create.assert_called_once_with("deadbeef", "abcd1234", "hi")
            mock_inbox.check_and_send_pending_messages.assert_called_once_with("abcd1234")

