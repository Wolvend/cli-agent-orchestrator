# CLI Agent Orchestrator (CAO) Plan (Stack Integration + Hardening)

Date: 2026-02-17

This plan focuses on making CAO a reliable integration layer for the local agent stack:

- TaskFork (task decomposition + orchestration)
- Model-Jump (cost-aware router / gateway)
- Spinnybot (run loop / envelopes / signals)
- Oroboros (memory + MCP tooling)

CAO’s core role is: **own terminal lifecycle + tmux isolation + status detection + message passing** via HTTP + MCP.

## Current State (Baseline)

- FastAPI HTTP server (default: `http://127.0.0.1:9889`)
- MCP server (tool surface for coordination)
- Providers (Codex CLI, Claude Code, Gemini CLI, Q CLI / Kiro CLI, etc.)
- tmux session/window management (create, send input, capture output)
- Permission-prompt detection (avoid injecting/delivering messages during confirmations)

## Integration Contract (What Must Stay Stable)

CAO should maintain a stable API surface that upstream orchestrators can depend on:

- `GET /health`
- session lifecycle:
  - create/list/delete sessions
- terminal input/output:
  - send input to a terminal
  - fetch last output / extract last message
- status:
  - `IDLE | PROCESSING | WAITING_USER_ANSWER | COMPLETED | ERROR`

TaskFork currently depends on this contract for `executor="cao-handoff"` and MCP tools (`cao_health`, `cao_assign`, `cao_handoff`, `cao_send_message`).

## Logging / Observability (Hard Requirement)

Goals:

- Never log raw prompts/tool inputs (may contain secrets).
- Prefer structured, grep-friendly logs:
  - session name, terminal id, provider, status transitions, poll loop counts, timeouts
- Make debug logging opt-in via env (ex: `CAO_LOG_LEVEL=DEBUG`).

## Roadmap

### Phase 1: Local Reliability (In Progress)

- Keep permission-prompt detection conservative (prefer `WAITING_USER_ANSWER` when uncertain).
- Keep provider integration tests resilient:
  - skip/xfail when a CLI binary is a shim for another tool (ex: `q` -> `kiro-cli`)
- Reduce log risk (no prompt content in INFO logs).

### Phase 2: “Stack Mode” (Next)

Add a lightweight “stack mode” doc + examples that show CAO running as the terminal substrate for:

1. TaskFork orchestrating via CAO (handoff/assign/send_message)
2. Model-Jump routing to API backends while CAO handles CLI backends
3. Oroboros providing MCP memory/tools for long-running projects

### Phase 3: Contract Tests (Next)

Add a small set of HTTP contract tests that:

- start CAO
- create a session
- send an input
- assert status transitions
- fetch last output
- delete the session

These tests should be runnable without real provider CLIs by using a “dummy provider” that echoes input.

## Validation Commands (Local)

```bash
cd source/agents/frameworks/cli-agent-orchestrator
.venv/bin/python -m pytest -q
```

Optional provider smoke (best-effort; depends on local CLI installs):

```bash
cd source/agents/frameworks/cli-agent-orchestrator
CAO_OLLAMA_MODEL=qwen2.5:0.5b .venv/bin/python scripts/smoke_test_all_providers.py
```

