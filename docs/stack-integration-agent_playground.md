# Stack Integration (agent_playground)

This doc is specific to the **agent_playground** workspace. It explains how CAO fits into the local stack:

- TaskFork: task decomposition + orchestration
- Model-Jump: cost-aware routing / API backends
- Spinnybot: run loops / run envelopes
- CAO (this repo): **tmux isolation + terminal lifecycle + status detection + message passing**

## CAO Server

Start CAO’s HTTP server (default `http://localhost:9889`):

```bash
cao-server
```

Optional env overrides:

- `CAO_SERVER_HOST` (default: `localhost`)
- `CAO_SERVER_PORT` (default: `9889`)

## Pointing Other Tools At CAO

CAO clients can target the server using:

- `CAO_API_BASE_URL` (example: `http://127.0.0.1:9889`)

This is primarily useful for:

- `cao-mcp-server` (which calls CAO’s HTTP API)
- helper utilities that poll CAO terminals via HTTP

## TaskFork -> CAO

TaskFork delegates terminal work to CAO via HTTP.

In **TaskFork**, set:

```bash
export TASKFORK_CAO_API_BASE_URL="http://127.0.0.1:9889"
```

Then use TaskFork’s CAO tools/executor (see TaskFork docs for exact commands):

- `cao_health`
- `cao_assign`
- `cao_handoff`
- `cao_send_message`

### Session Names (Important)

CAO session names are expected to start with the `cao-` prefix.

- If you pass `session_name=tf_123` to `POST /sessions`, CAO may normalize it to `cao-tf_123`.
- Treat the `session_name` returned by the CAO API as canonical for follow-on calls (delete/attach/etc).

## Model-Jump + CAO

A common pattern in this stack:

- Use CAO for **CLI providers** (Codex CLI, Claude Code, Gemini CLI, etc.)
- Use Model-Jump for **API providers** (Azure OpenAI, Vertex, OpenRouter, etc.)
- Use TaskFork as the “ingress” that decides which executor/backends to use.

