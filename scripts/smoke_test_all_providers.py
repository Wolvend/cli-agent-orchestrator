#!/usr/bin/env python3
"""
End-to-end smoke test for CLI Agent Orchestrator providers.

This script starts `cao-server`, creates a session per provider, sends a basic prompt,
waits for a response, and cleans up. It is intentionally best-effort: providers that
require interactive auth/setup will be reported as PARTIAL (WAITING_USER_ANSWER).

Run from the repo root:
  . .venv/bin/activate
  python scripts/smoke_test_all_providers.py
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    status: str  # PASS | PARTIAL | FAIL | BLOCKED
    detail: str
    terminal_id: Optional[str] = None
    session_name: Optional[str] = None


CAO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CAO_BIN = CAO_ROOT / ".venv" / "bin" / "cao-server"

LLAMA_BIN_DEFAULT = "/home/wolvend/codex/agent_playground/llama.cpp/build/bin/llama-cli"
LLAMA_GGUF_DEFAULT = (
    "/tmp/cao-e2e-llama-cpp/providers/llama-cpp-home/.cache/llama.cpp/"
    "ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf"
)

BASE_URL = "http://localhost:9889"


def _copy_tree(src: pathlib.Path, dst: pathlib.Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Python 3.12: dirs_exist_ok supported for shutil.copytree
    if dst.exists():
        # Keep existing but update; we want "best effort", not perfect sync.
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _start_server(env: dict[str, str], log_path: pathlib.Path) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "ab", buffering=0)
    # NOTE: We intentionally keep the server process alive only for one provider test.
    return subprocess.Popen(
        [str(CAO_BIN)],
        cwd=str(CAO_ROOT),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )


def _wait_health(timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    with httpx.Client() as client:
        while time.time() < deadline:
            try:
                r = client.get(f"{BASE_URL}/health", timeout=2.0)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.5)
    raise RuntimeError("cao-server did not become healthy")


def _tail_file(path: pathlib.Path, max_bytes: int = 12000) -> str:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return "(no log file)"
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    return data.decode("utf-8", errors="replace")


def _kill_tmux_session_best_effort(session_name: str) -> None:
    # If the server crashes mid-create, clean up the tmux session by name.
    try:
        subprocess.run(
            ["tmux", "kill-session", "-t", session_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _test_provider(
    provider: str,
    agent_profile: str,
    working_directory: str,
    env: dict[str, str],
    create_timeout_s: float,
    completion_timeout_s: float,
) -> ProviderResult:
    run_id = int(time.time())
    session_base = f"smoke-{provider}-{run_id}"
    expected_session = f"cao-{session_base}"

    # Start server for this provider.
    cao_home = pathlib.Path(env["CAO_HOME_DIR"])
    server_log = cao_home / "server.log"
    proc = _start_server(env=env, log_path=server_log)

    term_id: Optional[str] = None
    try:
        _wait_health()

        with httpx.Client() as client:
            # Create session (this blocks until provider initializes).
            r = client.post(
                f"{BASE_URL}/sessions",
                params={
                    "provider": provider,
                    "agent_profile": agent_profile,
                    "session_name": session_base,
                    "working_directory": working_directory,
                },
                timeout=create_timeout_s,
            )
            r.raise_for_status()
            term = r.json()
            term_id = term["id"]
            session_name = term["session_name"]

            # Check runtime status (Terminal returned from create is always IDLE).
            r = client.get(f"{BASE_URL}/terminals/{term_id}", timeout=10.0)
            r.raise_for_status()
            status = r.json().get("status")

            if status == "waiting_user_answer":
                # Common for interactive login/setup.
                client.delete(f"{BASE_URL}/sessions/{session_name}", timeout=30.0)
                return ProviderResult(
                    provider=provider,
                    status="PARTIAL",
                    detail="WAITING_USER_ANSWER (likely needs interactive setup/auth)",
                    terminal_id=term_id,
                    session_name=session_name,
                )

            # Send a basic prompt and wait for completion.
            prompt = "What is 2+2? Answer with just the number."
            r = client.post(
                f"{BASE_URL}/terminals/{term_id}/input",
                params={"message": prompt},
                timeout=10.0,
            )
            r.raise_for_status()

            deadline = time.time() + completion_timeout_s
            while time.time() < deadline:
                r = client.get(f"{BASE_URL}/terminals/{term_id}", timeout=10.0)
                r.raise_for_status()
                status = r.json().get("status")
                if status == "completed":
                    break
                if status in ("waiting_user_answer", "error"):
                    break
                time.sleep(1.0)

            if status != "completed":
                out = ""
                # If the provider needs interactive setup/auth, treat as PARTIAL (best-effort).
                if status == "waiting_user_answer":
                    try:
                        r = client.get(
                            f"{BASE_URL}/terminals/{term_id}/output",
                            params={"mode": "last"},
                            timeout=20.0,
                        )
                        r.raise_for_status()
                        out = (r.json().get("output") or "").strip()
                    except Exception:
                        out = ""

                    # Best effort cleanup
                    try:
                        client.delete(f"{BASE_URL}/sessions/{session_name}", timeout=30.0)
                    except Exception:
                        pass

                    detail = "WAITING_USER_ANSWER (likely needs interactive setup/auth)"
                    if out:
                        detail += f" (output={out[:160]!r})"
                    return ProviderResult(
                        provider=provider,
                        status="PARTIAL",
                        detail=detail,
                        terminal_id=term_id,
                        session_name=session_name,
                    )

                # For explicit errors, include a small output snippet if possible.
                if status == "error":
                    out = ""
                    try:
                        r = client.get(
                            f"{BASE_URL}/terminals/{term_id}/output",
                            params={"mode": "full"},
                            timeout=20.0,
                        )
                        r.raise_for_status()
                        out = (r.json().get("output") or "").strip()
                    except Exception:
                        out = ""

                # Best effort cleanup
                try:
                    client.delete(f"{BASE_URL}/sessions/{session_name}", timeout=30.0)
                except Exception:
                    pass
                return ProviderResult(
                    provider=provider,
                    status="FAIL",
                    detail=(
                        f"did not complete (status={status})"
                        + (f" (output={out[-240:]!r})" if out else "")
                    ),
                    terminal_id=term_id,
                    session_name=session_name,
                )

            # Prefer mode=last (provider parsing), fall back to full output.
            out = ""
            try:
                r = client.get(
                    f"{BASE_URL}/terminals/{term_id}/output",
                    params={"mode": "last"},
                    timeout=20.0,
                )
                r.raise_for_status()
                out = (r.json().get("output") or "").strip()
            except Exception:
                r = client.get(
                    f"{BASE_URL}/terminals/{term_id}/output",
                    params={"mode": "full"},
                    timeout=20.0,
                )
                r.raise_for_status()
                full = r.json().get("output") or ""
                out = full.strip()[-400:]

            client.delete(f"{BASE_URL}/sessions/{session_name}", timeout=30.0)

            if not out:
                return ProviderResult(
                    provider=provider,
                    status="FAIL",
                    detail="completed but empty output",
                    terminal_id=term_id,
                    session_name=session_name,
                )

            return ProviderResult(
                provider=provider,
                status="PASS",
                detail=f"ok (output={out[:80]!r})",
                terminal_id=term_id,
                session_name=session_name,
            )

    except httpx.ReadTimeout:
        # Likely provider init hang. Kill server and tmux session by expected name.
        return ProviderResult(
            provider=provider,
            status="FAIL",
            detail=f"create_session timed out after {create_timeout_s:g}s",
            terminal_id=term_id,
            session_name=expected_session,
        )
    except Exception as e:
        # Include a short server log tail for context.
        tail = _tail_file(server_log)
        return ProviderResult(
            provider=provider,
            status="FAIL",
            detail=f"{type(e).__name__}: {e}\n--- server.log tail ---\n{tail}",
            terminal_id=term_id,
            session_name=expected_session,
        )
    finally:
        # Ensure the server is stopped.
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        # Best-effort cleanup in case the server died mid-flight.
        _kill_tmux_session_best_effort(expected_session)


def main() -> int:
    if not CAO_BIN.exists():
        print(f"ERROR: cao-server not found at {CAO_BIN}", file=sys.stderr)
        return 2

    providers = [
        "llama_cpp",
        "gemini",
        "codex",
        "claude_code",
        "ollama",
        "q_cli",
        "kiro_cli",
    ]

    agent_profile = os.getenv("SMOKE_AGENT_PROFILE", "developer")
    working_directory = os.getenv("SMOKE_WORKDIR", "/home/wolvend/codex/agent_playground")

    # Per-provider time budgets (seconds).
    create_timeouts = {
        "llama_cpp": 120.0,
        "gemini": 120.0,
        "codex": 180.0,
        "claude_code": 180.0,
        "ollama": 180.0,
        "q_cli": 30.0,
        "kiro_cli": 30.0,
    }
    completion_timeouts = {
        "llama_cpp": 180.0,
        "gemini": 180.0,
        "codex": 240.0,
        "claude_code": 240.0,
        "ollama": 180.0,
        "q_cli": 120.0,
        "kiro_cli": 120.0,
    }

    results: list[ProviderResult] = []

    # Inputs for llama.cpp smoke test (use local GGUF if present to avoid network).
    llama_bin = os.getenv("CAO_LLAMA_CPP_BIN", LLAMA_BIN_DEFAULT)
    llama_model = os.getenv("CAO_LLAMA_CPP_MODEL", LLAMA_GGUF_DEFAULT)

    # Seed source gemini auth/config into CAO's isolated HOME for gemini.
    user_gemini = pathlib.Path.home() / ".gemini"

    for provider in providers:
        run_id = int(time.time())
        cao_home_dir = f"/tmp/cao-smoke-{provider}-{run_id}"
        env = os.environ.copy()
        env.update(
            {
                "CAO_HOME_DIR": cao_home_dir,
                "NO_COLOR": "1",
            }
        )

        # Provider-specific env/config.
        if provider == "llama_cpp":
            env["CAO_LLAMA_CPP_BIN"] = llama_bin
            env["CAO_LLAMA_CPP_MODEL"] = llama_model
            env["CAO_LLAMA_CPP_INIT_TIMEOUT"] = os.getenv("CAO_LLAMA_CPP_INIT_TIMEOUT", "300")

            if not pathlib.Path(llama_bin).exists():
                results.append(
                    ProviderResult(
                        provider=provider,
                        status="BLOCKED",
                        detail=f"llama-cli not found: {llama_bin}",
                    )
                )
                continue
            if not pathlib.Path(llama_model).exists():
                results.append(
                    ProviderResult(
                        provider=provider,
                        status="BLOCKED",
                        detail=f"GGUF model not found: {llama_model}",
                    )
                )
                continue

        if provider == "gemini":
            # Copy ~/.gemini to $HOME/.gemini within the provider HOME directory.
            gemini_home = pathlib.Path(cao_home_dir) / "providers" / "gemini-home"
            _copy_tree(user_gemini, gemini_home / ".gemini")

        if provider == "ollama":
            # Optional override for remote ollama.
            smoke_host = os.getenv("SMOKE_OLLAMA_HOST")
            if smoke_host:
                env["OLLAMA_HOST"] = smoke_host
            env["CAO_OLLAMA_INIT_TIMEOUT"] = os.getenv("CAO_OLLAMA_INIT_TIMEOUT", "180")

        # Fast-fail: q/kiro CLIs missing.
        if provider in ("q_cli", "kiro_cli"):
            binary = "q" if provider == "q_cli" else "kiro-cli"
            if shutil.which(binary) is None:
                results.append(
                    ProviderResult(
                        provider=provider,
                        status="BLOCKED",
                        detail=f"{binary} not found in PATH",
                    )
                )
                continue

        results.append(
            _test_provider(
                provider=provider,
                agent_profile=agent_profile,
                working_directory=working_directory,
                env=env,
                create_timeout_s=create_timeouts[provider],
                completion_timeout_s=completion_timeouts[provider],
            )
        )

    # Print a concise report.
    print("SMOKE TEST RESULTS")
    for r in results:
        msg = f"- {r.provider}: {r.status}"
        if r.detail:
            msg += f" :: {r.detail.splitlines()[0]}"
        print(msg)

    failed = [r for r in results if r.status in ("FAIL",)]
    blocked = [r for r in results if r.status in ("BLOCKED",)]
    partial = [r for r in results if r.status in ("PARTIAL",)]

    print("")
    print(
        f"PASS={len([r for r in results if r.status == 'PASS'])} "
        f"PARTIAL={len(partial)} FAIL={len(failed)} BLOCKED={len(blocked)}"
    )

    # Exit non-zero if anything failed.
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
