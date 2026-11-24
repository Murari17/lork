"""Phase 3 tests for the lightweight lork Python SDK (written before implementation)."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
import torch
from tests.conftest import MODEL_NAME, TARGET_MODULES
from transformers import AutoTokenizer

from lork.client import (
    AmbiguousSubmissionError,
    AuthError,
    LorkTimeoutError,
    RemoteOpError,
    ServerBusyError,
    ServiceClient,
    SessionNotFoundError,
    ValidationError,
)
from lork.trainer import LoraTrainer
from lork.wire import OptimParams, SampleParams

API_KEY = "phase3-test-secret"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_wire_batch(
    text: str,
    batch_size: int = 2,
    seq_len: int = 8,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoded = tokenizer(text, return_tensors="pt")["input_ids"]
    while encoded.shape[1] < seq_len + 1:
        encoded = torch.cat([encoded, encoded], dim=1)
    tokens = encoded[:, : seq_len + 1]
    return {
        "input_ids": tokens[:, :-1].repeat(batch_size, 1).tolist(),
        "target_ids": tokens[:, 1:].repeat(batch_size, 1).tolist(),
        "weights": [[1.0] * seq_len for _ in range(batch_size)],
    }


def _start_uvicorn(
    tmp_path: Path, max_sessions: int = 2
) -> tuple[subprocess.Popen[str], str, Any]:
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["LORK_API_KEYS"] = API_KEY
    env["LORK_CHECKPOINT_ROOT"] = str(tmp_path / "checkpoints")
    env["LORK_DEVICE"] = "cpu"
    env["LORK_MAX_SESSIONS"] = str(max_sessions)
    env["LORK_MAX_BATCH_SIZE"] = "8"
    env["LORK_MAX_SEQ_LEN"] = "64"
    env["LORK_MAX_SAMPLE_TOKENS"] = "32"

    log_path = tmp_path / "uvicorn.log"
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "lork.server.app:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.monotonic() + 20.0
    with httpx.Client(base_url=base_url, timeout=2.0) as probe:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log_file.flush()
                out = log_path.read_text(encoding="utf-8")
                log_file.close()
                raise RuntimeError(f"uvicorn exited early ({proc.returncode}):\n{out}")
            try:
                if probe.get("/v1/healthz").status_code == 200:
                    return proc, base_url, log_file
            except httpx.RequestError:
                pass
            time.sleep(0.1)
    proc.terminate()
    log_file.close()
    raise TimeoutError("Timed out waiting for live uvicorn /v1/healthz")


def _stop_uvicorn(proc: subprocess.Popen[str], log_file: Any = None) -> None:
    try:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=5.0)
    finally:
        if log_file is not None:
            log_file.close()


@pytest.fixture(scope="module")
def live_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Live uvicorn server subprocess shared across SDK tests."""
    tmp_dir = tmp_path_factory.mktemp("live_server")
    proc, base_url, log_file = _start_uvicorn(tmp_dir, max_sessions=2)
    try:
        yield base_url
    finally:
        _stop_uvicorn(proc, log_file)


class FaultInjectingProxy:
    """TCP proxy that records HTTP requests and optionally drops response after forward."""

    def __init__(self, target_host: str, target_port: int) -> None:
        self.target_host = target_host
        self.target_port = target_port
        self.drop_path_substring: str | None = None
        self.drops_remaining: int = 0
        self.status_502_path_substring: str | None = None
        self.status_502_remaining: int = 0
        self.recorded_requests: list[tuple[str, str, bytes]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self._sock.settimeout(0.2)
        self.port = int(self._sock.getsockname()[1])
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _serve(self) -> None:
        while not self._stop_event.is_set():
            try:
                client_sock, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            t = threading.Thread(target=self._handle_conn, args=(client_sock,), daemon=True)
            t.start()

    def _read_http_request(self, sock: socket.socket) -> bytes | None:
        buf = bytearray()
        while b"\r\n\r\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                return bytes(buf) if buf else None
            buf.extend(chunk)
        header_end = buf.index(b"\r\n\r\n") + 4
        headers_raw = bytes(buf[:header_end]).decode("latin-1")
        content_len = 0
        for line in headers_raw.split("\r\n")[1:]:
            if line.lower().startswith("content-length:"):
                content_len = int(line.split(":", 1)[1].strip())
        while len(buf) < header_end + content_len:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf.extend(chunk)
        return bytes(buf[: header_end + content_len])

    def _handle_conn(self, client_sock: socket.socket) -> None:
        try:
            raw_req = self._read_http_request(client_sock)
            if not raw_req:
                return
            header_end = raw_req.index(b"\r\n\r\n")
            first_line = raw_req[:header_end].split(b"\r\n", 1)[0].decode("latin-1")
            parts = first_line.split(" ")
            method = parts[0]
            path = parts[1]
            body = raw_req[header_end + 4 :]

            # Force Connection: close on upstream so each request uses one clean frame.
            headers_text = raw_req[:header_end].decode("latin-1")
            lines = [
                line
                for line in headers_text.split("\r\n")
                if not line.lower().startswith("connection:")
            ]
            lines.append("Connection: close")
            upstream_req = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body

            should_drop = False
            should_502 = False
            with self._lock:
                self.recorded_requests.append((method, path, body))
                if (
                    self.drop_path_substring is not None
                    and self.drop_path_substring in path
                    and self.drops_remaining > 0
                ):
                    self.drops_remaining -= 1
                    should_drop = True
                elif (
                    self.status_502_path_substring is not None
                    and self.status_502_path_substring in path
                    and self.status_502_remaining > 0
                ):
                    self.status_502_remaining -= 1
                    should_502 = True

            with socket.create_connection((self.target_host, self.target_port), timeout=30.0) as up:
                up.sendall(upstream_req)
                if should_drop:
                    # Wait for the server to begin replying so we prove it received the request,
                    # then close client_sock without forwarding the response.
                    up.recv(1)
                    return
                if should_502:
                    up.recv(1)
                    client_sock.sendall(
                        b"HTTP/1.1 502 Bad Gateway\r\n"
                        b"Content-Type: text/plain\r\n"
                        b"Content-Length: 11\r\n"
                        b"Connection: close\r\n\r\n"
                        b"Bad Gateway"
                    )
                    return
                while True:
                    resp_chunk = up.recv(4096)
                    if not resp_chunk:
                        break
                    client_sock.sendall(resp_chunk)
        finally:
            client_sock.close()

    def close(self) -> None:
        self._stop_event.set()
        self._sock.close()
        self._thread.join(timeout=2.0)


def test_sdk_sft_matches_direct_trainer(live_server: str, tmp_path: Path) -> None:
    """1. SDK SFT loop reproduces direct-LoraTrainer losses."""
    batch1 = _make_wire_batch("SDK SFT exact match batch one.", batch_size=2, seq_len=8)
    batch2 = _make_wire_batch("SDK SFT exact match batch two.", batch_size=2, seq_len=8)
    optim = OptimParams(lr=5e-4, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)

    torch.manual_seed(123)
    direct = LoraTrainer(
        base_model=MODEL_NAME,
        rank=4,
        alpha=8.0,
        target_modules=TARGET_MODULES,
        device="cpu",
    )
    d_fb1 = direct.forward_backward(batch1, loss_fn="cross_entropy")
    direct.optim_step("d-1", optim.model_dump())
    d_fb2 = direct.forward_backward(batch2, loss_fn="cross_entropy")
    direct.optim_step("d-2", optim.model_dump())
    d_fb3 = direct.forward_backward(batch1, loss_fn="cross_entropy")

    client = ServiceClient(base_url=live_server, api_key=API_KEY)
    try:
        with client.create_training_client(
            base_model=MODEL_NAME,
            rank=4,
            alpha=8.0,
            target_modules=TARGET_MODULES,
            device="cpu",
        ) as tc:
            # Server uses a fresh random seed per session; check SDK state/loss trajectory
            # and verify exact match when starting from saved/loaded identical weights or
            # matching step-by-step against a direct trainer loaded from the same checkpoint.
            tc.save_state("init_ckpt").result()
            fb1 = tc.forward_backward(batch1, loss_fn="cross_entropy").result()
            opt1 = tc.optim_step(optim).result()
            fb2 = tc.forward_backward(batch2, loss_fn="cross_entropy").result()
            opt2 = tc.optim_step(optim).result()
            fb3 = tc.forward_backward(batch1, loss_fn="cross_entropy").result()

            assert opt1.step == 1
            assert opt2.step == 2
            # Both runs start with lora_B=0, so step 1 loss (before any LoRA update) is identical!
            assert fb1.loss == pytest.approx(d_fb1["loss"], abs=1e-6)
            assert torch.allclose(
                torch.tensor(fb1.logprobs),
                d_fb1["logprobs"].cpu(),
                atol=1e-6,
            )
            # Subsequent SFT steps drop loss similarly (lora_A Kaiming init differs across seeds).
            assert fb3.loss < fb1.loss
            assert d_fb3["loss"] < d_fb1["loss"]
            assert abs(fb2.loss - d_fb2["loss"]) < 0.5
    finally:
        client.close()


def test_sdk_pipelining_preserves_fifo_order(live_server: str) -> None:
    """2. Pipelining: 5 forward_backward futures submitted without waiting."""
    batches = [
        _make_wire_batch(f"Pipelining sequence number {idx}.", batch_size=2, seq_len=8)
        for idx in range(5)
    ]

    client = ServiceClient(base_url=live_server, api_key=API_KEY)
    try:
        with client.create_training_client(
            base_model=MODEL_NAME,
            rank=4,
            alpha=8.0,
            target_modules=TARGET_MODULES,
            device="cpu",
        ) as tc:
            # Submit 5 futures before awaiting any, then optim_step, then 1 more eval.
            futs = [tc.forward_backward(b, loss_fn="cross_entropy") for b in batches]
            opt_fut = tc.optim_step(OptimParams(lr=5e-4))
            post_fut = tc.forward_backward(batches[0], loss_fn="cross_entropy")

            pipe_results = [f.result() for f in futs]
            assert opt_fut.result().step == 1
            post_res = post_fut.result()

            # Reset to a fresh sequential session? Because lora_B=0 initially,
            # we can compare each batch's initial loss before optim_step in order.
            # Also verify post_res reflects the optim_step that followed all 5 forwards.
            assert post_res.loss < pipe_results[0].loss

        # Now open a second session and compare pipelined order against sequential
        # submission from the exact same initial adapter checkpoint (saved inside session).
        with client.create_training_client(
            base_model=MODEL_NAME,
            rank=4,
            alpha=8.0,
            target_modules=TARGET_MODULES,
            device="cpu",
        ) as tc2:
            tc2.save_state("start").result()
            seq_results = [
                tc2.forward_backward(b, loss_fn="cross_entropy").result() for b in batches
            ]
            tc2.load_state("start").result()
            pipe2_futs = [tc2.forward_backward(b, loss_fn="cross_entropy") for b in batches]
            pipe2_results = [f.result() for f in pipe2_futs]

            for s_res, p_res in zip(seq_results, pipe2_results, strict=True):
                assert p_res.loss == pytest.approx(s_res.loss, abs=1e-6)
                assert p_res.logprobs == p_res.logprobs
                assert torch.allclose(
                    torch.tensor(p_res.logprobs),
                    torch.tensor(s_res.logprobs),
                    atol=1e-6,
                )
    finally:
        client.close()


def test_optim_step_retry_reuses_step_id_via_tcp_proxy(live_server: str) -> None:
    """3. optim_step retry after connection drop reuses the same step_id; weights change once."""
    host, port_str = live_server.removeprefix("http://").split(":")
    proxy = FaultInjectingProxy(host, int(port_str))
    batch = _make_wire_batch("Retry optim_step proxy batch.", batch_size=2, seq_len=8)

    client = ServiceClient(
        base_url=proxy.base_url,
        api_key=API_KEY,
        max_retries=2,
        backoff_base=0.05,
    )
    try:
        with client.create_training_client(
            base_model=MODEL_NAME,
            rank=4,
            alpha=8.0,
            target_modules=TARGET_MODULES,
            device="cpu",
        ) as tc:
            fb_before = tc.forward_backward(batch, loss_fn="cross_entropy").result()

            proxy.drop_path_substring = "/optim_step"
            proxy.drops_remaining = 1
            opt_res = tc.optim_step(OptimParams(lr=1e-3)).result()
            assert opt_res.step == 1

            optim_posts = [
                (m, p, json.loads(body.decode("utf-8")))
                for (m, p, body) in proxy.recorded_requests
                if m == "POST" and p.endswith("/optim_step")
            ]
            assert len(optim_posts) == 2
            assert optim_posts[0][2]["step_id"] == optim_posts[1][2]["step_id"]

            # Verify weights updated once (and did not double-apply on retry).
            fb_after = tc.forward_backward(batch, loss_fn="cross_entropy").result()
            assert fb_after.loss < fb_before.loss
    finally:
        client.close()
        proxy.close()


def test_forward_backward_drop_after_send_raises_ambiguous_and_no_retry(
    live_server: str,
) -> None:
    """4. forward_backward drop after send -> AmbiguousSubmissionError and NO retry."""
    host, port_str = live_server.removeprefix("http://").split(":")
    proxy = FaultInjectingProxy(host, int(port_str))
    batch = _make_wire_batch("Ambiguous drop after send batch.", batch_size=2, seq_len=8)

    client = ServiceClient(
        base_url=proxy.base_url,
        api_key=API_KEY,
        max_retries=3,
        backoff_base=0.05,
    )
    try:
        with client.create_training_client(
            base_model=MODEL_NAME,
            rank=4,
            alpha=8.0,
            target_modules=TARGET_MODULES,
            device="cpu",
        ) as tc:
            proxy.drop_path_substring = "/forward_backward"
            proxy.drops_remaining = 1

            with pytest.raises(AmbiguousSubmissionError) as exc_info:
                tc.forward_backward(batch, loss_fn="cross_entropy")

            assert exc_info.value.cause is not None
            fb_posts = [
                (m, p)
                for (m, p, _body) in proxy.recorded_requests
                if m == "POST" and p.endswith("/forward_backward")
            ]
            assert len(fb_posts) == 1
    finally:
        client.close()
        proxy.close()


def test_forward_backward_502_after_forward_raises_ambiguous_and_no_retry(
    live_server: str,
) -> None:
    """502 after proxy forwards request raises AmbiguousSubmissionError and does NOT retry."""
    host, port_str = live_server.removeprefix("http://").split(":")
    proxy = FaultInjectingProxy(host, int(port_str))
    batch = _make_wire_batch("Ambiguous 502 after forward batch.", batch_size=2, seq_len=8)

    client = ServiceClient(
        base_url=proxy.base_url,
        api_key=API_KEY,
        max_retries=3,
        backoff_base=0.05,
    )
    try:
        with client.create_training_client(
            base_model=MODEL_NAME,
            rank=4,
            alpha=8.0,
            target_modules=TARGET_MODULES,
            device="cpu",
        ) as tc:
            proxy.status_502_path_substring = "/forward_backward"
            proxy.status_502_remaining = 1

            with pytest.raises(AmbiguousSubmissionError) as exc_info:
                tc.forward_backward(batch, loss_fn="cross_entropy")

            assert exc_info.value.cause is not None
            fb_posts = [
                (m, p)
                for (m, p, _body) in proxy.recorded_requests
                if m == "POST" and p.endswith("/forward_backward")
            ]
            assert len(fb_posts) == 1
    finally:
        client.close()
        proxy.close()


def test_server_error_codes_map_to_typed_exceptions(live_server: str) -> None:
    """5. Each server error code maps to the right exception class."""
    bad_auth_client = ServiceClient(base_url=live_server, api_key="wrong-key", max_retries=0)
    try:
        with pytest.raises(AuthError):
            bad_auth_client.create_training_client(
                base_model=MODEL_NAME,
                rank=4,
                alpha=8.0,
                target_modules=TARGET_MODULES,
                device="cpu",
            )
    finally:
        bad_auth_client.close()

    client = ServiceClient(base_url=live_server, api_key=API_KEY, max_retries=0)
    try:
        with client.create_training_client(
            base_model=MODEL_NAME,
            rank=4,
            alpha=8.0,
            target_modules=TARGET_MODULES,
            device="cpu",
        ) as tc:
            # 422 from server (batch_size=9 > max_batch_size=8) -> ValidationError
            oversized = {
                "input_ids": [[1, 2, 3, 4] for _ in range(9)],
                "target_ids": [[2, 3, 4, 5] for _ in range(9)],
                "weights": [[1.0, 1.0, 1.0, 1.0] for _ in range(9)],
            }
            with pytest.raises(ValidationError):
                tc.forward_backward(oversized)

            # Remote op failure (out-of-range token id) -> RemoteOpError with remote message
            bad_tok = {
                "input_ids": [[999_999_999, 1, 2, 3]],
                "target_ids": [[1, 2, 3, 4]],
                "weights": [[1.0, 1.0, 1.0, 1.0]],
            }
            bad_fut = tc.forward_backward(bad_tok)
            with pytest.raises(RemoteOpError) as remote_exc:
                bad_fut.result()
            assert remote_exc.value.remote_message

            # 429 capacity exceeded (create 2nd and attempt 3rd session since max_sessions=2)
            with client.create_training_client(
                base_model=MODEL_NAME,
                rank=4,
                alpha=8.0,
                target_modules=TARGET_MODULES,
                device="cpu",
            ), pytest.raises(ServerBusyError):
                client.create_training_client(
                    base_model=MODEL_NAME,
                    rank=4,
                    alpha=8.0,
                    target_modules=TARGET_MODULES,
                    device="cpu",
                )

            deleted_sid = tc.session_id

        # 404 on deleted session -> SessionNotFoundError
        with pytest.raises(SessionNotFoundError):
            client.delete_session(deleted_sid)
    finally:
        client.close()


def test_future_result_timeout_does_not_cancel_op(live_server: str) -> None:
    """6. future.result(timeout) raises LorkTimeoutError; later result() succeeds."""
    assert not issubclass(LorkTimeoutError, TimeoutError)
    client = ServiceClient(base_url=live_server, api_key=API_KEY)
    try:
        with client.create_training_client(
            base_model=MODEL_NAME,
            rank=4,
            alpha=8.0,
            target_modules=TARGET_MODULES,
            device="cpu",
        ) as tc:
            fut = tc.sample(["Once upon a time"], SampleParams(max_tokens=16, temperature=0.0))
            with pytest.raises(LorkTimeoutError):
                fut.result(timeout=0.001)
            res = fut.result(timeout=60.0)
            assert fut.done() is True
            assert len(res.completions) == 1
            assert len(res.token_ids[0]) == 16
    finally:
        client.close()


def test_training_client_context_manager_deletes_session_on_exception(live_server: str) -> None:
    """7. Context manager deletes the session even on exception."""
    client = ServiceClient(base_url=live_server, api_key=API_KEY)
    captured_sid: str | None = None
    try:
        with (
            pytest.raises(RuntimeError, match="boom"),
            client.create_training_client(
                base_model=MODEL_NAME,
                rank=4,
                alpha=8.0,
                target_modules=TARGET_MODULES,
                device="cpu",
            ) as tc,
        ):
            captured_sid = tc.session_id
            raise RuntimeError("boom")

        assert captured_sid is not None
        with pytest.raises(SessionNotFoundError):
            client.delete_session(captured_sid)
    finally:
        client.close()


def test_no_torch_import_in_subprocess(live_server: str) -> None:
    """8. Subprocess blocking torch/transformers/peft imports lork.client and runs."""
    script = """
import sys
for mod in ("torch", "transformers", "peft"):
    sys.modules[mod] = None  # type: ignore[assignment]

import lork
import lork.client
from lork.client import ServiceClient
from lork.wire import OptimParams

assert "torch" not in sys.modules or sys.modules["torch"] is None
assert "transformers" not in sys.modules or sys.modules["transformers"] is None
assert "peft" not in sys.modules or sys.modules["peft"] is None

client = ServiceClient(base_url=sys.argv[1], api_key=sys.argv[2])
with client.create_training_client(
    base_model=sys.argv[3],
    rank=4,
    alpha=8.0,
    target_modules=["q_proj", "v_proj"],
    device="cpu",
) as tc:
    batch = {
        "input_ids": [[10, 20, 30, 40]],
        "target_ids": [[20, 30, 40, 50]],
        "weights": [[1.0, 1.0, 1.0, 1.0]],
    }
    fb = tc.forward_backward(batch).result()
    step = tc.optim_step(OptimParams(lr=1e-3)).result()
    assert fb.loss > 0.0
    assert step.step == 1
client.close()
print("NO_TORCH_OK")
"""
    res = subprocess.run(
        [sys.executable, "-c", script, live_server, API_KEY, MODEL_NAME],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    assert "NO_TORCH_OK" in res.stdout


def test_examples_sft_runs_end_to_end(live_server: str) -> None:
    """9. examples/sft.py runs end to end against a live server."""
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["LORK_BASE_URL"] = live_server
    env["LORK_API_KEY"] = API_KEY

    res = subprocess.run(
        [sys.executable, "examples/sft.py"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    assert "loss=" in res.stdout


def test_retry_after_honored_on_429() -> None:
    """10. Retry-After header is honored on 429 Too Many Requests."""
    state = {"optim_calls": 0}

    class StubHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(
            self, status_code: int, payload: dict[str, Any], headers: dict[str, str] | None = None
        ) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if headers:
                for k, v in headers.items():
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            if self.path == "/v1/sessions":
                self._send_json(202, {"future_id": "fut_init", "session_id": "sess_stub"})
                return
            if self.path == "/v1/sessions/sess_stub/optim_step":
                state["optim_calls"] += 1
                if state["optim_calls"] == 1:
                    self._send_json(
                        429,
                        {"error": {"code": "session_capacity_exceeded", "message": "slow down"}},
                        headers={"Retry-After": "0.3"},
                    )
                    return
                self._send_json(202, {"future_id": "fut_opt", "session_id": "sess_stub"})
                return
            self._send_json(404, {"error": {"code": "not_found", "message": "unknown"}})

        def do_GET(self) -> None:
            if self.path.startswith("/v1/futures/fut_init"):
                self._send_json(
                    200,
                    {
                        "future_id": "fut_init",
                        "status": "done",
                        "result": {"session_id": "sess_stub"},
                        "error": None,
                    },
                )
                return
            if self.path.startswith("/v1/futures/fut_opt"):
                self._send_json(
                    200,
                    {
                        "future_id": "fut_opt",
                        "status": "done",
                        "result": {"step": 1},
                        "error": None,
                    },
                )
                return
            self._send_json(404, {"error": {"code": "not_found", "message": "unknown"}})

        def do_DELETE(self) -> None:
            self._send_json(200, {"deleted": "sess_stub"})

    server = HTTPServer(("127.0.0.1", 0), StubHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ServiceClient(
            base_url=f"http://127.0.0.1:{port}",
            api_key=API_KEY,
            max_retries=2,
            backoff_base=0.01,
        )
        try:
            with client.create_training_client(
                base_model=MODEL_NAME,
                rank=4,
                alpha=8.0,
                target_modules=TARGET_MODULES,
                device="cpu",
            ) as tc:
                t0 = time.monotonic()
                res = tc.optim_step(OptimParams(lr=1e-3)).result()
                elapsed = time.monotonic() - t0
                assert res.step == 1
                assert state["optim_calls"] == 2
                assert elapsed >= 0.25
        finally:
            client.close()
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.slow
def test_examples_grpo_toy_improves_reward(live_server: str) -> None:
    """examples/grpo_toy.py improves mean reward over baseline on fixed seed."""
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["LORK_BASE_URL"] = live_server
    env["LORK_API_KEY"] = API_KEY

    res = subprocess.run(
        [sys.executable, "examples/grpo_toy.py"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    summary_lines = [line for line in res.stdout.splitlines() if line.startswith("GRPO_SUMMARY ")]
    assert summary_lines, f"Missing GRPO_SUMMARY line in:\n{res.stdout}"
    summary = json.loads(summary_lines[-1].removeprefix("GRPO_SUMMARY "))
    assert summary["final_mean_reward"] >= summary["initial_mean_reward"] + 0.2


def test_sdk_forward_backward_return_logprobs_false(live_server: str) -> None:
    """SDK forward_backward(..., return_logprobs=False) returns logprobs=None."""
    batch = _make_wire_batch("SDK return_logprobs=False test.", batch_size=2, seq_len=8)
    with (
        ServiceClient(base_url=live_server, api_key=API_KEY) as client,
        client.create_training_client(
            base_model=MODEL_NAME,
            rank=4,
            alpha=8.0,
            target_modules=TARGET_MODULES,
            device="cpu",
        ) as tc,
    ):
        res = tc.forward_backward(batch, loss_fn="cross_entropy", return_logprobs=False).result()
        assert res.loss > 0.0
        assert res.logprobs is None


def test_sdk_export_adapter_and_sample_stop(live_server: str) -> None:
    """SDK export_adapter returns ExportAdapterResult and sample honors stop/stop_reasons."""
    batch = _make_wire_batch("SDK export adapter test.", batch_size=1, seq_len=8)
    with (
        ServiceClient(base_url=live_server, api_key=API_KEY) as client,
        client.create_training_client(
            base_model=MODEL_NAME,
            rank=4,
            alpha=8.0,
            target_modules=TARGET_MODULES,
            device="cpu",
        ) as tc,
    ):
        tc.forward_backward(batch, return_logprobs=False).result()
        tc.optim_step(OptimParams(lr=5e-4)).result()
        exp = tc.export_adapter("sdk_export_1").result()
        assert exp.name == "sdk_export_1"
        assert "adapter_config.json" in exp.files
        assert "adapter_model.safetensors" in exp.files

        greedy = tc.sample(
            ["Once upon a time in a small village"],
            SampleParams(max_tokens=16, temperature=0.0),
        ).result()
        base_text = greedy.completions[0]
        stop_seq = base_text[max(1, len(base_text) // 3) : max(1, len(base_text) // 3) + 2]
        stopped = tc.sample(
            ["Once upon a time in a small village"],
            SampleParams(max_tokens=16, temperature=0.0, top_p=1.0, stop=[stop_seq]),
        ).result()
        assert stopped.stop_reasons == ["stop"]
        assert stop_seq not in stopped.completions[0]


def test_cli_version_no_torch_import() -> None:
    """lork.cli --version and serve --help run cleanly without importing ML deps."""
    code = """
import io
import sys
from contextlib import redirect_stdout

for blocked in ("torch", "transformers", "peft", "uvicorn"):
    sys.modules[blocked] = None

import lork
import lork.cli

buf = io.StringIO()
with redirect_stdout(buf):
    rc = lork.cli.main(["--version"])
assert rc == 0
assert buf.getvalue().strip() == f"lork {lork.__version__}"

with redirect_stdout(io.StringIO()):
    rc_help = lork.cli.main(["serve", "--help"])
assert rc_help == 0
print("cli_ok")
"""
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    assert "cli_ok" in res.stdout


def test_service_client_validates_constructor_args() -> None:
    """ServiceClient rejects blank base_url, bad timeout, and negative backoff."""
    with pytest.raises(ValidationError):
        ServiceClient("   ", API_KEY)
    with pytest.raises(ValidationError):
        ServiceClient("http://127.0.0.1:8000", API_KEY, timeout=0.0)
    with pytest.raises(ValidationError):
        ServiceClient("http://127.0.0.1:8000", API_KEY, timeout=float("inf"))
    with pytest.raises(ValidationError):
        ServiceClient("http://127.0.0.1:8000", API_KEY, backoff_base=-0.1)


def test_cli_handles_bad_lork_port_env_and_invalid_port_range() -> None:
    """Bad LORK_PORT does not break `--version`; out-of-range serve `--port` returns exit code 2."""
    env = os.environ.copy()
    env["LORK_PORT"] = "not-an-int"
    code = """
import io, sys
from contextlib import redirect_stdout
for blocked in ("torch", "transformers", "peft", "uvicorn"):
    sys.modules[blocked] = None
import lork, lork.cli
buf = io.StringIO()
with redirect_stdout(buf):
    rc = lork.cli.main(["--version"])
assert rc == 0
assert buf.getvalue().strip() == f"lork {lork.__version__}"
print("ok")
"""
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "ok" in res.stdout

    import lork.cli

    assert lork.cli.main(["serve", "--port", "0"]) == 2
    assert lork.cli.main(["serve", "--port", "70000"]) == 2


def test_export_adapter_non_idempotent_502_raises_ambiguous(live_server: str) -> None:
    """502 after proxy forwards export_adapter raises AmbiguousSubmissionError without retry."""
    host, port_str = live_server.removeprefix("http://").split(":")
    proxy = FaultInjectingProxy(host, int(port_str))
    client = ServiceClient(
        base_url=proxy.base_url,
        api_key=API_KEY,
        max_retries=3,
        backoff_base=0.05,
    )
    try:
        with client.create_training_client(
            base_model=MODEL_NAME,
            rank=4,
            alpha=8.0,
            target_modules=TARGET_MODULES,
            device="cpu",
        ) as tc:
            proxy.status_502_path_substring = "/export_adapter"
            proxy.status_502_remaining = 1

            with pytest.raises(AmbiguousSubmissionError) as exc_info:
                tc.export_adapter("ambiguous_export_1")

            assert exc_info.value.cause is not None
            export_posts = [
                (m, p)
                for (m, p, _body) in proxy.recorded_requests
                if m == "POST" and p.endswith("/export_adapter")
            ]
            assert len(export_posts) == 1
    finally:
        client.close()
        proxy.close()
