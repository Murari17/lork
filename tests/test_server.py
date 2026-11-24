"""Phase 2 tests for the async FastAPI HTTP server (written before implementation)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import torch
from fastapi.testclient import TestClient
from tests.conftest import MODEL_NAME, TARGET_MODULES
from transformers import AutoTokenizer

from lork.server.app import create_app
from lork.server.models import ServerSettings
from lork.trainer import LoraTrainer

API_KEY = "test-secret-api-key"
AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"}


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
    input_ids = tokens[:, :-1].repeat(batch_size, 1).tolist()
    target_ids = tokens[:, 1:].repeat(batch_size, 1).tolist()
    weights = [[1.0] * seq_len for _ in range(batch_size)]
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "weights": weights,
    }


def _make_settings(tmp_path: Path, **overrides: Any) -> ServerSettings:
    defaults: dict[str, Any] = {
        "api_keys": (API_KEY,),
        "max_batch_size": 8,
        "max_seq_len": 64,
        "max_sample_tokens": 32,
        "max_prompts": 8,
        "max_sessions": 4,
        "session_idle_ttl_seconds": 300.0,
        "max_futures": 500,
        "future_ttl_seconds": 300.0,
        "checkpoint_root": tmp_path / "checkpoints",
        "default_device": "cpu",
    }
    defaults.update(overrides)
    return ServerSettings(**defaults)


def _wait_future(client: TestClient, future_id: str, wait: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        resp = client.get(f"/v1/futures/{future_id}", params={"wait": wait}, headers=AUTH_HEADERS)
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        if body["status"] != "pending":
            return body
    raise TimeoutError(f"Future {future_id} remained pending past deadline")


def _create_session(
    client: TestClient,
    rank: int = 4,
    alpha: float = 8.0,
    target_modules: list[str] | None = None,
) -> str:
    resp = client.post(
        "/v1/sessions",
        json={
            "base_model": MODEL_NAME,
            "rank": rank,
            "alpha": alpha,
            "target_modules": target_modules or TARGET_MODULES,
            "device": "cpu",
        },
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 202
    submitted = resp.json()
    assert "future_id" in submitted
    done = _wait_future(client, submitted["future_id"])
    assert done["status"] == "done"
    session_id: str = done["result"]["session_id"]
    return session_id


def test_full_cycle_matches_direct_trainer(tmp_path: Path) -> None:
    """1. Full cycle over HTTP reproduces direct-LoraTrainer loss values exactly."""
    batch1 = _make_wire_batch(
        "HTTP full cycle exact equivalence batch one.", batch_size=2, seq_len=8
    )
    batch2 = _make_wire_batch(
        "HTTP full cycle exact equivalence batch two.", batch_size=2, seq_len=8
    )
    optim_params = {"lr": 5e-4, "betas": [0.9, 0.999], "weight_decay": 0.01, "eps": 1e-8}

    torch.manual_seed(101)
    direct = LoraTrainer(
        base_model=MODEL_NAME,
        rank=4,
        alpha=8.0,
        target_modules=TARGET_MODULES,
        device="cpu",
    )
    d_fb1 = direct.forward_backward(batch1, loss_fn="cross_entropy")
    d_step1 = direct.optim_step("step-1", optim_params)
    direct.save_state(tmp_path / "direct_ckpt")
    d_fb2 = direct.forward_backward(batch2, loss_fn="cross_entropy")
    d_sample = direct.sample(["Hello"], {"max_tokens": 4, "temperature": 0.0})

    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        torch.manual_seed(101)
        sid = _create_session(client, rank=4, alpha=8.0)

        r1 = client.post(
            f"/v1/sessions/{sid}/forward_backward",
            json={"batch": batch1, "loss_fn": "cross_entropy"},
            headers=AUTH_HEADERS,
        )
        assert r1.status_code == 202
        h_fb1 = _wait_future(client, r1.json()["future_id"])
        assert h_fb1["status"] == "done"
        assert h_fb1["result"]["loss"] == pytest.approx(d_fb1["loss"], abs=1e-6)
        assert torch.allclose(
            torch.tensor(h_fb1["result"]["logprobs"]),
            d_fb1["logprobs"].cpu(),
            atol=1e-6,
        )

        r_opt = client.post(
            f"/v1/sessions/{sid}/optim_step",
            json={"step_id": "step-1", "optim_params": optim_params},
            headers=AUTH_HEADERS,
        )
        assert r_opt.status_code == 202
        h_step1 = _wait_future(client, r_opt.json()["future_id"])
        assert h_step1["status"] == "done"
        assert h_step1["result"] == d_step1

        r_save = client.post(
            f"/v1/sessions/{sid}/save_state",
            json={"name": "ckpt_one"},
            headers=AUTH_HEADERS,
        )
        assert r_save.status_code == 202
        assert _wait_future(client, r_save.json()["future_id"])["status"] == "done"

        r_load = client.post(
            f"/v1/sessions/{sid}/load_state",
            json={"name": "ckpt_one"},
            headers=AUTH_HEADERS,
        )
        assert r_load.status_code == 202
        assert _wait_future(client, r_load.json()["future_id"])["status"] == "done"

        r2 = client.post(
            f"/v1/sessions/{sid}/forward_backward",
            json={"batch": batch2, "loss_fn": "cross_entropy"},
            headers=AUTH_HEADERS,
        )
        h_fb2 = _wait_future(client, r2.json()["future_id"])
        assert h_fb2["status"] == "done"
        assert h_fb2["result"]["loss"] == pytest.approx(d_fb2["loss"], abs=1e-6)

        r_samp = client.post(
            f"/v1/sessions/{sid}/sample",
            json={"prompts": ["Hello"], "params": {"max_tokens": 4, "temperature": 0.0}},
            headers=AUTH_HEADERS,
        )
        assert r_samp.status_code == 202
        h_sample = _wait_future(client, r_samp.json()["future_id"])
        assert h_sample["status"] == "done"
        assert h_sample["result"]["completions"] == d_sample["completions"]
        assert h_sample["result"]["token_ids"] == d_sample["token_ids"]


def test_fifo_submission_ordering(tmp_path: Path) -> None:
    """2. Ordering: submit fb, fb, optim_step, fb back-to-back; results reflect submission order."""
    batch = _make_wire_batch("FIFO ordering verification sequence.", batch_size=2, seq_len=8)
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        sid = _create_session(client)

        f1 = client.post(
            f"/v1/sessions/{sid}/forward_backward",
            json={"batch": batch, "loss_fn": "cross_entropy"},
            headers=AUTH_HEADERS,
        ).json()["future_id"]
        f2 = client.post(
            f"/v1/sessions/{sid}/forward_backward",
            json={"batch": batch, "loss_fn": "cross_entropy"},
            headers=AUTH_HEADERS,
        ).json()["future_id"]
        f3 = client.post(
            f"/v1/sessions/{sid}/optim_step",
            json={"step_id": "fifo-step-1", "optim_params": {"lr": 1e-3}},
            headers=AUTH_HEADERS,
        ).json()["future_id"]
        f4 = client.post(
            f"/v1/sessions/{sid}/forward_backward",
            json={"batch": batch, "loss_fn": "cross_entropy"},
            headers=AUTH_HEADERS,
        ).json()["future_id"]

        res1 = _wait_future(client, f1)
        res2 = _wait_future(client, f2)
        res3 = _wait_future(client, f3)
        res4 = _wait_future(client, f4)

        assert res1["status"] == "done"
        assert res2["status"] == "done"
        assert res3["status"] == "done"
        assert res4["status"] == "done"
        assert res3["result"] == {"step": 1}
        # First two forward_backward calls ran before optim_step, so losses match.
        assert res1["result"]["loss"] == pytest.approx(res2["result"]["loss"], abs=1e-6)
        # Fourth forward_backward ran after optim_step, so loss dropped.
        assert res4["result"]["loss"] < res1["result"]["loss"]


def test_two_sessions_interleaved_independence(tmp_path: Path) -> None:
    """3. Two sessions interleaved: adapters/optimizer states independent."""
    batch_a = _make_wire_batch("Session A unchanging sequence.", batch_size=2, seq_len=8)
    batch_b = _make_wire_batch("Session B updating weights rapidly.", batch_size=2, seq_len=8)

    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        torch.manual_seed(77)
        sid_a = _create_session(client)
        torch.manual_seed(77)
        sid_b = _create_session(client)

        fa_0 = client.post(
            f"/v1/sessions/{sid_a}/forward_backward",
            json={"batch": batch_a, "loss_fn": "cross_entropy"},
            headers=AUTH_HEADERS,
        ).json()["future_id"]
        loss_a_before = _wait_future(client, fa_0)["result"]["loss"]

        # Step session B multiple times while session A does not step.
        for idx in range(3):
            fb_b = client.post(
                f"/v1/sessions/{sid_b}/forward_backward",
                json={"batch": batch_b, "loss_fn": "cross_entropy"},
                headers=AUTH_HEADERS,
            ).json()["future_id"]
            st_b = client.post(
                f"/v1/sessions/{sid_b}/optim_step",
                json={"step_id": f"b-step-{idx}", "optim_params": {"lr": 1e-3}},
                headers=AUTH_HEADERS,
            ).json()["future_id"]
            assert _wait_future(client, fb_b)["status"] == "done"
            assert _wait_future(client, st_b)["result"] == {"step": idx + 1}

        fa_1 = client.post(
            f"/v1/sessions/{sid_a}/forward_backward",
            json={"batch": batch_a, "loss_fn": "cross_entropy"},
            headers=AUTH_HEADERS,
        ).json()["future_id"]
        loss_a_after = _wait_future(client, fa_1)["result"]["loss"]

        # Session A never ran optim_step, so its weights and loss must be unchanged.
        assert loss_a_after == loss_a_before

        # Session A's own optim_step must still start at step 1.
        st_a = client.post(
            f"/v1/sessions/{sid_a}/optim_step",
            json={"step_id": "a-step-0", "optim_params": {"lr": 1e-3}},
            headers=AUTH_HEADERS,
        ).json()["future_id"]
        assert _wait_future(client, st_a)["result"] == {"step": 1}


def test_optim_step_idempotency_and_survives_failed_op(tmp_path: Path) -> None:
    """4. optim_step replay returns stored result, weights changed once, even after failed op."""
    batch = _make_wire_batch("Idempotency and failed op survival test.", batch_size=2, seq_len=8)
    bad_batch = {
        "input_ids": [[999_999_999, 1, 2, 3]],
        "target_ids": [[1, 2, 3, 4]],
        "weights": [[1.0, 1.0, 1.0, 1.0]],
    }
    optim_cfg = {"lr": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.0, "eps": 1e-8}

    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        sid = _create_session(client)

        fb1 = _wait_future(
            client,
            client.post(
                f"/v1/sessions/{sid}/forward_backward",
                json={"batch": batch, "loss_fn": "cross_entropy"},
                headers=AUTH_HEADERS,
            ).json()["future_id"],
        )
        assert fb1["status"] == "done"

        step1_first = _wait_future(
            client,
            client.post(
                f"/v1/sessions/{sid}/optim_step",
                json={"step_id": "idem-1", "optim_params": optim_cfg},
                headers=AUTH_HEADERS,
            ).json()["future_id"],
        )
        assert step1_first["status"] == "done"
        assert step1_first["result"] == {"step": 1}

        # Replay before any failure.
        step1_replay = _wait_future(
            client,
            client.post(
                f"/v1/sessions/{sid}/optim_step",
                json={"step_id": "idem-1", "optim_params": optim_cfg},
                headers=AUTH_HEADERS,
            ).json()["future_id"],
        )
        assert step1_replay["status"] == "done"
        assert step1_replay["result"] == {"step": 1}

        # Intervening failed operation (out-of-range token ID).
        failed = _wait_future(
            client,
            client.post(
                f"/v1/sessions/{sid}/forward_backward",
                json={"batch": bad_batch, "loss_fn": "cross_entropy"},
                headers=AUTH_HEADERS,
            ).json()["future_id"],
        )
        assert failed["status"] == "error"
        assert failed["error"] is not None

        # Replay after intervening failed op still returns stored result {"step": 1}.
        step1_after_fail = _wait_future(
            client,
            client.post(
                f"/v1/sessions/{sid}/optim_step",
                json={"step_id": "idem-1", "optim_params": optim_cfg},
                headers=AUTH_HEADERS,
            ).json()["future_id"],
        )
        assert step1_after_fail["status"] == "done"
        assert step1_after_fail["result"] == {"step": 1}

        # Weights only changed once: evaluating the same batch yields identical loss.
        fb_after_replay_1 = _wait_future(
            client,
            client.post(
                f"/v1/sessions/{sid}/forward_backward",
                json={"batch": batch, "loss_fn": "cross_entropy"},
                headers=AUTH_HEADERS,
            ).json()["future_id"],
        )
        # Replaying optim_step again does not double-apply even with accumulated grads present.
        step1_replay_with_grads = _wait_future(
            client,
            client.post(
                f"/v1/sessions/{sid}/optim_step",
                json={"step_id": "idem-1", "optim_params": optim_cfg},
                headers=AUTH_HEADERS,
            ).json()["future_id"],
        )
        assert step1_replay_with_grads["result"] == {"step": 1}
        fb_after_replay_2 = _wait_future(
            client,
            client.post(
                f"/v1/sessions/{sid}/forward_backward",
                json={"batch": batch, "loss_fn": "cross_entropy"},
                headers=AUTH_HEADERS,
            ).json()["future_id"],
        )
        assert fb_after_replay_2["result"]["loss"] == fb_after_replay_1["result"]["loss"]


def test_failed_op_does_not_poison_worker(tmp_path: Path) -> None:
    """5. A failing op (out-of-range token id) yields an error future; the next op succeeds."""
    bad_batch = {
        "input_ids": [[999_999_999, 1, 2, 3]],
        "target_ids": [[1, 2, 3, 4]],
        "weights": [[1.0, 1.0, 1.0, 1.0]],
    }
    good_batch = _make_wire_batch("Valid batch after error.", batch_size=2, seq_len=8)

    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        sid = _create_session(client)

        bad_resp = client.post(
            f"/v1/sessions/{sid}/forward_backward",
            json={"batch": bad_batch, "loss_fn": "cross_entropy"},
            headers=AUTH_HEADERS,
        )
        assert bad_resp.status_code == 202
        bad_future = _wait_future(client, bad_resp.json()["future_id"])
        assert bad_future["status"] == "error"
        assert bad_future["error"] is not None
        assert "code" in bad_future["error"]
        assert "message" in bad_future["error"]

        good_fb = _wait_future(
            client,
            client.post(
                f"/v1/sessions/{sid}/forward_backward",
                json={"batch": good_batch, "loss_fn": "cross_entropy"},
                headers=AUTH_HEADERS,
            ).json()["future_id"],
        )
        assert good_fb["status"] == "done"

        good_step = _wait_future(
            client,
            client.post(
                f"/v1/sessions/{sid}/optim_step",
                json={"step_id": "recover-1", "optim_params": {"lr": 1e-3}},
                headers=AUTH_HEADERS,
            ).json()["future_id"],
        )
        assert good_step["status"] == "done"
        assert good_step["result"] == {"step": 1}


def test_path_traversal_and_symlink_rejection(tmp_path: Path) -> None:
    """6. Path traversal and symlink escaping checkpoint dir are rejected on save/load."""
    settings = _make_settings(tmp_path)
    app = create_app(settings)
    outside_dir = tmp_path / "outside_secret"
    outside_dir.mkdir(parents=True, exist_ok=True)

    with TestClient(app) as client:
        sid = _create_session(client)

        for bad_name in ("../x", "/abs/path", "a/b"):
            for endpoint in ("save_state", "load_state"):
                resp = client.post(
                    f"/v1/sessions/{sid}/{endpoint}",
                    json={"name": bad_name},
                    headers=AUTH_HEADERS,
                )
                assert resp.status_code == 422
                assert "error" in resp.json()

        # Create a symlink inside the session checkpoint dir pointing outside.
        session_ckpt_dir = settings.checkpoint_root.resolve() / sid
        session_ckpt_dir.mkdir(parents=True, exist_ok=True)
        evil_symlink = session_ckpt_dir / "evil_link"
        evil_symlink.symlink_to(outside_dir)

        for endpoint in ("save_state", "load_state"):
            resp = client.post(
                f"/v1/sessions/{sid}/{endpoint}",
                json={"name": "evil_link"},
                headers=AUTH_HEADERS,
            )
            if resp.status_code == 202:
                done = _wait_future(client, resp.json()["future_id"])
                assert done["status"] == "error"
            else:
                assert resp.status_code in (400, 403, 422)
                assert "error" in resp.json()


def test_oversized_batch_422_and_auth_401(tmp_path: Path) -> None:
    """7. Oversized batch -> 422. Missing/bad API key -> 401."""
    settings = _make_settings(tmp_path, max_batch_size=2, max_seq_len=8, max_sample_tokens=4)
    app = create_app(settings)
    with TestClient(app) as client:
        # Missing and invalid API keys -> 401.
        r_no_auth = client.post(
            "/v1/sessions",
            json={
                "base_model": MODEL_NAME,
                "rank": 4,
                "alpha": 8.0,
                "target_modules": TARGET_MODULES,
            },
        )
        assert r_no_auth.status_code == 401
        assert r_no_auth.json()["error"]["code"] == "unauthorized"

        r_bad_auth = client.post(
            "/v1/sessions",
            json={
                "base_model": MODEL_NAME,
                "rank": 4,
                "alpha": 8.0,
                "target_modules": TARGET_MODULES,
            },
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert r_bad_auth.status_code == 401
        assert r_bad_auth.json()["error"]["code"] == "unauthorized"

        sid = _create_session(client)

        # Oversized batch size (3 > max_batch_size=2) -> 422.
        over_bs = _make_wire_batch("Oversized batch size.", batch_size=3, seq_len=6)
        r_over_bs = client.post(
            f"/v1/sessions/{sid}/forward_backward",
            json={"batch": over_bs, "loss_fn": "cross_entropy"},
            headers=AUTH_HEADERS,
        )
        assert r_over_bs.status_code == 422
        assert "error" in r_over_bs.json()

        # Oversized sequence length (10 > max_seq_len=8) -> 422.
        over_seq = _make_wire_batch("Oversized sequence length.", batch_size=1, seq_len=10)
        r_over_seq = client.post(
            f"/v1/sessions/{sid}/forward_backward",
            json={"batch": over_seq, "loss_fn": "cross_entropy"},
            headers=AUTH_HEADERS,
        )
        assert r_over_seq.status_code == 422
        assert "error" in r_over_seq.json()

        # Oversized sample max_tokens (8 > max_sample_tokens=4) -> 422.
        r_over_sample = client.post(
            f"/v1/sessions/{sid}/sample",
            json={"prompts": ["Hello"], "params": {"max_tokens": 8, "temperature": 0.0}},
            headers=AUTH_HEADERS,
        )
        assert r_over_sample.status_code == 422
        assert "error" in r_over_sample.json()


def test_ttl_eviction_frees_session_and_returns_404(tmp_path: Path) -> None:
    """8. TTL eviction frees the session; further calls -> 404."""
    settings = _make_settings(tmp_path, session_idle_ttl_seconds=0.3)
    app = create_app(settings)
    batch = _make_wire_batch("TTL eviction test batch.", batch_size=1, seq_len=6)

    with TestClient(app) as client:
        sid = _create_session(client)
        time.sleep(0.5)

        r_after = client.post(
            f"/v1/sessions/{sid}/forward_backward",
            json={"batch": batch, "loss_fn": "cross_entropy"},
            headers=AUTH_HEADERS,
        )
        assert r_after.status_code == 404
        assert "error" in r_after.json()


def test_long_poll_returns_early_on_completion(tmp_path: Path) -> None:
    """9. Long-poll returns early when the future completes."""
    batch = _make_wire_batch("Long-poll early wakeup test.", batch_size=2, seq_len=8)
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        sid = _create_session(client)
        fid = client.post(
            f"/v1/sessions/{sid}/forward_backward",
            json={"batch": batch, "loss_fn": "cross_entropy"},
            headers=AUTH_HEADERS,
        ).json()["future_id"]

        t0 = time.monotonic()
        resp = client.get(
            f"/v1/futures/{fid}",
            params={"wait": 25.0},
            headers=AUTH_HEADERS,
        )
        elapsed = time.monotonic() - t0
        assert resp.status_code == 200
        assert resp.json()["status"] == "done"
        assert elapsed < 15.0

        # Invalid wait > 30 must return 422.
        r_bad_wait = client.get(
            f"/v1/futures/{fid}",
            params={"wait": 35.0},
            headers=AUTH_HEADERS,
        )
        assert r_bad_wait.status_code == 422


def test_session_capacity_limit_returns_429(tmp_path: Path) -> None:
    """Session creation beyond ServerSettings.max_sessions (default 2) returns 429."""
    default_settings = ServerSettings(
        api_keys=(API_KEY,),
        checkpoint_root=tmp_path / "checkpoints",
    )
    assert default_settings.max_sessions == 2

    app = create_app(default_settings)
    with TestClient(app) as client:
        _create_session(client)
        _create_session(client)

        over_cap = client.post(
            "/v1/sessions",
            json={
                "base_model": MODEL_NAME,
                "rank": 4,
                "alpha": 8.0,
                "target_modules": TARGET_MODULES,
                "device": "cpu",
            },
            headers=AUTH_HEADERS,
        )
        assert over_cap.status_code == 429
        body = over_cap.json()
        assert body["error"]["code"] == "session_capacity_exceeded"
        assert "message" in body["error"]


def test_ttl_eviction_does_not_evict_busy_or_queued_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Idle TTL eviction must NOT evict a session with running or queued ops."""
    batch = _make_wire_batch("Busy session TTL test.", batch_size=1, seq_len=6)
    settings = _make_settings(tmp_path, session_idle_ttl_seconds=0.25)
    app = create_app(settings)

    with TestClient(app) as client:
        sid = _create_session(client)

        def _slow_fb(
            self: LoraTrainer,
            batch: dict[str, Any] | Any,
            loss_fn: str | torch.Tensor | list[list[float]] = "cross_entropy",
            return_logprobs: bool = True,
        ) -> dict[str, Any]:
            time.sleep(0.45)
            out: dict[str, Any] = {"loss": 1.0}
            if return_logprobs:
                out["logprobs"] = torch.zeros((1, 6))
            return out

        monkeypatch.setattr(LoraTrainer, "forward_backward", _slow_fb)

        f1 = client.post(
            f"/v1/sessions/{sid}/forward_backward",
            json={"batch": batch, "loss_fn": "cross_entropy"},
            headers=AUTH_HEADERS,
        ).json()["future_id"]
        f2 = client.post(
            f"/v1/sessions/{sid}/forward_backward",
            json={"batch": batch, "loss_fn": "cross_entropy"},
            headers=AUTH_HEADERS,
        ).json()["future_id"]

        # Sleep past session_idle_ttl_seconds while f1 is running and f2 is queued.
        time.sleep(0.35)
        r_still_alive = client.post(
            f"/v1/sessions/{sid}/forward_backward",
            json={"batch": batch, "loss_fn": "cross_entropy"},
            headers=AUTH_HEADERS,
        )
        assert r_still_alive.status_code == 202
        f3 = r_still_alive.json()["future_id"]

        assert _wait_future(client, f1)["status"] == "done"
        assert _wait_future(client, f2)["status"] == "done"
        assert _wait_future(client, f3)["status"] == "done"


@pytest.mark.anyio
async def test_event_loop_responsive_during_slow_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """10. Event loop stays responsive (healthz answers) while a slow op runs."""
    def _slow_forward_backward(
        self: LoraTrainer,
        batch: dict[str, Any] | Any,
        loss_fn: str | torch.Tensor | list[list[float]] = "cross_entropy",
        return_logprobs: bool = True,
    ) -> dict[str, Any]:
        time.sleep(1.5)
        out: dict[str, Any] = {"loss": 1.0}
        if return_logprobs:
            out["logprobs"] = torch.zeros((4, 16))
        return out

    monkeypatch.setattr(LoraTrainer, "forward_backward", _slow_forward_backward)
    app = create_app(_make_settings(tmp_path))
    batch = _make_wire_batch("Slow op concurrency batch.", batch_size=4, seq_len=16)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            create_resp = await client.post(
                "/v1/sessions",
                json={
                    "base_model": MODEL_NAME,
                    "rank": 4,
                    "alpha": 8.0,
                    "target_modules": TARGET_MODULES,
                    "device": "cpu",
                },
                headers=AUTH_HEADERS,
            )
            assert create_resp.status_code == 202
            init_fid = create_resp.json()["future_id"]
            init_done = await client.get(
                f"/v1/futures/{init_fid}",
                params={"wait": 30.0},
                headers=AUTH_HEADERS,
            )
            assert init_done.json()["status"] == "done"
            sid = init_done.json()["result"]["session_id"]

            # Submit a slow forward_backward op (1.5s sleep on the worker thread).
            r = await client.post(
                f"/v1/sessions/{sid}/forward_backward",
                json={"batch": batch, "loss_fn": "cross_entropy"},
                headers=AUTH_HEADERS,
            )
            assert r.status_code == 202
            fid = r.json()["future_id"]

            # Immediately poll /v1/healthz while the worker thread is running.
            t0 = time.monotonic()
            health_resps = await asyncio.gather(
                client.get("/v1/healthz"),
                client.get("/v1/healthz"),
                client.get("/v1/healthz"),
            )
            health_elapsed = time.monotonic() - t0
            for hr in health_resps:
                assert hr.status_code == 200
                assert hr.json() == {"status": "ok"}
            assert health_elapsed < 0.5

            # Confirm the slow op is still pending when healthz finishes.
            mid_status = await client.get(f"/v1/futures/{fid}", headers=AUTH_HEADERS)
            assert mid_status.status_code == 200
            assert mid_status.json()["status"] == "pending"

            # Drain the submitted future before shutdown.
            done = await client.get(
                f"/v1/futures/{fid}",
                params={"wait": 30.0},
                headers=AUTH_HEADERS,
            )
            assert done.json()["status"] == "done"


def test_server_refuses_start_without_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """ServerSettings.from_env and create_app refuse to start without LORK_API_KEYS."""
    monkeypatch.delenv("LORK_API_KEYS", raising=False)
    with pytest.raises(ValueError, match="LORK_API_KEYS"):
        ServerSettings.from_env()
    with pytest.raises(ValueError, match="LORK_API_KEYS"):
        create_app()

    monkeypatch.setenv("LORK_API_KEYS", "   ,  ")
    with pytest.raises(ValueError, match="LORK_API_KEYS"):
        create_app()


def test_server_defaults_bind_localhost_and_no_cors(tmp_path: Path) -> None:
    """Server defaults host to 127.0.0.1 and does not enable CORS headers."""
    settings = ServerSettings(
        api_keys=(API_KEY,),
        checkpoint_root=tmp_path / "checkpoints",
    )
    assert settings.host == "127.0.0.1"

    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get(
            "/v1/healthz",
            headers={"Origin": "https://evil.example"},
        )
        assert resp.status_code == 200
        assert "access-control-allow-origin" not in {k.lower() for k in resp.headers}


def test_request_body_size_limit_enforced_before_json_parse(tmp_path: Path) -> None:
    """Body size limit rejects oversized payloads (even invalid JSON) with 413 before parsing."""
    settings = _make_settings(tmp_path, max_request_body_bytes=256)
    app = create_app(settings)
    with TestClient(app) as client:
        oversized_malformed = b"{not-json" + b"x" * 300
        resp = client.post(
            "/v1/sessions",
            content=oversized_malformed,
            headers={**AUTH_HEADERS, "Content-Type": "application/json"},
        )
        assert resp.status_code == 413
        body = resp.json()
        assert body["error"]["code"] == "request_too_large"


def test_request_body_size_limit_chunked_no_content_length(tmp_path: Path) -> None:
    """Body size limit enforces cap on streaming/chunked bodies with no Content-Length."""
    settings = _make_settings(tmp_path, max_request_body_bytes=256)
    app = create_app(settings)

    def _chunk_stream() -> Iterator[bytes]:
        yield b'{"not_json": "'
        yield b"a" * 200
        yield b"b" * 200
        yield b'"}'

    with TestClient(app) as client:
        resp = client.post(
            "/v1/sessions",
            content=_chunk_stream(),
            headers={**AUTH_HEADERS, "Content-Type": "application/json"},
        )
        assert resp.status_code == 413
        body = resp.json()
        assert body["error"]["code"] == "request_too_large"


def test_forward_backward_return_logprobs_false_omits_logprobs(tmp_path: Path) -> None:
    """HTTP forward_backward with return_logprobs=false omits logprobs from future result."""
    batch = _make_wire_batch("Omit logprobs HTTP test.", batch_size=2, seq_len=8)
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        sid = _create_session(client)
        r = client.post(
            f"/v1/sessions/{sid}/forward_backward",
            json={"batch": batch, "loss_fn": "cross_entropy", "return_logprobs": False},
            headers=AUTH_HEADERS,
        )
        assert r.status_code == 202
        done = _wait_future(client, r.json()["future_id"])
        assert done["status"] == "done"
        assert "logprobs" not in done["result"]
        assert isinstance(done["result"]["loss"], float)
        assert done["result"]["loss"] > 0.0


def test_export_adapter_endpoint_and_symlink_containment(tmp_path: Path) -> None:
    """export_adapter writes PEFT files under exports/, rejects traversal and outside symlinks."""
    ckpt_root = tmp_path / "ckpts"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(_make_settings(tmp_path, checkpoint_root=ckpt_root))

    with TestClient(app) as client:
        sid = _create_session(client)

        r = client.post(
            f"/v1/sessions/{sid}/export_adapter",
            json={"name": "prod_v1"},
            headers=AUTH_HEADERS,
        )
        assert r.status_code == 202
        done = _wait_future(client, r.json()["future_id"])
        assert done["status"] == "done"
        assert done["result"]["name"] == "prod_v1"
        assert "adapter_config.json" in done["result"]["files"]
        assert "adapter_model.safetensors" in done["result"]["files"]
        export_path = Path(done["result"]["path"])
        assert export_path.is_relative_to(ckpt_root.resolve() / sid / "exports")
        assert (export_path / "adapter_config.json").is_file()

        r_trav = client.post(
            f"/v1/sessions/{sid}/export_adapter",
            json={"name": "../escape"},
            headers=AUTH_HEADERS,
        )
        assert r_trav.status_code == 422

        exports_dir = ckpt_root.resolve() / sid / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        (exports_dir / "evil_link").symlink_to(outside_dir, target_is_directory=True)
        r_sym = client.post(
            f"/v1/sessions/{sid}/export_adapter",
            json={"name": "evil_link"},
            headers=AUTH_HEADERS,
        )
        assert r_sym.status_code == 403
        assert r_sym.json()["error"]["code"] == "invalid_checkpoint_path"


def test_sample_stop_and_top_p_over_http(tmp_path: Path) -> None:
    """POST /v1/sessions/{sid}/sample accepts top_p/stop, returns stop_reasons, validates 422."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        sid = _create_session(client)
        r_bad = client.post(
            f"/v1/sessions/{sid}/sample",
            json={"prompts": ["Hello"], "params": {"max_tokens": 4, "top_p": 1.5}},
            headers=AUTH_HEADERS,
        )
        assert r_bad.status_code == 422

        r_ok = client.post(
            f"/v1/sessions/{sid}/sample",
            json={
                "prompts": ["Once upon a time"],
                "params": {"max_tokens": 12, "temperature": 0.7, "top_p": 0.9, "stop": ["\n"]},
            },
            headers=AUTH_HEADERS,
        )
        assert r_ok.status_code == 202
        done = _wait_future(client, r_ok.json()["future_id"])
        assert done["status"] == "done"
        assert done["result"]["stop_reasons"] in (["length"], ["eos"], ["stop"])


def test_failed_session_init_releases_capacity_slot(tmp_path: Path) -> None:
    """Failed session init frees its slot under max_sessions instead of permanently leaking."""
    app = create_app(_make_settings(tmp_path, max_sessions=1))
    with TestClient(app) as client:
        bad_init = client.post(
            "/v1/sessions",
            json={
                "base_model": MODEL_NAME,
                "rank": 4,
                "alpha": 8.0,
                "target_modules": ["nonexistent_module_xyz"],
                "device": "cpu",
            },
            headers=AUTH_HEADERS,
        )
        assert bad_init.status_code == 202
        bad_done = _wait_future(client, bad_init.json()["future_id"])
        assert bad_done["status"] == "error"

        # Slot must be released immediately so creating a valid session returns 202, not 429.
        good_init = client.post(
            "/v1/sessions",
            json={
                "base_model": MODEL_NAME,
                "rank": 4,
                "alpha": 8.0,
                "target_modules": TARGET_MODULES,
                "device": "cpu",
            },
            headers=AUTH_HEADERS,
        )
        assert good_init.status_code == 202
        good_done = _wait_future(client, good_init.json()["future_id"])
        assert good_done["status"] == "done"


def test_load_state_resets_optim_step_idempotency_cache(tmp_path: Path) -> None:
    """load_state rollback clears later cached step_ids so reusing a rolled-back step_id applies."""
    batch = _make_wire_batch("Load state idempotency reset test.", batch_size=2, seq_len=8)
    optim_cfg = {"lr": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.0}
    app = create_app(_make_settings(tmp_path))

    with TestClient(app) as client:
        sid = _create_session(client)
        _wait_future(
            client,
            client.post(
                f"/v1/sessions/{sid}/forward_backward",
                json={"batch": batch, "loss_fn": "cross_entropy"},
                headers=AUTH_HEADERS,
            ).json()["future_id"],
        )
        s1 = _wait_future(
            client,
            client.post(
                f"/v1/sessions/{sid}/optim_step",
                json={"step_id": "s1", "optim_params": optim_cfg},
                headers=AUTH_HEADERS,
            ).json()["future_id"],
        )
        assert s1["result"] == {"step": 1}

        assert _wait_future(
            client,
            client.post(
                f"/v1/sessions/{sid}/save_state",
                json={"name": "cp1"},
                headers=AUTH_HEADERS,
            ).json()["future_id"],
        )["status"] == "done"

        loss_before_s2 = _wait_future(
            client,
            client.post(
                f"/v1/sessions/{sid}/forward_backward",
                json={"batch": batch, "loss_fn": "cross_entropy"},
                headers=AUTH_HEADERS,
            ).json()["future_id"],
        )["result"]["loss"]
        s2_first = _wait_future(
            client,
            client.post(
                f"/v1/sessions/{sid}/optim_step",
                json={"step_id": "s2", "optim_params": optim_cfg},
                headers=AUTH_HEADERS,
            ).json()["future_id"],
        )
        assert s2_first["result"] == {"step": 2}

        # Roll back to cp1 (taken before step_id="s2").
        assert _wait_future(
            client,
            client.post(
                f"/v1/sessions/{sid}/load_state",
                json={"name": "cp1"},
                headers=AUTH_HEADERS,
            ).json()["future_id"],
        )["status"] == "done"

        _wait_future(
            client,
            client.post(
                f"/v1/sessions/{sid}/forward_backward",
                json={"batch": batch, "loss_fn": "cross_entropy"},
                headers=AUTH_HEADERS,
            ).json()["future_id"],
        )
        # Re-submitting step_id="s2" after rollback must apply the step, changing weights/loss.
        s2_after_load = _wait_future(
            client,
            client.post(
                f"/v1/sessions/{sid}/optim_step",
                json={"step_id": "s2", "optim_params": optim_cfg},
                headers=AUTH_HEADERS,
            ).json()["future_id"],
        )
        assert s2_after_load["result"] == {"step": 2}

        loss_after_s2 = _wait_future(
            client,
            client.post(
                f"/v1/sessions/{sid}/forward_backward",
                json={"batch": batch, "loss_fn": "cross_entropy"},
                headers=AUTH_HEADERS,
            ).json()["future_id"],
        )["result"]["loss"]
        assert loss_after_s2 < loss_before_s2


def test_server_rejects_empty_prompt_and_negative_tokens_with_422(tmp_path: Path) -> None:
    """Server rejects empty sample prompts and negative input/target token IDs with 422."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        sid = _create_session(client)
        r_empty_prompt = client.post(
            f"/v1/sessions/{sid}/sample",
            json={"prompts": [""], "params": {"max_tokens": 4}},
            headers=AUTH_HEADERS,
        )
        assert r_empty_prompt.status_code == 422

        r_neg_in = client.post(
            f"/v1/sessions/{sid}/forward_backward",
            json={
                "batch": {
                    "input_ids": [[-1, 2]],
                    "target_ids": [[2, 3]],
                    "weights": [[1.0, 1.0]],
                },
                "loss_fn": "cross_entropy",
            },
            headers=AUTH_HEADERS,
        )
        assert r_neg_in.status_code == 422

        r_neg_tgt = client.post(
            f"/v1/sessions/{sid}/forward_backward",
            json={
                "batch": {
                    "input_ids": [[1, 2]],
                    "target_ids": [[2, -100]],
                    "weights": [[1.0, 1.0]],
                },
                "loss_fn": "cross_entropy",
            },
            headers=AUTH_HEADERS,
        )
        assert r_neg_tgt.status_code == 422
