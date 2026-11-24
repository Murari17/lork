"""Phase 1 tests for LoraTrainer core (written before implementation)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from pydantic import ValidationError
from tests.conftest import TARGET_MODULES
from transformers import AutoModelForCausalLM, AutoTokenizer

from lork.trainer import LoraTrainer
from lork.wire import SampleParams


def _make_toy_batch(
    tokenizer_name: str,
    text: str,
    batch_size: int = 2,
    seq_len: int = 10,
) -> dict[str, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoded = tokenizer(text, return_tensors="pt")["input_ids"]
    while encoded.shape[1] < seq_len + 1:
        encoded = torch.cat([encoded, encoded], dim=1)
    tokens = encoded[:, : seq_len + 1]
    input_ids = tokens[:, :-1].repeat(batch_size, 1)
    target_ids = tokens[:, 1:].repeat(batch_size, 1)
    weights = torch.ones_like(input_ids, dtype=torch.float32)
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "weights": weights,
    }


def _snapshot_lora_state(trainer: LoraTrainer) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone().cpu()
        for name, param in trainer.model.named_parameters()
        if param.requires_grad
    }


def test_50_step_toy_sft_loss_drop(model_name: str, test_device: str) -> None:
    """1. 50-step toy SFT: loss drops by >= 50%."""
    torch.manual_seed(1234)
    trainer = LoraTrainer(
        base_model=model_name,
        rank=8,
        alpha=16,
        target_modules=TARGET_MODULES,
        device=test_device,
    )
    batch = _make_toy_batch(
        model_name,
        "The quick brown fox jumps over the lazy dog near the river bank.",
        batch_size=2,
        seq_len=12,
    )

    losses: list[float] = []
    for step_idx in range(50):
        out = trainer.forward_backward(batch, loss_fn="cross_entropy")
        losses.append(out["loss"])
        trainer.optim_step(
            step_id=f"sft-{step_idx}",
            optim_params={"lr": 5e-4, "betas": (0.9, 0.999), "weight_decay": 0.0},
        )

    assert losses[-1] <= 0.5 * losses[0], (
        f"Expected >=50% loss drop, got initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    )


def test_equivalence_with_plain_peft_adamw(model_name: str, test_device: str) -> None:
    """2. Equivalence: same seed, same data, LoraTrainer matches plain PEFT+AdamW within 1e-4."""
    batch = _make_toy_batch(
        model_name,
        "LoRA fine-tuning updates low-rank adapter matrices efficiently.",
        batch_size=2,
        seq_len=10,
    )
    optim_cfg = {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 0.01}

    torch.manual_seed(42)
    trainer = LoraTrainer(
        base_model=model_name,
        rank=4,
        alpha=8,
        target_modules=TARGET_MODULES,
        device=test_device,
    )
    trainer_losses: list[float] = []
    for i in range(5):
        res = trainer.forward_backward(batch, loss_fn="cross_entropy")
        trainer_losses.append(res["loss"])
        trainer.optim_step(step_id=f"eq-{i}", optim_params=optim_cfg)

    torch.manual_seed(42)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    for param in base_model.parameters():
        param.requires_grad_(False)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
    )
    ref_model = get_peft_model(base_model, lora_cfg).to(test_device)
    ref_model.train()
    ref_params = [p for p in ref_model.parameters() if p.requires_grad]
    ref_opt = torch.optim.AdamW(
        ref_params,
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    ref_input_ids = batch["input_ids"].to(test_device)
    ref_target_ids = batch["target_ids"].to(test_device)
    ref_weights = batch["weights"].to(test_device)

    ref_losses: list[float] = []
    for _ in range(5):
        ref_opt.zero_grad(set_to_none=True)
        logits = ref_model(input_ids=ref_input_ids).logits.float()
        log_probs = F.log_softmax(logits, dim=-1)
        tok_lp = log_probs.gather(dim=-1, index=ref_target_ids.unsqueeze(-1)).squeeze(-1)
        loss = (-tok_lp * ref_weights).sum() / ref_weights.sum()
        loss.backward()
        ref_opt.step()
        ref_losses.append(float(loss.item()))

    for step_idx, (t_loss, r_loss) in enumerate(zip(trainer_losses, ref_losses, strict=True)):
        assert abs(t_loss - r_loss) < 1e-4, (
            f"Step {step_idx} loss mismatch: trainer={t_loss}, ref={r_loss}"
        )


def test_grad_accumulation_equivalence(
    model_name: str, test_device: str, tmp_path: Path
) -> None:
    """3. Grad accumulation: 2 micro-batches of size N == 1 batch of size 2N."""
    torch.manual_seed(7)
    trainer_accum = LoraTrainer(
        base_model=model_name,
        rank=4,
        alpha=8,
        target_modules=TARGET_MODULES,
        device=test_device,
    )
    init_dir = tmp_path / "init_state"
    trainer_accum.save_state(init_dir)

    trainer_single = LoraTrainer(
        base_model=model_name,
        rank=4,
        alpha=8,
        target_modules=TARGET_MODULES,
        device=test_device,
    )
    trainer_single.load_state(init_dir)

    mb1 = _make_toy_batch(
        model_name,
        "First micro-batch sequence for gradient accumulation testing.",
        batch_size=2,
        seq_len=8,
    )
    mb2 = _make_toy_batch(
        model_name,
        "Second micro-batch sequence with masked padding tokens here.",
        batch_size=2,
        seq_len=8,
    )
    # Use non-uniform weights across micro-batches to verify exact weighted normalization.
    mb2["weights"][:, -3:] = 0.0

    full_batch = {
        "input_ids": torch.cat([mb1["input_ids"], mb2["input_ids"]], dim=0),
        "target_ids": torch.cat([mb1["target_ids"], mb2["target_ids"]], dim=0),
        "weights": torch.cat([mb1["weights"], mb2["weights"]], dim=0),
    }

    optim_cfg = {"lr": 5e-4, "betas": (0.9, 0.999), "weight_decay": 0.0, "eps": 1e-3}

    trainer_accum.forward_backward(mb1, loss_fn="cross_entropy")
    trainer_accum.forward_backward(mb2, loss_fn="cross_entropy")
    trainer_single.forward_backward(full_batch, loss_fn="cross_entropy")

    accum_grads = [
        p.grad.detach().cpu() / trainer_accum._accum_weight_sum
        for p in trainer_accum._trainable_params
        if p.grad is not None
    ]
    single_grads = [
        p.grad.detach().cpu() / trainer_single._accum_weight_sum
        for p in trainer_single._trainable_params
        if p.grad is not None
    ]
    assert len(accum_grads) == len(single_grads) > 0
    for idx, (g_acc, g_single) in enumerate(zip(accum_grads, single_grads, strict=True)):
        max_grad_diff = (g_acc - g_single).abs().max().item()
        assert max_grad_diff < 1e-5, (
            f"Accumulated grad {idx} diverged: max_diff={max_grad_diff}"
        )

    trainer_accum.optim_step(step_id="accum-step-1", optim_params=optim_cfg)
    trainer_single.optim_step(step_id="single-step-1", optim_params=optim_cfg)

    accum_weights = _snapshot_lora_state(trainer_accum)
    single_weights = _snapshot_lora_state(trainer_single)
    for name in accum_weights:
        max_diff = (accum_weights[name] - single_weights[name]).abs().max().item()
        assert max_diff < 1e-5, f"Weight {name} diverged after grad accum: max_diff={max_diff}"

    eval_accum = trainer_accum.forward_backward(full_batch, loss_fn="cross_entropy")
    eval_single = trainer_single.forward_backward(full_batch, loss_fn="cross_entropy")
    assert abs(eval_accum["loss"] - eval_single["loss"]) < 1e-5


def test_adam_cold_start_sensitivity(
    model_name: str, test_device: str, tmp_path: Path
) -> None:
    """Document fp32 GEMM rounding + cold-start AdamW eps=1e-8 behavior on near-zero grads."""
    torch.manual_seed(7)
    trainer_accum = LoraTrainer(
        base_model=model_name,
        rank=4,
        alpha=8,
        target_modules=TARGET_MODULES,
        device=test_device,
    )
    init_dir = tmp_path / "init_state"
    trainer_accum.save_state(init_dir)

    trainer_single = LoraTrainer(
        base_model=model_name,
        rank=4,
        alpha=8,
        target_modules=TARGET_MODULES,
        device=test_device,
    )
    trainer_single.load_state(init_dir)

    mb1 = _make_toy_batch(
        model_name,
        "First micro-batch sequence for gradient accumulation testing.",
        batch_size=2,
        seq_len=8,
    )
    mb2 = _make_toy_batch(
        model_name,
        "Second micro-batch sequence with masked padding tokens here.",
        batch_size=2,
        seq_len=8,
    )
    mb2["weights"][:, -3:] = 0.0

    full_batch = {
        "input_ids": torch.cat([mb1["input_ids"], mb2["input_ids"]], dim=0),
        "target_ids": torch.cat([mb1["target_ids"], mb2["target_ids"]], dim=0),
        "weights": torch.cat([mb1["weights"], mb2["weights"]], dim=0),
    }

    lr = 5e-4
    optim_cfg = {"lr": lr, "betas": (0.9, 0.999), "weight_decay": 0.0, "eps": 1e-8}

    trainer_accum.forward_backward(mb1, loss_fn="cross_entropy")
    trainer_accum.forward_backward(mb2, loss_fn="cross_entropy")
    trainer_single.forward_backward(full_batch, loss_fn="cross_entropy")

    accum_grads = {
        name: param.grad.detach().cpu() / trainer_accum._accum_weight_sum
        for name, param in trainer_accum.model.named_parameters()
        if param.requires_grad and param.grad is not None
    }
    single_grads = {
        name: param.grad.detach().cpu() / trainer_single._accum_weight_sum
        for name, param in trainer_single.model.named_parameters()
        if param.requires_grad and param.grad is not None
    }

    trainer_accum.optim_step(step_id="cold-accum-1", optim_params=optim_cfg)
    trainer_single.optim_step(step_id="cold-single-1", optim_params=optim_cfg)

    accum_weights = _snapshot_lora_state(trainer_accum)
    single_weights = _snapshot_lora_state(trainer_single)

    for name in accum_weights:
        g_acc = accum_grads[name]
        g_single = single_grads[name]
        diff = (accum_weights[name] - single_weights[name]).abs()
        significant_mask = torch.maximum(g_acc.abs(), g_single.abs()) > 1e-6
        if bool(significant_mask.any().item()):
            sig_max_diff = diff[significant_mask].max().item()
            assert sig_max_diff < 1e-5, (
                f"Weight {name} diverged on non-tiny gradients: max_diff={sig_max_diff}"
            )
        near_zero_mask = ~significant_mask
        if bool(near_zero_mask.any().item()):
            near_zero_max_diff = diff[near_zero_mask].max().item()
            assert near_zero_max_diff <= 2.0 * lr, (
                f"Weight {name} exceeded 2*lr on near-zero grads: {near_zero_max_diff}"
            )


def test_optim_step_idempotency(model_name: str, test_device: str) -> None:
    """4. optim_step called twice with same step_id changes weights once."""
    torch.manual_seed(99)
    trainer = LoraTrainer(
        base_model=model_name,
        rank=4,
        alpha=8,
        target_modules=TARGET_MODULES,
        device=test_device,
    )
    batch = _make_toy_batch(model_name, "Idempotency test sequence.", batch_size=2, seq_len=8)
    optim_cfg = {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 0.0}

    before_weights = _snapshot_lora_state(trainer)
    trainer.forward_backward(batch, loss_fn="cross_entropy")

    first_res = trainer.optim_step(step_id="step-idempotent-1", optim_params=optim_cfg)
    after_first_weights = _snapshot_lora_state(trainer)

    # Verify first call actually updated weights (lora_B starts at 0, so after 1 step it changes).
    changed = any(
        not torch.equal(before_weights[name], after_first_weights[name])
        for name in before_weights
    )
    assert changed, "Expected first optim_step to update LoRA weights"

    second_res = trainer.optim_step(step_id="step-idempotent-1", optim_params=optim_cfg)
    after_second_weights = _snapshot_lora_state(trainer)

    assert first_res == {"step": 1}
    assert second_res == {"step": 1}
    for name in after_first_weights:
        assert torch.equal(after_first_weights[name], after_second_weights[name]), (
            f"Weight {name} changed on duplicate optim_step call"
        )


def test_save_load_bit_identical_resume(
    model_name: str, test_device: str, tmp_path: Path
) -> None:
    """5. save -> load -> next step loss identical to uninterrupted run."""
    torch.manual_seed(2025)
    trainer = LoraTrainer(
        base_model=model_name,
        rank=4,
        alpha=8,
        target_modules=TARGET_MODULES,
        device=test_device,
    )
    batch1 = _make_toy_batch(
        model_name, "First training batch for AdamW state.", batch_size=2, seq_len=8
    )
    batch2 = _make_toy_batch(
        model_name, "Second batch after checkpoint restore.", batch_size=2, seq_len=8
    )
    optim_cfg = {"lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 0.01}

    # Run 3 steps so AdamW exp_avg, exp_avg_sq, and step buffers are populated.
    for i in range(3):
        trainer.forward_backward(batch1, loss_fn="cross_entropy")
        trainer.optim_step(step_id=f"pre-{i}", optim_params=optim_cfg)

    ckpt_path = tmp_path / "checkpoint"
    trainer.save_state(ckpt_path)

    # Continue uninterrupted for step 4 + step 5.
    out_uninterrupted_1 = trainer.forward_backward(batch2, loss_fn="cross_entropy")
    step_uninterrupted = trainer.optim_step(step_id="post-3", optim_params=optim_cfg)
    out_uninterrupted_2 = trainer.forward_backward(batch2, loss_fn="cross_entropy")

    # Restore into a brand-new LoraTrainer instance and execute the exact same steps.
    resumed = LoraTrainer(
        base_model=model_name,
        rank=4,
        alpha=8,
        target_modules=TARGET_MODULES,
        device=test_device,
    )
    resumed.load_state(ckpt_path)

    out_resumed_1 = resumed.forward_backward(batch2, loss_fn="cross_entropy")
    step_resumed = resumed.optim_step(step_id="post-3", optim_params=optim_cfg)
    out_resumed_2 = resumed.forward_backward(batch2, loss_fn="cross_entropy")

    assert step_resumed == step_uninterrupted == {"step": 4}
    assert out_resumed_1["loss"] == out_uninterrupted_1["loss"]
    assert torch.equal(out_resumed_1["logprobs"].cpu(), out_uninterrupted_1["logprobs"].cpu())
    assert out_resumed_2["loss"] == out_uninterrupted_2["loss"]
    assert torch.equal(out_resumed_2["logprobs"].cpu(), out_uninterrupted_2["logprobs"].cpu())


def test_only_lora_params_require_grad(model_name: str, test_device: str) -> None:
    """6. Only LoRA params have requires_grad."""
    trainer = LoraTrainer(
        base_model=model_name,
        rank=4,
        alpha=8,
        target_modules=TARGET_MODULES,
        device=test_device,
    )
    trainable_names = [
        name for name, param in trainer.model.named_parameters() if param.requires_grad
    ]
    frozen_names = [
        name for name, param in trainer.model.named_parameters() if not param.requires_grad
    ]

    assert len(trainable_names) > 0
    assert len(frozen_names) > 0
    assert all("lora_" in name for name in trainable_names), (
        f"Found non-LoRA trainable parameters: {[n for n in trainable_names if 'lora_' not in n]}"
    )


def test_policy_gradient_advantage_loss(model_name: str, test_device: str) -> None:
    """Precomputed per-token advantages work in forward_backward and update weights."""
    torch.manual_seed(55)
    trainer = LoraTrainer(
        base_model=model_name,
        rank=4,
        alpha=8,
        target_modules=TARGET_MODULES,
        device=test_device,
    )
    batch = _make_toy_batch(
        model_name, "Policy gradient advantage token loss.", batch_size=2, seq_len=8
    )
    advantages = torch.full_like(batch["weights"], 0.5)
    advantages[1, :] = -0.25

    out = trainer.forward_backward(batch, loss_fn=advantages)
    weighted_adv_lp = (out["logprobs"].cpu() * advantages * batch["weights"]).sum()
    expected_loss = float((-weighted_adv_lp / batch["weights"].sum()).item())
    assert abs(out["loss"] - expected_loss) < 1e-5
    step_out = trainer.optim_step("pg-step-1", {"lr": 1e-3})
    assert step_out == {"step": 1}


def test_sample_completions_and_logprobs(model_name: str, test_device: str) -> None:
    """sample() returns completions, token_ids, and per-token logprobs."""
    torch.manual_seed(11)
    trainer = LoraTrainer(
        base_model=model_name,
        rank=4,
        alpha=8,
        target_modules=TARGET_MODULES,
        device=test_device,
    )
    result = trainer.sample(
        prompts=["Hello, my name is", "1 + 1 ="],
        params={"max_tokens": 5, "temperature": 0.0},
    )
    assert len(result["completions"]) == 2
    assert len(result["token_ids"]) == 2
    assert len(result["logprobs"]) == 2
    for text, toks, lps in zip(
        result["completions"], result["token_ids"], result["logprobs"], strict=True
    ):
        assert isinstance(text, str)
        assert 1 <= len(toks) <= 5
        assert len(toks) == len(lps)
        assert all(lp <= 0.0 for lp in lps)


def test_sample_seed_determinism(model_name: str, test_device: str) -> None:
    """SampleParams.seed makes stochastic sampling deterministic per call."""
    trainer = LoraTrainer(
        base_model=model_name,
        rank=4,
        alpha=8,
        target_modules=TARGET_MODULES,
        device=test_device,
    )
    prompt = ["Write a short story about a forest:"]
    run_a = trainer.sample(
        prompts=prompt,
        params={"max_tokens": 12, "temperature": 1.0, "seed": 42},
    )
    run_b = trainer.sample(
        prompts=prompt,
        params={"max_tokens": 12, "temperature": 1.0, "seed": 42},
    )
    run_c = trainer.sample(
        prompts=prompt,
        params={"max_tokens": 12, "temperature": 1.0, "seed": 43},
    )

    assert run_a["completions"] == run_b["completions"]
    assert run_a["token_ids"] == run_b["token_ids"]
    assert run_a["logprobs"] == run_b["logprobs"]
    assert run_a["token_ids"] != run_c["token_ids"]


def test_pydantic_input_validation_errors(model_name: str, test_device: str) -> None:
    """Invalid external inputs fail explicitly via Pydantic ValidationError or ValueError."""
    with pytest.raises(ValidationError):
        LoraTrainer(
            base_model=model_name,
            rank=0,
            alpha=8,
            target_modules=TARGET_MODULES,
            device=test_device,
        )
    with pytest.raises(ValidationError):
        LoraTrainer(
            base_model=model_name,
            rank=4,
            alpha=8,
            target_modules=[],
            device=test_device,
        )

    trainer = LoraTrainer(
        base_model=model_name,
        rank=4,
        alpha=8,
        target_modules=TARGET_MODULES,
        device=test_device,
    )
    batch = _make_toy_batch(model_name, "Validation check.", batch_size=1, seq_len=6)

    with pytest.raises(ValidationError):
        trainer.forward_backward(
            {
                "input_ids": batch["input_ids"],
                "target_ids": batch["target_ids"][:, :-1],
                "weights": batch["weights"],
            }
        )

    with pytest.raises(ValueError, match="Unsupported loss_fn"):
        trainer.forward_backward(batch, loss_fn="mse")

    with pytest.raises(ValidationError):
        trainer.forward_backward(batch, loss_fn=torch.ones((1, 2)))

    with pytest.raises(ValidationError):
        trainer.optim_step("", {"lr": 1e-3})

    with pytest.raises(ValidationError):
        trainer.optim_step("bad-lr", {"lr": -1e-3})

    with pytest.raises(ValidationError):
        trainer.optim_step("bad-eps-zero", {"lr": 1e-3, "eps": 0.0})

    with pytest.raises(ValidationError):
        trainer.optim_step("bad-eps-neg", {"lr": 1e-3, "eps": -1e-8})

    with pytest.raises(ValidationError):
        trainer.sample([], {"max_tokens": 4, "temperature": 0.0})


def test_float16_rejected_on_cpu(model_name: str) -> None:
    """Explicit float16 dtype on CPU raises a clear ValueError."""
    with pytest.raises(ValueError, match="float16 is not supported on CPU"):
        LoraTrainer(
            base_model=model_name,
            rank=4,
            alpha=8,
            target_modules=TARGET_MODULES,
            device="cpu",
            dtype="float16",
        )


def test_default_dtype_float32_on_cpu(model_name: str) -> None:
    """Default dtype on CPU is float32 for both base and LoRA weights."""
    trainer = LoraTrainer(
        base_model=model_name,
        rank=4,
        alpha=8,
        target_modules=TARGET_MODULES,
        device="cpu",
    )
    assert trainer.dtype_name == "float32"
    assert trainer.torch_dtype == torch.float32
    assert all(p.dtype == torch.float32 for p in trainer.model.parameters())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_bfloat16_training_and_save_load(model_name: str, tmp_path: Path) -> None:
    """On CUDA, default dtype is bf16 base with fp32 LoRA/AdamW and loss decreases."""
    torch.manual_seed(42)
    trainer = LoraTrainer(
        base_model=model_name,
        rank=4,
        alpha=8,
        target_modules=TARGET_MODULES,
        device="cuda",
    )
    if torch.cuda.is_bf16_supported():
        assert trainer.dtype_name == "bfloat16"
        base_dtypes = {p.dtype for p in trainer.model.parameters() if not p.requires_grad}
        assert base_dtypes == {torch.bfloat16}
    lora_dtypes = {p.dtype for p in trainer.model.parameters() if p.requires_grad}
    assert lora_dtypes == {torch.float32}

    batch = _make_toy_batch(
        model_name, "CUDA bfloat16 LoRA training check.", batch_size=2, seq_len=8
    )
    l0 = trainer.forward_backward(batch, loss_fn="cross_entropy")["loss"]
    trainer.optim_step("cuda-0", {"lr": 5e-4})
    ckpt = tmp_path / "cuda_ckpt"
    trainer.save_state(ckpt)
    l1 = trainer.forward_backward(batch, loss_fn="cross_entropy")["loss"]

    resumed = LoraTrainer(
        base_model=model_name,
        rank=4,
        alpha=8,
        target_modules=TARGET_MODULES,
        device="cuda",
    )
    resumed.load_state(ckpt)
    l1_resumed = resumed.forward_backward(batch, loss_fn="cross_entropy")["loss"]
    assert l1 < l0
    assert l1_resumed == l1


def test_forward_backward_return_logprobs_false(model_name: str, test_device: str) -> None:
    """forward_backward(..., return_logprobs=False) omits logprobs and matches loss."""
    trainer = LoraTrainer(
        base_model=model_name,
        rank=4,
        alpha=8,
        target_modules=TARGET_MODULES,
        device=test_device,
    )
    batch = _make_toy_batch(model_name, "Omit logprobs test.", batch_size=2, seq_len=8)
    with_lp = trainer.forward_backward(batch, loss_fn="cross_entropy", return_logprobs=True)
    without_lp = trainer.forward_backward(batch, loss_fn="cross_entropy", return_logprobs=False)

    assert "logprobs" in with_lp
    assert "logprobs" not in without_lp
    assert without_lp["loss"] == with_lp["loss"]


def test_sample_params_validation_top_p_and_stop() -> None:
    """SampleParams validates top_p bounds and non-empty stop strings (allowing whitespace)."""
    with pytest.raises(ValidationError):
        SampleParams(max_tokens=4, top_p=0.0)
    with pytest.raises(ValidationError):
        SampleParams(max_tokens=4, top_p=1.5)
    with pytest.raises(ValidationError):
        SampleParams(max_tokens=4, stop=[""])
    with pytest.raises(ValidationError):
        SampleParams(max_tokens=4, stop=["x"] * 9)

    valid_ws = SampleParams(max_tokens=4, top_p=0.9, stop=["\n"])
    assert valid_ws.stop == ["\n"]
    assert valid_ws.top_p == 0.9


def test_sample_top_p_and_stop_sequences(model_name: str, test_device: str) -> None:
    """sample() enforces stop strings (returning stop_reasons='stop') and top_p filtering."""
    trainer = LoraTrainer(
        base_model=model_name,
        rank=4,
        alpha=8,
        target_modules=TARGET_MODULES,
        device=test_device,
    )
    prompt = "Once upon a time in a small village"
    greedy = trainer.sample([prompt], {"max_tokens": 16, "temperature": 0.0})
    assert len(greedy["token_ids"][0]) == 16
    assert greedy["stop_reasons"] == ["length"]

    base_text = greedy["completions"][0]
    cut_start = max(1, len(base_text) // 3)
    stop_seq = base_text[cut_start : cut_start + 3]
    assert stop_seq in base_text

    stopped = trainer.sample(
        [prompt],
        {"max_tokens": 16, "temperature": 0.0, "stop": [stop_seq]},
    )
    assert stopped["stop_reasons"] == ["stop"]
    assert stop_seq not in stopped["completions"][0]
    assert stopped["completions"][0] == base_text[: base_text.index(stop_seq)]
    assert len(stopped["token_ids"][0]) < 16

    # Tiny top_p=1e-6 keeps only the highest-probability token, matching greedy decoding.
    num_prompt = "1, 2, 3, 4, 5, 6, 7,"
    greedy_num = trainer.sample([num_prompt], {"max_tokens": 6, "temperature": 0.0})
    nuc_num = trainer.sample(
        [num_prompt],
        {"max_tokens": 6, "temperature": 1.0, "top_p": 1e-6, "seed": 42},
    )
    assert nuc_num["token_ids"] == greedy_num["token_ids"]
    assert nuc_num["completions"] == greedy_num["completions"]


def test_export_adapter_loads_in_raw_peft(
    model_name: str, test_device: str, tmp_path: Path
) -> None:
    """export_adapter writes standard PEFT adapter files matching trainer logits."""
    trainer = LoraTrainer(
        base_model=model_name,
        rank=4,
        alpha=8,
        target_modules=TARGET_MODULES,
        device=test_device,
    )
    batch = _make_toy_batch(model_name, "Export adapter PEFT parity test.", batch_size=2, seq_len=8)
    for idx in range(3):
        trainer.forward_backward(batch, loss_fn="cross_entropy")
        trainer.optim_step(f"exp-{idx}", {"lr": 1e-3})

    export_dir = tmp_path / "exported_adapter"
    files = trainer.export_adapter(export_dir)
    assert "adapter_config.json" in files
    assert "adapter_model.safetensors" in files
    assert (export_dir / "adapter_config.json").is_file()
    assert (export_dir / "adapter_model.safetensors").is_file()
    assert not (export_dir / "trainer_state.pt").exists()

    fresh_base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(
        trainer.device
    )
    loaded_peft = PeftModel.from_pretrained(fresh_base, str(export_dir)).to(trainer.device)
    loaded_peft.eval()
    trainer.model.eval()
    dev_input_ids = batch["input_ids"].to(trainer.device)
    with torch.no_grad():
        trainer_logits = trainer.model(input_ids=dev_input_ids).logits.float()
        exported_logits = loaded_peft(input_ids=dev_input_ids).logits.float()
    assert torch.allclose(exported_logits, trainer_logits, atol=1e-5)
