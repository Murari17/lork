"""Core LoRA fine-tuning execution engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from peft import (
    LoraConfig,
    PeftMixedModel,
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from lork.schemas import (
    ForwardBackwardBatch,
    OptimParams,
    OptimStepRequest,
    SampleParams,
    SampleRequest,
    StatePathInput,
    TrainerInitConfig,
    validate_loss_fn,
)


def _ensure_device_available(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device_str!r}, but CUDA is not available")
    if device_str == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested device 'mps', but MPS is not available")
    return torch.device(device_str)


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _resolve_dtype(
    device_str: str,
    dtype: Literal["float32", "bfloat16", "float16"] | None,
) -> tuple[Literal["float32", "bfloat16", "float16"], torch.dtype]:
    if dtype is None:
        if (
            device_str.startswith("cuda")
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):
            return "bfloat16", torch.bfloat16
        return "float32", torch.float32
    if dtype == "float16" and device_str == "cpu":
        raise ValueError("float16 is not supported on CPU; use float32 or bfloat16")
    return dtype, _DTYPE_MAP[dtype]


def _move_optimizer_state_to_cpu(state_dict: dict[str, Any]) -> dict[str, Any]:
    copied_state: dict[Any, dict[str, Any]] = {}
    for key, entry in state_dict["state"].items():
        copied_entry: dict[str, Any] = {}
        for field_name, field_val in entry.items():
            if isinstance(field_val, torch.Tensor):
                copied_entry[field_name] = field_val.detach().cpu().clone()
            else:
                copied_entry[field_name] = field_val
        copied_state[key] = copied_entry
    return {
        "state": copied_state,
        "param_groups": [dict(group) for group in state_dict["param_groups"]],
    }


class LoraTrainer:
    """Low-level LoRA training primitive owning model, optimizer, and step state."""

    def __init__(
        self,
        base_model: str,
        rank: int,
        alpha: float,
        target_modules: list[str],
        device: str = "cpu",
        dtype: Literal["float32", "bfloat16", "float16"] | None = None,
    ) -> None:
        cfg = TrainerInitConfig(
            base_model=base_model,
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
            device=device,
            dtype=dtype,
        )
        self.device: torch.device = _ensure_device_available(cfg.device)
        self.dtype_name, self.torch_dtype = _resolve_dtype(cfg.device, cfg.dtype)
        self.config: TrainerInitConfig = cfg

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(cfg.base_model)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=self.torch_dtype)
        for param in base.parameters():
            param.requires_grad_(False)

        lora_alpha_val: int = (
            int(cfg.alpha) if cfg.alpha.is_integer() else cast(int, cfg.alpha)
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.rank,
            lora_alpha=lora_alpha_val,
            target_modules=cfg.target_modules,
            lora_dropout=0.0,
            bias="none",
        )
        peft_model: PeftModel | PeftMixedModel = get_peft_model(
            base, lora_config, autocast_adapter_dtype=True
        )
        self.model: PeftModel | PeftMixedModel = peft_model.to(self.device)
        self.model.train()
        for param in self.model.parameters():
            if param.requires_grad and param.dtype != torch.float32:
                param.data = param.data.to(dtype=torch.float32)

        trainable_names_and_params = [
            (name, param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        ]
        if not trainable_names_and_params:
            raise RuntimeError(
                f"No trainable LoRA parameters created for target_modules={cfg.target_modules}"
            )
        non_lora = [name for name, _ in trainable_names_and_params if "lora_" not in name]
        if non_lora:
            raise RuntimeError(f"Non-LoRA parameters unexpectedly require grad: {non_lora}")

        self._trainable_params: list[torch.nn.Parameter] = [
            param for _, param in trainable_names_and_params
        ]
        self.optimizer: torch.optim.AdamW = torch.optim.AdamW(
            self._trainable_params,
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            eps=1e-8,
        )

        self._step_count: int = 0
        self._applied_steps: dict[str, int] = {}
        self._accum_weight_sum: float = 0.0

    @property
    def applied_steps(self) -> dict[str, int]:
        """Return copy of applied step_id -> step ledger."""
        return dict(self._applied_steps)

    def forward_backward(
        self,
        batch: dict[str, Any] | ForwardBackwardBatch,
        loss_fn: str | torch.Tensor | list[list[float]] = "cross_entropy",
        return_logprobs: bool = True,
    ) -> dict[str, Any]:
        """Run forward and backward pass, accumulating gradients without stepping weights."""
        validated_batch = (
            batch
            if isinstance(batch, ForwardBackwardBatch)
            else ForwardBackwardBatch.model_validate(batch)
        )
        batch_shape = (
            int(validated_batch.input_ids.shape[0]),
            int(validated_batch.input_ids.shape[1]),
        )
        validated_loss = validate_loss_fn(loss_fn, expected_shape=batch_shape)

        input_ids = validated_batch.input_ids.to(self.device)
        target_ids = validated_batch.target_ids.to(self.device)
        weights = validated_batch.weights.to(self.device)

        self.model.train()
        outputs = self.model(input_ids=input_ids)
        logits: torch.Tensor = outputs.logits.float()
        log_probs_all = F.log_softmax(logits, dim=-1)
        token_logprobs = log_probs_all.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)

        if isinstance(validated_loss, str):
            per_token_loss = -token_logprobs * weights
        else:
            advantages = validated_loss.to(self.device)
            per_token_loss = -token_logprobs * advantages * weights

        loss_sum = per_token_loss.sum()
        weight_sum = weights.sum()
        torch.autograd.backward(loss_sum)

        self._accum_weight_sum += float(weight_sum.item())
        batch_mean_loss = float((loss_sum.detach() / weight_sum.detach()).item())

        out: dict[str, Any] = {"loss": batch_mean_loss}
        if return_logprobs:
            out["logprobs"] = token_logprobs.detach()
        return out

    def optim_step(
        self,
        step_id: str,
        optim_params: dict[str, Any] | OptimParams,
    ) -> dict[str, int]:
        """Normalize accumulated gradients, apply AdamW step, and zero grads (idempotent)."""
        req = OptimStepRequest.model_validate(
            {"step_id": step_id, "optim_params": optim_params}
        )
        if req.step_id in self._applied_steps:
            return {"step": self._applied_steps[req.step_id]}

        if self._accum_weight_sum <= 0.0:
            raise RuntimeError(
                "Cannot run optim_step without accumulated gradients from forward_backward"
            )

        inv_weight = 1.0 / self._accum_weight_sum
        for param in self._trainable_params:
            if param.grad is not None:
                param.grad.mul_(inv_weight)

        for group in self.optimizer.param_groups:
            group["lr"] = req.optim_params.lr
            group["betas"] = req.optim_params.betas
            group["weight_decay"] = req.optim_params.weight_decay
            group["eps"] = req.optim_params.eps

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._accum_weight_sum = 0.0
        self._step_count += 1
        self._applied_steps[req.step_id] = self._step_count

        return {"step": self._step_count}

    def sample(
        self,
        prompts: list[str],
        params: dict[str, Any] | SampleParams,
    ) -> dict[str, Any]:
        """Autoregressively sample completions and per-token log-probabilities."""
        req = SampleRequest.model_validate({"prompts": prompts, "params": params})

        completions: list[str] = []
        all_token_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        stop_reasons: list[Literal["length", "eos", "stop"]] = []

        was_training = self.model.training
        self.model.eval()
        generator: torch.Generator | None = None
        if req.params.seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(req.params.seed)
        try:
            with torch.no_grad():
                for prompt in req.prompts:
                    encoded = self.tokenizer(prompt, return_tensors="pt")
                    cur_input: torch.Tensor = encoded["input_ids"].to(self.device)
                    if cur_input.shape[1] == 0:
                        raise ValueError("Prompt tokenized to 0 tokens")
                    past_key_values: Any = None
                    gen_ids: list[int] = []
                    gen_lps: list[float] = []
                    stop_reason: Literal["length", "eos", "stop"] = "length"
                    decoded_override: str | None = None

                    for _ in range(req.params.max_tokens):
                        outputs = self.model(
                            input_ids=cur_input,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                        past_key_values = outputs.past_key_values
                        next_logits: torch.Tensor = outputs.logits[:, -1, :].float()
                        log_probs = F.log_softmax(next_logits, dim=-1)

                        if req.params.temperature == 0.0:
                            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                        else:
                            probs = torch.softmax(
                                next_logits / req.params.temperature,
                                dim=-1,
                            )
                            if req.params.top_p < 1.0:
                                sorted_probs, sorted_indices = torch.sort(
                                    probs, descending=True, dim=-1
                                )
                                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                                remove_mask = (cumulative_probs - sorted_probs) > req.params.top_p
                                remove_mask[:, 0] = False
                                filtered_probs = sorted_probs.masked_fill(remove_mask, 0.0)
                                filtered_probs = filtered_probs / filtered_probs.sum(
                                    dim=-1, keepdim=True
                                )
                                sampled_idx = torch.multinomial(
                                    filtered_probs, num_samples=1, generator=generator
                                )
                                next_token = torch.gather(sorted_indices, -1, sampled_idx)
                            else:
                                next_token = torch.multinomial(
                                    probs, num_samples=1, generator=generator
                                )

                        tok_id = int(next_token[0, 0].item())
                        tok_lp = float(log_probs[0, tok_id].item())
                        gen_ids.append(tok_id)
                        gen_lps.append(tok_lp)
                        cur_input = next_token

                        if (
                            self.tokenizer.eos_token_id is not None
                            and tok_id == self.tokenizer.eos_token_id
                        ):
                            stop_reason = "eos"
                            break

                        if req.params.stop:
                            partial = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                            if not isinstance(partial, str):
                                raise TypeError(
                                    f"Expected decoded str, got {type(partial).__name__}"
                                )
                            earliest_idx: int | None = None
                            for stop_seq in req.params.stop:
                                found_at = partial.find(stop_seq)
                                if found_at != -1 and (
                                    earliest_idx is None or found_at < earliest_idx
                                ):
                                    earliest_idx = found_at
                            if earliest_idx is not None:
                                decoded_override = partial[:earliest_idx]
                                stop_reason = "stop"
                                break

                    if decoded_override is not None:
                        decoded = decoded_override
                    else:
                        raw_decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                        if not isinstance(raw_decoded, str):
                            raise TypeError(
                                f"Expected decoded str, got {type(raw_decoded).__name__}"
                            )
                        decoded = raw_decoded
                    completions.append(decoded)
                    all_token_ids.append(gen_ids)
                    all_logprobs.append(gen_lps)
                    stop_reasons.append(stop_reason)
        finally:
            if was_training:
                self.model.train()

        return {
            "completions": completions,
            "token_ids": all_token_ids,
            "logprobs": all_logprobs,
            "stop_reasons": stop_reasons,
        }

    def export_adapter(self, path: str | Path) -> list[str]:
        """Export standalone PEFT LoRA adapter files (for vLLM / PeftModel.from_pretrained)."""
        validated = StatePathInput.model_validate({"path": path})
        export_dir = validated.path
        export_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(export_dir), safe_serialization=True)
        return sorted(p.name for p in export_dir.iterdir() if p.is_file())

    def save_state(self, path: str | Path) -> None:
        """Persist LoRA weights, AdamW optimizer state, step ledger, and pending grads."""
        validated = StatePathInput.model_validate({"path": path})
        ckpt_dir = validated.path
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        adapter_sd = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in get_peft_model_state_dict(self.model).items()
        }
        save_safetensors(adapter_sd, str(ckpt_dir / "adapter_weights.safetensors"))

        pending_grads: list[torch.Tensor | None] = [
            param.grad.detach().cpu().clone() if param.grad is not None else None
            for param in self._trainable_params
        ]
        state_bundle = {
            "optimizer_state_dict": _move_optimizer_state_to_cpu(
                self.optimizer.state_dict()
            ),
            "step_count": self._step_count,
            "applied_steps": dict(self._applied_steps),
            "accum_weight_sum": self._accum_weight_sum,
            "pending_grads": pending_grads,
            "torch_rng_state": torch.get_rng_state(),
        }
        torch.save(state_bundle, ckpt_dir / "trainer_state.pt")

    def load_state(self, path: str | Path) -> None:
        """Restore LoRA weights, AdamW optimizer state, step ledger, and pending grads."""
        validated = StatePathInput.model_validate({"path": path})
        ckpt_dir = validated.path
        adapter_file = ckpt_dir / "adapter_weights.safetensors"
        state_file = ckpt_dir / "trainer_state.pt"
        if not adapter_file.is_file() or not state_file.is_file():
            raise FileNotFoundError(
                f"Incomplete checkpoint at {ckpt_dir}: expected "
                f"{adapter_file.name} and {state_file.name}"
            )

        adapter_sd = {
            name: tensor.to(dtype=torch.float32)
            for name, tensor in load_safetensors(
                str(adapter_file), device=str(self.device)
            ).items()
        }
        set_peft_model_state_dict(self.model, adapter_sd)

        state_bundle: dict[str, Any] = torch.load(
            state_file,
            map_location="cpu",
            weights_only=True,
        )
        self.optimizer.load_state_dict(state_bundle["optimizer_state_dict"])

        self._step_count = int(state_bundle["step_count"])
        self._applied_steps = {
            str(k): int(v) for k, v in state_bundle["applied_steps"].items()
        }
        self._accum_weight_sum = float(state_bundle["accum_weight_sum"])

        pending_grads: list[torch.Tensor | None] = state_bundle["pending_grads"]
        if len(pending_grads) != len(self._trainable_params):
            raise RuntimeError(
                "Loaded pending_grads count does not match trainable parameters"
            )
        for param, grad_tensor in zip(
            self._trainable_params, pending_grads, strict=True
        ):
            param.grad = None if grad_tensor is None else grad_tensor.to(self.device)

        rng_state: torch.Tensor = state_bundle["torch_rng_state"]
        torch.set_rng_state(rng_state)
