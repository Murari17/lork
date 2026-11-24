"""Toy GRPO policy-gradient task using dado sample + advantage loss_fn."""

from __future__ import annotations

import json
import os
import sys

from transformers import AutoTokenizer

from dado.client import OptimParams, SampleParams, ServiceClient, TrainingClient

MODEL = "Qwen/Qwen2.5-0.5B"
BASE_URL = os.environ.get("DADO_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("DADO_API_KEY", "dev-secret-key")

PROMPT = "Reply with only one word (apple or banana):"
TARGET_WORD = "banana"
GROUP_SIZE = 8
NUM_STEPS = 12


def _reward(text: str) -> float:
    return 1.0 if TARGET_WORD in text.lower() else 0.0


def _sample_group(
    tc: TrainingClient,
    step_seed_offset: int,
) -> tuple[list[str], list[list[int]]]:
    futs = [
        tc.sample(
            [PROMPT],
            SampleParams(max_tokens=4, temperature=1.0, seed=step_seed_offset + idx),
        )
        for idx in range(GROUP_SIZE)
    ]
    completions: list[str] = []
    token_ids: list[list[int]] = []
    for fut in futs:
        res = fut.result()
        completions.append(res.completions[0])
        token_ids.append(res.token_ids[0])
    return completions, token_ids


def run_grpo(base_seed: int = 42) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids: list[int] = tokenizer(PROMPT, return_tensors="pt")["input_ids"][0].tolist()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    with (
        ServiceClient(BASE_URL, API_KEY) as client,
        client.create_training_client(
            MODEL, rank=8, alpha=16.0, target_modules=["q_proj", "v_proj"]
        ) as tc,
    ):
        skipped_steps = 0
        step_rewards: list[float] = []
        for step in range(NUM_STEPS):
            completions, sampled_token_ids = _sample_group(tc, base_seed + step * 1000)
            rewards = [_reward(c) for c in completions]
            mean_r = sum(rewards) / len(rewards)
            step_rewards.append(mean_r)
            advantages_scalar = [r - mean_r for r in rewards]

            if all(abs(a) < 1e-8 for a in advantages_scalar):
                skipped_steps += 1
                print(
                    f"step={step + 1:02d} mean_reward={mean_r:.3f} "
                    "(skipped: saturation / all-zero advantages)"
                )
                continue

            print(f"step={step + 1:02d} mean_reward={mean_r:.3f}")
            seqs = [prompt_ids + gen_ids for gen_ids in sampled_token_ids]
            max_len = max(len(s) for s in seqs) - 1

            input_ids: list[list[int]] = []
            target_ids: list[list[int]] = []
            weights: list[list[float]] = []
            advantages: list[list[float]] = []

            prompt_pred_len = len(prompt_ids) - 1
            for seq, gen_ids, adv in zip(seqs, sampled_token_ids, advantages_scalar, strict=True):
                inp = seq[:-1]
                tgt = seq[1:]
                w = [0.0] * prompt_pred_len + [1.0] * len(gen_ids)
                a = [0.0] * prompt_pred_len + [adv] * len(gen_ids)
                pad = max_len - len(inp)
                if pad > 0:
                    inp = inp + [pad_id] * pad
                    tgt = tgt + [pad_id] * pad
                    w = w + [0.0] * pad
                    a = a + [0.0] * pad
                input_ids.append(inp)
                target_ids.append(tgt)
                weights.append(w)
                advantages.append(a)

            batch = {
                "input_ids": input_ids,
                "target_ids": target_ids,
                "weights": weights,
            }
            tc.forward_backward(batch, loss_fn=advantages).result()
            tc.optim_step(OptimParams(lr=1e-3)).result()

        # Measure initial reward identically via step 1 (pre-update) and final reward post-step-12.
        initial_mean_reward = step_rewards[0]
        final_completions, _ = _sample_group(tc, base_seed + NUM_STEPS * 1000)
        final_rewards = [_reward(c) for c in final_completions]
        final_mean_reward = sum(final_rewards) / len(final_rewards)
        print(
            f"base_seed={base_seed} initial_mean_reward={initial_mean_reward:.3f} "
            f"final_mean_reward={final_mean_reward:.3f} skipped_steps={skipped_steps}"
        )

        summary: dict[str, object] = {
            "base_seed": base_seed,
            "initial_mean_reward": initial_mean_reward,
            "final_mean_reward": final_mean_reward,
            "skipped_steps": skipped_steps,
            "step_rewards": step_rewards,
        }
        print("GRPO_SUMMARY " + json.dumps(summary))
        return summary


def main() -> None:
    seed_str = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DADO_GRPO_SEED", "42")
    run_grpo(base_seed=int(seed_str))


if __name__ == "__main__":
    main()
