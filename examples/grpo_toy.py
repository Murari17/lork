"""Toy GRPO policy-gradient example using loraforge sample + advantage loss_fn."""

from __future__ import annotations

import json
import os
import random

from transformers import AutoTokenizer

from loraforge.client import OptimParams, SampleParams, ServiceClient

MODEL = "Qwen/Qwen2.5-0.5B"
BASE_URL = os.environ.get("LORAFORGE_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("LORAFORGE_API_KEY", "dev-secret-key")

PROMPT = "Reply with only one word (apple or banana):"
TARGET_WORD = "banana"
GROUP_SIZE = 8
NUM_STEPS = 12


def _reward(text: str) -> float:
    return 1.0 if TARGET_WORD in text.lower() else 0.0


def main() -> None:
    random.seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids: list[int] = tokenizer(PROMPT, return_tensors="pt")["input_ids"][0].tolist()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    with (
        ServiceClient(BASE_URL, API_KEY) as client,
        client.create_training_client(
            MODEL, rank=8, alpha=16.0, target_modules=["q_proj", "v_proj"]
        ) as tc,
    ):
        initial_eval = tc.sample(
            [PROMPT] * GROUP_SIZE,
            SampleParams(max_tokens=4, temperature=1.0),
        ).result()
        initial_rewards = [_reward(c) for c in initial_eval.completions]
        initial_mean_reward = sum(initial_rewards) / len(initial_rewards)
        print(f"initial_mean_reward={initial_mean_reward:.3f}")

        for step in range(NUM_STEPS):
            sampled = tc.sample(
                [PROMPT] * GROUP_SIZE,
                SampleParams(max_tokens=4, temperature=1.0),
            ).result()
            rewards = [_reward(c) for c in sampled.completions]
            mean_r = sum(rewards) / len(rewards)
            advantages_scalar = [r - mean_r for r in rewards]
            print(f"step={step + 1:02d} mean_reward={mean_r:.3f}")

            if all(abs(a) < 1e-8 for a in advantages_scalar):
                continue

            seqs = [prompt_ids + gen_ids for gen_ids in sampled.token_ids]
            max_len = max(len(s) for s in seqs) - 1

            input_ids: list[list[int]] = []
            target_ids: list[list[int]] = []
            weights: list[list[float]] = []
            advantages: list[list[float]] = []

            prompt_pred_len = len(prompt_ids) - 1
            for seq, gen_ids, adv in zip(seqs, sampled.token_ids, advantages_scalar, strict=True):
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

        final_eval = tc.sample(
            [PROMPT] * GROUP_SIZE,
            SampleParams(max_tokens=4, temperature=1.0),
        ).result()
        final_rewards = [_reward(c) for c in final_eval.completions]
        final_mean_reward = sum(final_rewards) / len(final_rewards)
        print(f"final_mean_reward={final_mean_reward:.3f}")
        print(
            "GRPO_SUMMARY "
            + json.dumps(
                {
                    "initial_mean_reward": initial_mean_reward,
                    "final_mean_reward": final_mean_reward,
                }
            )
        )


if __name__ == "__main__":
    main()
