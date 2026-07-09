"""
lr_schedule.py — Cosine LR schedule with linear warmup.

Matches the plan exactly:
  - Linear warmup for `warmup_steps` steps (step 0 → peak_lr at warmup_steps)
  - Cosine decay from peak_lr → min_lr over the remaining steps
  - LR never drops below min_lr

Usage (called every step in the training loop):
    lr = get_lr(step, config)
    for group in optimizer.param_groups:
        group["lr"] = lr
"""

import math

from config import TrainConfig


def get_lr(step: int, config: TrainConfig) -> float:
    """
    Returns the learning rate scalar for the given training step.

    Args:
        step:   current global step (0-indexed)
        config: TrainConfig with peak_lr, min_lr, warmup_steps, total_steps
    """
    # 1. Warmup: linear ramp from 0 to peak_lr
    if step < config.warmup_steps:
        return config.peak_lr * (step + 1) / config.warmup_steps

    # 2. Past the decay horizon: clamp to min_lr
    if step >= config.total_steps:
        return config.min_lr

    # 3. Cosine decay between warmup_steps and total_steps
    decay_steps   = config.total_steps - config.warmup_steps
    steps_done    = step - config.warmup_steps
    cosine_ratio  = 0.5 * (1.0 + math.cos(math.pi * steps_done / decay_steps))
    return config.min_lr + cosine_ratio * (config.peak_lr - config.min_lr)
