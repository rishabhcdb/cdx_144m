"""
logger.py — CSV logger with optional wandb backend.

Interface:
    logger = Logger(config)
    logger.log_train(step, loss, grad_norm, lr)
    logger.log_val(step, val_loss, val_ppl)
    logger.close()

Both backends implement the same interface.  Backend is chosen by
config.log_backend: "csv" (default) or "wandb".

CSV layout (logs/train_log.csv):
    step, train_loss, grad_norm, lr, val_loss, val_ppl
Empty val columns are written as empty strings for non-eval steps.
"""

import csv
import math
import os
from typing import Optional

from config import TrainConfig


class CSVLogger:
    def __init__(self, config: TrainConfig):
        os.makedirs(config.log_dir, exist_ok=True)
        self._path = os.path.join(config.log_dir, "train_log.csv")
        self._file = open(self._path, "a", newline="")
        self._writer = csv.writer(self._file)
        # Write header only if file is new (position 0 after open in append mode)
        if self._file.tell() == 0:
            self._writer.writerow(["step", "train_loss", "grad_norm", "lr", "val_loss", "val_ppl"])

    def log_train(self, step: int, loss: float, grad_norm: float, lr: float):
        self._writer.writerow([step, f"{loss:.6f}", f"{grad_norm:.4f}", f"{lr:.6e}", "", ""])
        self._file.flush()

    def log_val(self, step: int, val_loss: float, val_ppl: float):
        self._writer.writerow([step, "", "", "", f"{val_loss:.6f}", f"{val_ppl:.4f}"])
        self._file.flush()

    def close(self):
        self._file.close()


class WandbLogger:
    def __init__(self, config: TrainConfig):
        import wandb
        wandb.init(
            project="llm144m",
            config=vars(config),
        )
        self._wandb = wandb

    def log_train(self, step: int, loss: float, grad_norm: float, lr: float):
        self._wandb.log({"train/loss": loss, "train/grad_norm": grad_norm, "train/lr": lr}, step=step)

    def log_val(self, step: int, val_loss: float, val_ppl: float):
        self._wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=step)

    def close(self):
        self._wandb.finish()


def Logger(config: TrainConfig):
    """Factory — returns the appropriate logger for config.log_backend."""
    if config.log_backend == "wandb":
        return WandbLogger(config)
    return CSVLogger(config)
