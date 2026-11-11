"""Terminal progress and machine-readable updates for notebook displays."""
from __future__ import annotations

import json
from tqdm import tqdm

PREFIX = "@@TRAIN_PROGRESS@@"


def emit_progress(event: str, **values) -> None:
    print(PREFIX + json.dumps({"event": event, **values}), flush=True)


class TrainingProgress:
    def __init__(self, *, fold: int, epochs: int, start_epoch: int,
                 json_output: bool = False, enabled: bool = True):
        self.fold, self.epochs, self.epoch = fold, epochs, start_epoch
        self.json_output, self.enabled = json_output, enabled
        self.batches = None
        self.epochs_bar = (tqdm(total=epochs, initial=start_epoch - 1,
                                desc=f"Fold {fold}/25 epochs", unit="epoch", position=0)
                           if enabled and not json_output else None)

    def event(self, event: str, **values) -> None:
        if self.enabled and self.json_output:
            emit_progress(event, fold=self.fold, epoch=self.epoch,
                          epochs=self.epochs, **values)

    def start(self, epoch: int) -> None:
        self.epoch = epoch
        self.event("epoch_start")

    def batch(self, phase: str, completed: int, total: int, loss: float) -> None:
        self.event("batch", phase=phase, completed=completed, total=total, loss=loss)
        if self.epochs_bar is not None:
            if completed == 1:
                if self.batches is not None:
                    self.batches.close()
                self.batches = tqdm(total=total, desc=f"{phase} epoch {self.epoch}",
                                    unit="batch", position=1, leave=False)
            self.batches.set_postfix(loss=f"{loss:.4f}", refresh=False)
            self.batches.update(completed - self.batches.n)

    def saving(self) -> None:
        self.event("saving")
        if self.epochs_bar is not None:
            self.epochs_bar.set_postfix(status="saving checkpoint")

    def complete(self, train_loss: float, val_loss: float | None) -> None:
        self.event("epoch_end", train_loss=train_loss, val_loss=val_loss)
        if self.batches is not None:
            self.batches.close()
            self.batches = None
        if self.epochs_bar is not None:
            self.epochs_bar.set_postfix(train_loss=f"{train_loss:.4f}")
            self.epochs_bar.update(1)

    def close(self) -> None:
        if self.batches is not None:
            self.batches.close()
        if self.epochs_bar is not None:
            self.epochs_bar.close()
