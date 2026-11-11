"""Display streamed training progress in Jupyter, VS Code and Colab."""
from __future__ import annotations

from dataclasses import dataclass, field
import html
import json
import os
import signal
import subprocess
import time

from utils.progress import PREFIX


@dataclass
class ProgressState:
    title: str
    fold_start: int
    fold_end: int
    epochs: int
    fold: int = 0
    completed_folds: int = 0
    epoch: int = 0
    completed_epochs: int = 0
    batches: int = 0
    total_batches: int = 1
    phase: str = "Starting"
    loss: float | None = None
    started: float = field(default_factory=time.monotonic)
    training_started: float | None = None
    new_epochs: int = 0

    def update(self, event: dict) -> None:
        kind = event["event"]
        self.fold = event.get("fold", self.fold)
        if kind == "fold_start":
            self.epoch = self.completed_epochs = self.batches = 0
            self.loss = None
            self.phase = "Preparing / resuming fold"
        elif kind == "epoch_start":
            self.epoch = event["epoch"]
            self.completed_epochs = self.epoch - 1
            self.batches = 0
            self.training_started = self.training_started or time.monotonic()
        elif kind == "batch":
            self.phase = event["phase"]
            self.batches, self.total_batches = event["completed"], event["total"]
            self.loss = event["loss"]
        elif kind == "saving":
            self.phase = "Saving checkpoint"
        elif kind == "epoch_end":
            self.completed_epochs = event["epoch"]
            self.new_epochs += 1
            self.loss = event["train_loss"]
            self.phase = "Epoch completed"
        elif kind == "fold_end":
            if event["return_code"] == 0:
                self.completed_folds += 1
                self.completed_epochs = self.epochs
                self.phase = "Fold completed"
            else:
                self.phase = "Training stopped — see error output"

    @staticmethod
    def duration(seconds: float) -> str:
        minutes, seconds = divmod(max(0, int(seconds)), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def render(self) -> str:
        def bar(label, value, total):
            return (f'<div style="margin-top:8px">{html.escape(label)}: {value}/{total}</div>'
                    f'<progress value="{value}" max="{max(1, total)}" '
                    'style="width:100%;height:16px;accent-color:#2563eb"></progress>')
        elapsed = self.duration(time.monotonic() - self.started)
        eta = "estimating…"
        if self.new_epochs and self.training_started is not None:
            remaining = max(0, (self.fold_end - self.fold) * self.epochs
                            + self.epochs - self.completed_epochs)
            eta = self.duration((time.monotonic() - self.training_started)
                                / self.new_epochs * remaining)
        loss = f" · loss {self.loss:.4f}" if self.loss is not None else ""
        return (f'<div style="max-width:760px;padding:16px;border:1px solid #888;'
                f'border-radius:10px"><b>{html.escape(self.title)}</b> · fold {self.fold}'
                f'<div>{html.escape(self.phase)}{loss}</div>'
                + bar("Folds completed", self.completed_folds, self.fold_end - self.fold_start + 1)
                + bar("Epochs completed in this fold", self.completed_epochs, self.epochs)
                + bar("Batches in current phase", self.batches, self.total_batches)
                + f'<div style="margin-top:8px">Elapsed {elapsed} · estimated remaining {eta}</div></div>')


def run_with_progress(command: list[str], *, title: str, fold_start: int,
                      fold_end: int, epochs: int) -> None:
    from IPython.display import HTML, display
    state = ProgressState(title, fold_start, fold_end, epochs)
    view = display(HTML(state.render()), display_id=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, start_new_session=True)
    refreshed = 0.0
    try:
        for line in process.stdout:
            if line.startswith(PREFIX):
                event = json.loads(line[len(PREFIX):])
                state.update(event)
                now = time.monotonic()
                if event["event"] != "batch" or now - refreshed >= 0.5:
                    view.update(HTML(state.render()))
                    refreshed = now
            else:
                print(line, end="", flush=True)
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
        state.phase = "Stopped — resume from the last completed checkpoint"
        raise
    finally:
        process.stdout.close()
        view.update(HTML(state.render()))
