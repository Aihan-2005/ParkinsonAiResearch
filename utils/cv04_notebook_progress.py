"""Live fold, stage-epoch and batch progress for the compact CV04 notebook."""
from dataclasses import dataclass
import html
import time

from utils.notebook_training import ProgressState, run_with_progress


@dataclass
class CV04ProgressState(ProgressState):
    stage: str = "Preparing"

    def update(self, event):
        kind = event["event"]
        if kind == "cv04_fold_start":
            self.fold = event["fold"]
            self.completed_epochs = self.batches = 0
            self.loss = None
            self.phase = "Preparing fold"
        elif kind == "cv04_stage":
            self.stage = event["stage"]
            self.epochs = event["epochs"]
            self.completed_epochs = event.get("completed_epochs", 0)
            self.batches, self.total_batches = 0, 1
            self.loss = None
            self.phase = self.stage
        elif kind == "cv04_phase":
            self.phase = event["phase"]
        elif kind == "cv04_features":
            self.phase = f"Extracting {event['split']} features"
            self.batches, self.total_batches = event["completed"], event["total"]
        elif kind == "cv04_stage_done":
            self.completed_epochs = self.epochs
            self.phase = f"{self.stage} complete"
        elif kind == "cv04_fold_end":
            self.completed_folds += 1
            self.phase = "Fold completed"
        elif kind == "cv04_complete":
            self.phase = "Training complete — ready for KNN selection"
        else:
            super().update(event)

    def render(self):
        def bar(label, value, total):
            percent = 100 * value / max(1, total)
            return (f'<div style="margin-top:10px">{html.escape(label)}: '
                    f'{value}/{total} · {percent:.0f}%</div>'
                    f'<progress value="{value}" max="{max(1, total)}" '
                    'style="width:100%;height:18px;accent-color:#2563eb"></progress>')
        loss = f" · loss {self.loss:.5f}" if self.loss is not None else ""
        return ('<div style="max-width:760px;padding:16px;border:1px solid #888;border-radius:10px">'
                f'<b>CV04 · fold {self.fold} · {html.escape(self.stage)}</b>'
                f'<div>{html.escape(self.phase)}{loss}</div>'
                + bar("Folds finished", self.completed_folds, self.fold_end - self.fold_start + 1)
                + bar(f"{self.stage} epochs", self.completed_epochs, self.epochs)
                + bar("Current training / validation / feature batches", self.batches, self.total_batches)
                + f'<div style="margin-top:10px">Elapsed {self.duration(time.monotonic() - self.started)}</div></div>')


def run_cv04_with_progress(command, *, fold_start, fold_end, baseline_epochs, log_path=None):
    state = CV04ProgressState("CV04", fold_start, fold_end, baseline_epochs)
    run_with_progress(command, title="CV04", fold_start=fold_start, fold_end=fold_end,
                      epochs=baseline_epochs, progress_state=state, log_path=log_path)
