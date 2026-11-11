"""Run the complete prescribed holdout schedule for predetermined seeds and summarize OOF predictions."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from config import ROOT_DIR
from summarize_validation_seeds import validate_seeds
from train_all_folds import parse_args as parse_training_args
from validation_audit import build_validation_audit, expected_validation_roster, read_csv


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        epilog="Additional options such as --loss, --epochs and --augmentation are forwarded to train_all_folds.py; see that script's --help.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs/development_evaluation")
    parser.add_argument("--assignments", type=Path)
    parser.add_argument("--aggregate-only", action="store_true", help="Reuse saved predictions without training")
    parser.add_argument("--dry-run", action="store_true", help="Save the real cohort audit and execution plan without training")
    parser.add_argument("--resume", action="store_true")
    args, training_argv = parser.parse_known_args()
    reserved = {"--seed", "--checkpoint-dir", "--summary-path", "--overwrite", "--continue-on-error"}
    if any(token.split("=", 1)[0] in reserved for token in training_argv):
        parser.error("Use --seeds and --output-dir; per-seed paths are managed automatically. Overwriting or skipping failed runs is not supported.")
    try:
        validate_seeds(args.seeds)
    except ValueError as error:
        parser.error(str(error))
    training = parse_training_args(training_argv)
    if training.fold_start != 1 or training.fold_end != 25:
        parser.error("Repeated evaluation requires all 25 folds; use train_all_folds.py for a pilot subset")
    manifest = training.manifest.resolve()
    assignments = (args.assignments or manifest.with_name("fold_assignments.csv")).resolve()
    if not manifest.is_file() or not assignments.is_file():
        parser.error("Both the fold manifest and workbook assignment table are required")
    rows, assigned_rows = read_csv(manifest), read_csv(assignments)
    try:
        expected = expected_validation_roster(rows, training.label_source, assigned_rows)
        audit = build_validation_audit(rows, assigned_rows, training.label_source)
    except ValueError as error:
        parser.error(str(error))
    if not expected or any(row["pair_status"] in {"invalid_assignment", "unverified_assignments"} for row in audit):
        parser.error("Expected two assigned participants in every workbook development fold and a nonempty available cohort")
    excluded = {"seed", "checkpoint_dir", "summary_path", "dry_run", "resume", "overwrite", "continue_on_error"}
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(training).items() if key not in excluded}
    config["manifest"] = str(manifest)
    config["root"] = str(training.root.resolve())
    plan = {
        "seeds": args.seeds,
        "training_settings": config,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "assignments_sha256": hashlib.sha256(assignments.read_bytes()).hexdigest(),
        "expected_images_per_seed": len(expected),
        "expected_subjects_per_seed": len({row['participant_id'] for row in expected.values()}),
        "pair_status_counts": dict(Counter(row["pair_status"] for row in audit)),
        "test_partition_loaded_or_evaluated": False,
        "selection_policy": "fixed_epoch_budget_final_checkpoint_no_best_seed_selection",
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "evaluation_plan.json"
    if plan_path.is_file() and json.loads(plan_path.read_text()) != plan:
        parser.error("This output folder has a different evaluation plan. Use a new folder for changed settings, seeds or data.")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    (output / "pair_coverage.json").write_text(json.dumps(audit, indent=2) + "\n")
    commands = []
    summaries = []
    for seed in args.seeds:
        folder = output / f"seed_{seed}"
        if not args.aggregate_only:
            command = [sys.executable, str(ROOT_DIR / "train_all_folds.py"), *training_argv,
                       "--manifest", str(manifest), "--root", str(training.root.resolve()),
                       "--seed", str(seed), "--checkpoint-dir", str(folder / "checkpoints"),
                       "--summary-path", str(folder / "training_runs.json")]
            if args.resume:
                command.append("--resume")
            commands.append(command)
        commands.append([
            sys.executable, str(ROOT_DIR / "aggregate_validation.py"),
            "--manifest", str(manifest), "--assignments", str(assignments),
            "--predictions-dir", str(folder / "validation_predictions"),
            "--output-dir", str(folder / "reports"), "--loss", training.loss,
            "--label-source", training.label_source, "--freeze-policy", training.freeze_policy,
            "--focal-alpha-mode", training.focal_alpha_mode, "--seed", str(seed), "--require-complete",
        ])
        summaries.append(folder / "reports/oof_validation_summary.json")
    commands.append([
        sys.executable, str(ROOT_DIR / "summarize_validation_seeds.py"),
        "--summaries", *map(str, summaries), "--seeds", *map(str, args.seeds),
        "--output-dir", str(output / "combined"),
    ])
    execution = {"status": "planned" if args.dry_run else "running", "commands": commands}
    run_path = output / "evaluation_run.json"
    run_path.write_text(json.dumps(execution, indent=2) + "\n")
    print(f"Cohort: {plan['expected_images_per_seed']} images / {plan['expected_subjects_per_seed']} subjects per seed.")
    print(f"Pairs: {plan['pair_status_counts']}; seeds: {args.seeds}.")
    if args.dry_run:
        print(f"Plan saved: {plan_path}. No training or metric calculation was performed.")
        return
    try:
        for index, command in enumerate(commands):
            print(f"Evaluation step {index + 1}/{len(commands)}: {Path(command[1]).name}", flush=True)
            run_command(command)
    except (subprocess.CalledProcessError, OSError) as error:
        execution.update(status="failed", failed_command=command, error=str(error))
        run_path.write_text(json.dumps(execution, indent=2) + "\n")
        raise
    execution["status"] = "complete"
    run_path.write_text(json.dumps(execution, indent=2) + "\n")
    print(f"Evaluation complete: {output / 'combined/Report.md'}")


if __name__ == "__main__":
    main()
