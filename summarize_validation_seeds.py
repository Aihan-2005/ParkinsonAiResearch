"""Summarize complete available-data OOF evaluations across predetermined seeds."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from utils.validation_evaluation import comparable_training_config, roster_sha256, score_predictions
from validation_audit import read_csv


def validate_seeds(seeds: list[int]) -> None:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Choose a nonempty list of distinct seeds before running evaluation")
    if any(type(seed) is not int or not 0 <= seed < 2 ** 63 for seed in seeds):
        raise ValueError("Seeds must be integers in [0, 2**63)")


def summarize_seed_reports(paths: list[Path], seeds: list[int], output_dir: Path) -> dict:
    validate_seeds(seeds)
    reports = {}
    reference = None
    per_seed = []
    for path in paths:
        path = Path(path).resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        seed = report["seed"]
        if seed not in seeds or seed in reports:
            raise ValueError(f"Unexpected or duplicate seed report: {seed}")
        if report.get("schema_version") != 2:
            raise ValueError("Re-run aggregate_validation.py to generate the current summary format")
        if report.get("status") != "complete" or report.get("prediction_coverage_complete") is not True:
            raise ValueError(f"Seed {seed} has incomplete prediction coverage; do not summarize a partial run")
        if report.get("test_partition_loaded_or_evaluated") is not False:
            raise ValueError("Only development-only evaluations can be summarized")
        if report.get("selection_policy") != "fixed_epoch_budget_final_checkpoint":
            raise ValueError("Seed evaluations must use the same fixed final-epoch policy")
        pair_counts = report["pair_status_counts"]
        if set(pair_counts) - {"complete_pair", "incomplete_pair", "empty_pair"} or sum(pair_counts.values()) != 25:
            raise ValueError("Workbook pair assignments must be verified for all 25 runs")
        if not report.get("assignments_sha256"):
            raise ValueError("Workbook assignment provenance is required")
        config = report["training_config"]
        if config.get("seed") != seed:
            raise ValueError("Training configuration seed differs from the report seed")
        identity = {
            "manifest_sha256": report["manifest_sha256"],
            "assignments_sha256": report["assignments_sha256"],
            "oof_roster_sha256": report["oof_roster_sha256"],
            "classes": report["classes"],
            "label_source": report["label_source"],
            "epochs": report["epochs"],
            "training_config": comparable_training_config(config),
            "pair_status_counts": pair_counts,
            "expected_oof_images": report["expected_oof_images"],
            "expected_oof_subjects": report["expected_oof_subjects"],
            "assigned_development_subjects": report["assigned_development_subjects"],
        }
        if reference is not None and identity != reference:
            raise ValueError("Seed reports differ in cohort, assignments, epoch budget or training settings")
        reference = identity

        filename = report["outputs"]["oof_predictions"]
        if Path(filename).name != filename:
            raise ValueError("Prediction filename must refer to a file beside the seed summary")
        predictions_path = path.parent / filename
        if hashlib.sha256(predictions_path.read_bytes()).hexdigest() != report["oof_predictions_sha256"]:
            raise ValueError(f"Seed {seed} prediction file changed after aggregation")
        rows = read_csv(predictions_path)
        if (len(rows) != report["expected_oof_images"]
                or len({row["drawing_id"] for row in rows}) != len(rows)
                or len({row["participant_id"] for row in rows}) != report["expected_oof_subjects"]
                or roster_sha256(rows) != report["oof_roster_sha256"]):
            raise ValueError(f"Seed {seed} does not contain the complete expected OOF roster")
        if any(int(row["seed"]) != seed or row["manifest_sha256"] != report["manifest_sha256"] for row in rows):
            raise ValueError(f"Seed {seed} prediction provenance mismatch")
        scores = score_predictions(rows, tuple(report["classes"]))
        per_seed.append({"seed": seed, "summary_path": str(path), "scores": scores})
        reports[seed] = report

    if set(reports) != set(seeds):
        raise ValueError(f"Missing predetermined seeds: {sorted(set(seeds) - set(reports))}")
    per_seed.sort(key=lambda item: seeds.index(item["seed"]))
    metrics = {}
    csv_rows = []
    for weighting in ("image_weighted", "subject_balanced"):
        metrics[weighting] = {}
        names = ["accuracy", "macro_f1_all_classes", "macro_f1_present_classes"]
        names += [f"{name}_{metric}" for name in reference["classes"] for metric in ("precision", "recall", "f1")]
        for metric_name in names:
            values = []
            for item in per_seed:
                score = item["scores"][weighting]
                if metric_name in score:
                    value = score[metric_name]
                else:
                    class_name, metric = metric_name.rsplit("_", 1)
                    value = score["per_class"][class_name][metric]
                if value is None or not math.isfinite(value):
                    raise ValueError(f"Cannot summarize undefined metric: {weighting}.{metric_name}")
                values.append(value)
                csv_rows.append({"seed": item["seed"], "weighting": weighting,
                                 "metric": metric_name, "value": value})
            metrics[weighting][metric_name] = {
                "mean": mean(values), "std_across_seeds": stdev(values) if len(values) > 1 else None,
                "values_by_seed": {str(item["seed"]): value for item, value in zip(per_seed, values)},
            }
    result = {
        "status": "complete", "schema_version": 1,
        "protocol": "Repeated available-data evaluation of prescribed paired subject holdouts",
        "seeds": seeds, "seed_count": len(seeds),
        **reference,
        "primary_metric": "subject_balanced.macro_f1_all_classes",
        "metrics": metrics,
        "per_seed": per_seed,
        "all_subject_pairs_complete": all(report["all_subject_pairs_complete"] for report in reports.values()),
        "uncertainty_note": "Seed standard deviation measures training randomness, not a population confidence interval. Images from a subject and CV runs are dependent.",
        "interpretation": "Pooled metrics describe held-out predictions on the available cohort. Seed repetitions do not create additional independent subjects; no best seed is selected.",
        "ensemble_weights_derived": False,
        "test_partition_loaded_or_evaluated": False,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "repeated_validation_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output_dir / "metrics_by_seed.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "weighting", "metric", "value"])
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = ["# Development validation across seeds", "",
             f"Predetermined seeds: {', '.join(map(str, seeds))}.", "",
             f"Each seed evaluated {reference['expected_oof_images']} images from "
             f"{reference['expected_oof_subjects']} available subjects. Repetitions are not additional subjects.", "",
             f"Pair coverage: {json.dumps(reference['pair_status_counts'], sort_keys=True)}.", "",
             "| Weighting | Metric | Mean | Standard deviation across seeds |",
             "| --- | --- | ---: | ---: |"]
    for weighting in metrics:
        for metric in ("accuracy", "macro_f1_all_classes", "parkinson_f1"):
            if metric not in metrics[weighting]:
                continue
            item = metrics[weighting][metric]
            deviation = f"{item['std_across_seeds']:.4f}" if item["std_across_seeds"] is not None else "not estimated"
            lines.append(f"| {weighting} | {metric} | {item['mean']:.4f} | {deviation} |")
    lines += ["", "Subject-balanced results give every subject total weight 1 across their images. "
              "They are image-label metrics, not clinical patient diagnoses.", "",
              result["uncertainty_note"], "",
              "Incomplete pairs contribute available held-out predictions. Empty pairs contribute no predictions or score. "
              "No test data or validation-derived ensemble weights are used."]
    (output_dir / "Report.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = summarize_seed_reports(args.summaries, args.seeds, args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    print(f"Combined {result['seed_count']} seeds; {result['expected_oof_subjects']} subjects per seed.")
    print(f"Report: {args.output_dir / 'Report.md'}")


if __name__ == "__main__":
    main()
