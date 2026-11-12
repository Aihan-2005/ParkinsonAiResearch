"""Export one inference-only bundle containing the trained baseline and DeepRBF."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from config import DATA, ROOT_DIR
from model.deep_rbf_system import BUNDLE_FORMAT_VERSION
from utils.io import sha256_file
from utils.loops import select_device
from utils.rbf_pipeline import load_decision_config, load_deep_rbf_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep-rbf-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--decision-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DATA.fold_manifest_path)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    rbf_payload = torch.load(args.deep_rbf_checkpoint, map_location="cpu", weights_only=True)
    source = rbf_payload["source_baseline"]
    baseline_path = Path(args.baseline_checkpoint or source["path"])
    if sha256_file(baseline_path) != source["sha256"]:
        raise ValueError("Baseline checkpoint SHA-256 differs from the DeepRBF source")

    device = select_device(args.device)
    _, _, checkpoint = load_deep_rbf_checkpoint(
        args.deep_rbf_checkpoint, args.manifest, args.root, device, baseline_path
    )
    decision = load_decision_config(args.decision_config, checkpoint)
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)

    loss = baseline.get("loss", {})
    if not isinstance(loss, dict) or loss.get("name") != "focal":
        raise ValueError("The combined pipeline requires a Focal-trained WideResNet baseline")

    bundle = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "class_names": list(checkpoint["class_names"]),
        "fold_id": int(source["fold_id"]),
        "baseline": {
            "architecture": baseline["architecture"],
            "freeze_policy": baseline["freeze_policy"],
            "feature_dim": baseline["feature_dim"],
            "num_classes": baseline["num_classes"],
            "class_names": baseline["class_names"],
            "label_source": baseline["label_source"],
            "loss": baseline["loss"],
            "epoch": baseline["epoch"],
            "fold_id": baseline["fold_id"],
            "fold_manifest_sha256": baseline["fold_manifest_sha256"],
            "image_size": baseline["image_size"],
            "preprocessing": baseline["preprocessing"],
            "checkpoint_sha256": sha256_file(baseline_path),
            "model_state_dict": baseline["model_state_dict"],
        },
        "deep_rbf": {
            "fold_id": int(source["fold_id"]),
            "config": checkpoint["config"],
            "epoch": checkpoint["epoch"],
            "stage_checkpoint_sha256": checkpoint["stage_checkpoint_sha256"],
            "baseline_sha256": source["sha256"],
            "manifest_sha256": source["fold_manifest_sha256"],
            "training_drawing_ids": checkpoint["training_drawing_ids"],
            "training_participant_ids": checkpoint["training_participant_ids"],
            "model_state_dict": checkpoint["model_state_dict"],
        },
        "decision": decision,
        "inference_policy": (
            "Focal-trained WideResNet forward_features -> class-specific affine DeepRBF "
            "distances -> validation-selected rejection thresholds. The WideResNet "
            "classifier logits are not averaged with the DeepRBF distances."
        ),
        "test_partition_loaded_or_evaluated": False,
    }

    output = args.output or args.deep_rbf_checkpoint.parent / "combined_model.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(bundle, temporary)
    temporary.replace(output)
    print(f"Combined inference bundle: {output}")
    print(f"Baseline SHA-256: {bundle['baseline']['checkpoint_sha256']}")
    print(f"DeepRBF SHA-256: {bundle['deep_rbf']['stage_checkpoint_sha256']}")
    print("Bundle includes model weights and decision metadata, not optimizer states.")


if __name__ == "__main__":
    main()
