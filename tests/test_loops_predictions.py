import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils.loops import run_epoch


class TinyValidationDataset(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        return {
            "image": torch.tensor([float(index), 1.0]),
            "label": torch.tensor(index % 3, dtype=torch.long),
            "drawing_id": f"drawing-{index}",
            "participant_id": f"subject-{index}",
            "image_name": f"{index}.png",
            "image_path": f"/tmp/{index}.png",
        }


def test_validation_epoch_exports_one_probability_record_per_image():
    torch.manual_seed(7)
    model = nn.Linear(2, 3)
    loader = DataLoader(TinyValidationDataset(), batch_size=2, shuffle=False)

    result = run_epoch(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        class_names=("control", "parkinson", "unknown"),
        include_predictions=True,
    )

    assert result["samples"] == 3
    assert len(result["predictions"]) == 3
    first = result["predictions"][0]
    assert first["drawing_id"] == "drawing-0"
    assert first["participant_id"] == "subject-0"
    assert set(first["probabilities"]) == {"control", "parkinson", "unknown"}
    assert abs(sum(first["probabilities"].values()) - 1.0) < 1e-6
