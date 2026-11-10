from utils.metrics import classification_metrics


def test_metrics_report_absent_classes_separately_from_present_class_macro():
    result = classification_metrics(
        targets=[0, 0, 0, 0],
        predictions=[0, 0, 0, 1],
        class_names=("control", "parkinson", "unknown"),
    )
    assert result["samples"] == 4
    assert result["per_class"]["control"]["support"] == 4
    assert result["per_class"]["parkinson"]["support"] == 0
    assert result["per_class"]["unknown"]["support"] == 0
    assert result["present_classes"] == ["control"]
    assert result["absent_classes"] == ["parkinson", "unknown"]
    assert result["macro_f1_present_classes"] == result["per_class"]["control"]["f1"]
    assert result["macro_f1_all_classes"] == result["macro_f1"]


def test_metrics_handle_empty_input_without_nan():
    result = classification_metrics([], [], ("control", "parkinson", "unknown"))
    assert result["samples"] == 0
    assert result["accuracy"] == 0.0
    assert result["macro_f1_present_classes"] is None
    assert result["present_classes"] == []
