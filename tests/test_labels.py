from data.labels import build_labels, RETAINED_DOCTOR_IDS


def rating(drawing_id, doctor_id, label, confidence="sure", position=1):
    cohort, subject, image = drawing_id.split("::", 2)
    return {
        "drawing_id": drawing_id,
        "participant_id": f"{cohort}::{subject}",
        "cohort": cohort,
        "subject_name": subject,
        "image_name": image,
        "doctor_id": doctor_id,
        "doctor_name_raw": doctor_id,
        "label_raw": label,
        "label_normalized": label,
        "confidence_raw": confidence,
        "confidence_normalized": confidence,
        "source_csv": "fixture.csv",
        "source_row": "2",
        "source_row_id": "0",
        "source_list_position": str(position),
        "included_in_expert_consensus": "true",
        "vote_counted": "false",
        "repeated_rating_flag": "false",
    }


def test_all_eight_three_rater_vote_patterns():
    # Enumerate the 2^3 possible vote patterns and check the agreed rule.
    import itertools

    for index, votes in enumerate(itertools.product(("control", "parkinson"), repeat=3)):
        drawing = f"control::subject 01::image_{index}.png"
        rows = [
            rating(drawing, doctor, vote, position=i)
            for i, (doctor, vote) in enumerate(zip(RETAINED_DOCTOR_IDS, votes), start=1)
        ]
        labels, _ = build_labels(rows)
        expected = (
            "parkinson" if all(v == "parkinson" for v in votes)
            else "control" if votes.count("control") >= 2
            else "unknown"
        )
        assert labels[0]["expert_label"] == expected


def test_d1_is_not_part_of_the_three_rater_vote():
    drawing = "parkinson::subject 01::image.png"
    rows = [
        rating(drawing, "D1", "control"),
        rating(drawing, "D2", "parkinson"),
        rating(drawing, "D3", "parkinson"),
        rating(drawing, "D4", "parkinson"),
    ]
    labels, _ = build_labels(rows)
    assert labels[0]["expert_label"] == "parkinson"
    assert labels[0]["votes_parkinson"] == "3"


def test_repeated_same_doctor_same_label_counts_once_and_keeps_confidence_conflict():
    drawing = "parkinson::subject 29::1_subject 29_1_1.png"
    rows = [
        rating(drawing, "D2", "control", "sure", 1),
        rating(drawing, "D3", "control", "sure", 2),
        rating(drawing, "D3", "control", "not sure", 3),
        rating(drawing, "D4", "parkinson", "sure", 4),
    ]
    labels, report = build_labels(rows)
    row = labels[0]
    assert row["expert_label"] == "control"
    assert row["votes_control"] == "2"
    assert row["unique_retained_doctors"] == "3"
    assert row["repeated_rating_doctors"] == "D3"
    assert row["confidence_conflict_doctors"] == "D3"
    assert report["repeated_retained_drawing_doctor_pairs"] == 1


def test_conflicting_labels_from_same_doctor_are_unresolved():
    drawing = "control::subject 02::image.png"
    rows = [
        rating(drawing, "D2", "control", position=1),
        rating(drawing, "D3", "control", position=2),
        rating(drawing, "D3", "parkinson", position=3),
        rating(drawing, "D4", "parkinson", position=4),
    ]
    labels, _ = build_labels(rows)
    assert labels[0]["expert_label"] == ""
    assert labels[0]["label_status"] == "needs_adjudication"
