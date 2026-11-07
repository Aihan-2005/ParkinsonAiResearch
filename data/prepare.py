"""Legacy preparation entry point disabled.

The former implementation performed a random subject split. The thesis protocol
uses the supplied folding_TLO.xlsx workbook instead.
"""


def prepare_dataset(*args, **kwargs):
    raise RuntimeError(
        "The legacy random split pipeline is disabled. Run audit_data.py, "
        "finalize_labels.py, then build_folds.py --workbook folding_TLO.xlsx."
    )
