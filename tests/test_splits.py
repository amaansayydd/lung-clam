"""Guards against the mistakes that silently inflate WSI results: patient leakage between splits."""
import glob

import pandas as pd
import pytest

from lungclam import paths

MANIFEST = pd.read_csv(paths.MANIFEST)
PATIENT = dict(zip(MANIFEST.slide_id, MANIFEST.patient_id))
SPLIT_FILES = sorted(glob.glob(str(paths.SPLITS_DIR / "lung3_lf*" / "splits_*.csv")))


def _patients(df, col):
    return {PATIENT[s] for s in df[col].dropna()}


@pytest.mark.parametrize("path", SPLIT_FILES)
def test_no_patient_in_two_splits(path):
    df = pd.read_csv(path, index_col=0)
    tr, va, te = (_patients(df, c) for c in ("train", "val", "test"))
    assert not tr & va and not tr & te and not va & te


@pytest.mark.parametrize("path", SPLIT_FILES)
def test_all_slides_of_a_patient_stay_together(path):
    df = pd.read_csv(path, index_col=0)
    for col in ("val", "test"):  # train may be a label-fraction subset, but whole patients only
        pats = _patients(df, col)
        expected = set(MANIFEST[MANIFEST.patient_id.isin(pats)].slide_id)
        assert set(df[col].dropna()) == expected


def test_test_set_identical_across_folds_and_fractions():
    tests = {frozenset(pd.read_csv(p, index_col=0).test.dropna()) for p in SPLIT_FILES}
    assert len(tests) == 1


def test_external_patients_never_in_cptac():
    ext = set(MANIFEST[MANIFEST.split_group == "external"].patient_id)
    assert not ext & set(MANIFEST[MANIFEST.split_group == "cptac"].patient_id)


def test_label_fractions_are_nested():
    for k in range(5):
        prev = set()
        for frac in (10, 25, 50, 100):
            cur = set(pd.read_csv(paths.SPLITS_DIR / f"lung3_lf{frac}" / f"splits_{k}.csv", index_col=0).train.dropna())
            assert prev <= cur
            prev = cur
