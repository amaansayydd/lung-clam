"""Sanity checks for the metric definitions and the patient-level bootstrap."""
import numpy as np
import pandas as pd

from lungclam import metrics as M


def _toy(n_per_class=10, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for label in range(3):
        for i in range(n_per_class):
            p = np.full(3, 0.05)
            p[label] = 0.9
            p = np.clip(p + rng.normal(0, noise, 3), 1e-3, None)
            rows.append({"slide_id": f"s{label}_{i}", "patient_id": f"p{i}", "label": label,
                         **dict(zip(M.PROB, p / p.sum()))})
    return pd.DataFrame(rows)


def test_perfect_predictions_score_one():
    df = _toy()
    for name in ["macro_auc", "balanced_acc", "tumor_vs_normal_auc", "tumor_sensitivity",
                 "normal_specificity", "luad_vs_lusc_auc", "luad_vs_lusc_bacc"]:
        assert M.METRICS[name](df) == 1.0, name


def test_missing_normal_class_gives_nan_not_error():
    df = _toy()
    tumours = df[df.label != 0]
    assert np.isnan(M.macro_auc(tumours))
    assert np.isnan(M.normal_specificity(tumours))
    assert M.luad_vs_lusc_auc(tumours) == 1.0


def test_tumour_called_normal_lowers_sensitivity():
    df = _toy()
    i = df.index[df.label == 1][0]
    df.loc[i, M.PROB] = [0.9, 0.05, 0.05]
    assert M.tumor_sensitivity(df) == 19 / 20


def test_bootstrap_ci_contains_point_estimate():
    df = _toy(noise=0.4)
    res = M.bootstrap(df, {"macro_auc": M.macro_auc}, n=200)["macro_auc"]
    assert res["lo"] <= res["value"] <= res["hi"]


def test_bootstrap_resamples_whole_patients():
    # every patient has one slide per class; a resample must keep the 3 slides together
    # (a patient drawn k times contributes 3k slides)
    df = _toy()
    seen = []
    M.bootstrap(df, {"check": lambda d: seen.append(d.groupby("patient_id").size().to_numpy()) or 0.0}, n=20)
    assert all((s % 3 == 0).all() for s in seen)
