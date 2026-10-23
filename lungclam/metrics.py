"""Metrics with patient-level bootstrap confidence intervals.

All metrics take a predictions table with columns slide_id, label (0 Normal, 1 LUAD, 2 LUSC),
p_Normal, p_LUAD, p_LUSC, plus patient_id for resampling.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

CLASSES = ["Normal", "LUAD", "LUSC"]
PROB = [f"p_{c}" for c in CLASSES]


def _probs(df):
    return df[PROB].to_numpy()


def macro_auc(df):
    y = df.label.to_numpy()
    if len(np.unique(y)) < 3:
        return np.nan
    return roc_auc_score(y, _probs(df), multi_class="ovr", average="macro")


def balanced_acc(df):
    return balanced_accuracy_score(df.label, _probs(df).argmax(1))


def tumor_vs_normal_auc(df):
    """Detection: is there cancer on this slide? score = P(LUAD) + P(LUSC)."""
    y = (df.label != 0).astype(int)
    return roc_auc_score(y, df.p_LUAD + df.p_LUSC) if y.nunique() == 2 else np.nan


def luad_vs_lusc_auc(df):
    """Subtyping among tumour slides only: score = P(LUSC) / (P(LUAD) + P(LUSC))."""
    t = df[df.label != 0]
    y = (t.label == 2).astype(int)
    return roc_auc_score(y, t.p_LUSC / (t.p_LUAD + t.p_LUSC + 1e-12)) if y.nunique() == 2 else np.nan


def luad_vs_lusc_bacc(df):
    t = df[df.label != 0]
    pred = np.where(t.p_LUSC > t.p_LUAD, 2, 1)
    return balanced_accuracy_score(t.label, pred) if t.label.nunique() == 2 else np.nan


def tumor_sensitivity(df):
    """Fraction of tumour slides NOT called Normal (missed cancers are the costliest error)."""
    t = df[df.label != 0]
    return float((_probs(t).argmax(1) != 0).mean()) if len(t) else np.nan


def normal_specificity(df):
    n = df[df.label == 0]
    return float((_probs(n).argmax(1) == 0).mean()) if len(n) else np.nan


METRICS = {
    "macro_auc": macro_auc,
    "balanced_acc": balanced_acc,
    "tumor_vs_normal_auc": tumor_vs_normal_auc,
    "tumor_sensitivity": tumor_sensitivity,
    "normal_specificity": normal_specificity,
    "luad_vs_lusc_auc": luad_vs_lusc_auc,
    "luad_vs_lusc_bacc": luad_vs_lusc_bacc,
}


def bootstrap(df: pd.DataFrame, metrics=METRICS, n: int = 1000, seed: int = 0) -> dict:
    """Point estimate and 95% percentile CI, resampling patients (all of a patient's slides together)."""
    rng = np.random.default_rng(seed)
    groups = {p: g for p, g in df.groupby("patient_id")}
    pats = np.array(list(groups))
    out = {}
    samples = {k: [] for k in metrics}
    for _ in range(n):
        boot = pd.concat([groups[p] for p in rng.choice(pats, len(pats), replace=True)])
        for k, f in metrics.items():
            try:
                samples[k].append(f(boot))
            except ValueError:
                samples[k].append(np.nan)
    for k, f in metrics.items():
        s = np.array(samples[k], dtype=float)
        out[k] = {"value": float(f(df)), "lo": float(np.nanpercentile(s, 2.5)) if np.isfinite(s).any() else np.nan,
                  "hi": float(np.nanpercentile(s, 97.5)) if np.isfinite(s).any() else np.nan}
    return out


def confusion(df) -> np.ndarray:
    return confusion_matrix(df.label, _probs(df).argmax(1), labels=[0, 1, 2])


def expected_calibration_error(df, bins: int = 10) -> float:
    p = _probs(df)
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == df.label.to_numpy()).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
