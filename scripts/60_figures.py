"""Figures for the README, all generated from saved predictions and results/tables/metrics.csv.

  results/figures/model_comparison.png    CV macro AUC ± 95% CI for every full-label experiment
  results/figures/label_efficiency.png    CV macro AUC vs % of training labels (CLAM-SB, ABMIL, mean pooling)
  results/figures/confusion_<exp>.png     out-of-fold confusion matrix
  results/figures/roc_<exp>.png           CV one-vs-rest ROC + external LUAD-vs-LUSC ROC

  uv run python scripts/60_figures.py --main clam_sb_phikon
"""
import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import auc as area, roc_curve  # noqa: E402

from lungclam import metrics as M  # noqa: E402
from lungclam import paths  # noqa: E402

_spec = importlib.util.spec_from_file_location("ev", Path(__file__).with_name("30_evaluate.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

# validated reference palette (dataviz skill): first three categorical slots pass all-pairs CVD checks
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
ENCODER_COLOR = {"resnet50_trunc": SERIES[0], "phikon": SERIES[1], "phikon_v2": SERIES[2], "uni_v1": SERIES[2]}
MODEL_NAME = {"clam_sb": "CLAM-SB", "clam_mb": "CLAM-MB", "abmil": "ABMIL", "maxmil": "Max-pool MIL",
              "meanpool": "Mean-pool", "shortcut_lr": "Shortcut (colour/size, LR)",
              "shortcut_rf": "Shortcut (colour/size, RF)"}
ENCODER_NAME = {"resnet50_trunc": "ResNet50 (ImageNet)", "phikon": "Phikon (pathology)",
                "phikon_v2": "Phikon-v2", "uni_v1": "UNI"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def parse(name: str) -> dict:
    """Model, encoder and label fraction of an experiment, from its saved config."""
    if name.startswith("shortcut_"):
        return {"model": name, "encoder": None, "lf": 100}
    cfg = yaml.safe_load((paths.RESULTS / "runs" / name / "config.yaml").read_text())
    return {"model": cfg["model"], "encoder": cfg["encoder"], "lf": int(cfg.get("label_frac", 100))}


def model_comparison(tab: pd.DataFrame, out: Path):
    d = tab[(tab.split == "cv") & (tab.metric == "macro_auc")].copy()
    d = d.join(d.experiment.apply(parse).apply(pd.Series))
    d = d[(d.lf == 100) & ~d.experiment.str.startswith("final")].sort_values("value")
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(d) + 1.4))
    y = np.arange(len(d))
    for yi, r in zip(y, d.itertuples()):
        c = ENCODER_COLOR.get(r.encoder, INK2)
        ax.plot([r.lo, r.hi], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax.plot(r.value, yi, "o", ms=8, color=c, mec=SURFACE, mew=2)
        ax.text(r.hi + 0.004, yi, f"{r.value:.3f}", va="center", color=INK2, fontsize=9)
    labels = [MODEL_NAME.get(r.model, r.model) + (f" · {ENCODER_NAME.get(r.encoder, r.encoder)}" if r.encoder else "")
              for r in d.itertuples()]
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Macro one-vs-rest AUC (5-fold CV, 95% CI over patients)")
    ax.set_title("Model and encoder comparison (Normal / LUAD / LUSC)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=ENCODER_COLOR[e], ms=8, label=ENCODER_NAME[e])
               for e in dict.fromkeys(d.encoder.dropna())]
    if len(handles) >= 2:
        ax.legend(handles=handles, loc="lower right")
    lo = max(0.0, d.lo.min() - 0.03)
    ax.set_xlim(lo, min(1.0, d.hi.max() + 0.04))
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=200)
    plt.close(fig)


def label_efficiency(tab: pd.DataFrame, out: Path, encoder="phikon"):
    d = tab[(tab.split == "cv") & (tab.metric == "macro_auc")].copy()
    d = d.join(d.experiment.apply(parse).apply(pd.Series))
    d = d[(d.encoder == encoder) & d.model.isin(["clam_sb", "abmil", "meanpool"]) & ~d.experiment.str.startswith("final")]
    if d.lf.nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c, model in zip(SERIES, ["clam_sb", "abmil", "meanpool"]):
        s = d[d.model == model].sort_values("lf")
        if s.empty:
            continue
        ax.fill_between(s.lf, s.lo, s.hi, color=c, alpha=0.12, lw=0)
        ax.plot(s.lf, s.value, "-o", color=c, lw=2, ms=7, mec=SURFACE, mew=2, label=MODEL_NAME[model])
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100], ["10%", "25%", "50%", "100%"])
    ax.minorticks_off()
    ax.set_xlim(8, 125)
    ax.set_xlabel("Training labels used (share of training patients)")
    ax.set_ylabel("Macro AUC (CV)")
    ax.set_title(f"Label efficiency · {ENCODER_NAME.get(encoder, encoder).split(' (')[0]} features")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "label_efficiency.png", dpi=200)
    plt.close(fig)


def confusion(name: str, out: Path):
    p = ev.load_preds(name)
    cv = ev.average(p[p.split == "val"])
    cm_ = M.confusion(cv)
    cmap = LinearSegmentedColormap.from_list("seq", ["#f1f6fd", SERIES[0], "#143d70"])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(cm_ / cm_.sum(1, keepdims=True), cmap=cmap, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            frac = cm_[i, j] / max(cm_[i].sum(), 1)
            ax.text(j, i, f"{cm_[i, j]}\n{frac:.0%}", ha="center", va="center", fontsize=10,
                    color=SURFACE if frac > 0.55 else INK)
    ax.set_xticks(range(3), M.CLASSES)
    ax.set_yticks(range(3), M.CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.grid(False)
    ax.set_title(f"Out-of-fold confusion ({len(cv)} slides)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"confusion_{name}.png", dpi=200)
    plt.close(fig)


def roc(name: str, out: Path):
    p = ev.load_preds(name)
    cv, ext = ev.average(p[p.split == "val"]), ev.average(p[p.split == "external"])
    fig, axes = plt.subplots(1, 2 if len(ext) else 1, figsize=(9 if len(ext) else 4.6, 4.2), squeeze=False)
    ax = axes[0, 0]
    for c, (k, cls) in zip(SERIES, enumerate(M.CLASSES)):
        fpr, tpr, _ = roc_curve(cv.label == k, cv[f"p_{cls}"])
        auc = area(fpr, tpr)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{cls} vs rest (AUC {auc:.3f})")
    ax.set_title("CPTAC, out-of-fold")
    if len(ext):
        ax2 = axes[0, 1]
        t = ext[ext.label != 0]
        score = t.p_LUSC / (t.p_LUAD + t.p_LUSC + 1e-12)
        fpr, tpr, _ = roc_curve(t.label == 2, score)
        ax2.plot(fpr, tpr, color=SERIES[0], lw=2, label=f"LUSC vs LUAD (AUC {area(fpr, tpr):.3f})")
        ax2.set_title(f"TCGA external ({len(t)} slides)")
    for a in axes[0]:
        a.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
        a.set_xlabel("False positive rate")
        a.set_ylabel("True positive rate")
        a.set_aspect("equal")
        a.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / f"roc_{name}.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default="clam_sb_phikon", help="experiment for confusion matrix and ROC")
    args = ap.parse_args()
    tab = pd.read_csv(paths.RESULTS / "tables" / "metrics.csv")
    out = paths.RESULTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    model_comparison(tab, out)
    label_efficiency(tab, out)
    if (paths.RESULTS / "runs" / args.main).exists():
        confusion(args.main, out)
        roc(args.main, out)
    print("figures:", ", ".join(sorted(p.name for p in out.glob("*.png"))))


if __name__ == "__main__":
    main()
