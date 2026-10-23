"""Score experiments from their saved predictions: tables with patient-bootstrap 95% CIs.

For each experiment in results/runs/<name>/:
  cv        out-of-fold validation predictions (each CPTAC dev patient scored once by the
            model that did not train on it), averaged over seeds
  cv_verified   the same, restricted to patients whose diagnosis is in the clinical record
  external  TCGA slides, probabilities averaged over all fold models and seeds (ensemble)
  test      internal test set, ensemble; only with --unseal (do this once, at the end)

Writes results/tables/metrics.csv (long format) and results/tables/metrics.md (readable).

  uv run python scripts/30_evaluate.py                    # every experiment
  uv run python scripts/30_evaluate.py clam_sb_resnet meanpool_resnet
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from lungclam import metrics as M  # noqa: E402
from lungclam import paths  # noqa: E402

CV_METRICS = ["macro_auc", "balanced_acc", "tumor_vs_normal_auc", "tumor_sensitivity", "normal_specificity",
              "luad_vs_lusc_auc"]
EXT_METRICS = ["luad_vs_lusc_auc", "luad_vs_lusc_bacc", "tumor_sensitivity"]


def load_preds(name: str) -> pd.DataFrame:
    files = sorted((paths.RESULTS / "runs" / name).glob("s*/preds_fold*.csv"))
    if not files:
        raise FileNotFoundError(f"no predictions for {name}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["seed"], df["fold"] = f.parent.name, int(f.stem.replace("preds_fold", ""))
        dfs.append(df)
    man = pd.read_csv(paths.MANIFEST)[["slide_id", "patient_id", "label_verified"]]
    return pd.concat(dfs).merge(man, on="slide_id", how="left")


def average(df: pd.DataFrame) -> pd.DataFrame:
    """One row per slide: probabilities averaged over whatever models scored it."""
    keep = ["slide_id", "patient_id", "label", "label_verified"]
    return df.groupby(keep, as_index=False)[M.PROB].mean()


def score(name: str, unseal: bool, n_boot: int) -> list[dict]:
    p = load_preds(name)
    rows = []

    def add(split, df, metric_names):
        if df.empty:
            return
        res = M.bootstrap(df, {k: M.METRICS[k] for k in metric_names}, n=n_boot)
        for k, v in res.items():
            rows.append({"experiment": name, "split": split, "metric": k, "n_slides": len(df),
                         "n_folds": p[p.split == split.replace("_verified", "")].fold.nunique(), **v})

    cv = average(p[p.split == "val"])
    add("cv", cv, CV_METRICS)
    add("cv_verified", cv[cv.label_verified == True], CV_METRICS)  # noqa: E712
    add("external", average(p[p.split == "external"]), EXT_METRICS)
    if unseal:
        add("test", average(p[p.split == "test"]), CV_METRICS)
    return rows


def fmt(r) -> str:
    return f"{r.value:.3f} ({r.lo:.3f}–{r.hi:.3f})" if pd.notna(r.value) else "—"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="*")
    ap.add_argument("--unseal", action="store_true", help="also score the internal test set")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    names = args.experiments or sorted(d.name for d in (paths.RESULTS / "runs").iterdir() if d.is_dir())
    rows = []
    for name in names:
        try:
            rows += score(name, args.unseal, args.n_boot)
            print(f"scored {name}")
        except FileNotFoundError as e:
            print(f"skip: {e}")
    if not rows:
        return
    df = pd.DataFrame(rows)
    out = paths.RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "metrics.csv", index=False)

    md = []
    for split in ["cv", "cv_verified", "external", "test"]:
        sub = df[df.split == split]
        if sub.empty:
            continue
        wide = sub.assign(cell=sub.apply(fmt, axis=1)).pivot(index="experiment", columns="metric", values="cell")
        cols = [c for c in (CV_METRICS if split != "external" else EXT_METRICS) if c in wide.columns]
        n = sub.groupby("experiment").n_slides.first()
        wide = wide[cols].assign(slides=n)
        md.append(f"### {split}\n\n" + wide.to_markdown() + "\n")
    (out / "metrics.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
