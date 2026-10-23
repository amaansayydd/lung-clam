"""Shortcut detector (ROADMAP §6.4): can the label be predicted WITHOUT looking at tissue morphology?

Fits simple classifiers on low-level slide statistics only — mean/std colour of the tissue,
tissue area, number of tissue pieces, slide size, background colour, scan resolution — using
the same patient-level folds as the real models. If these reach a high AUC, the dataset leaks
non-biological signal (stain batches, scanner, how big the specimen is) and the real models'
scores must be read with that in mind. Either way, the result goes in the README.

Writes predictions in the same format as 20_train.py, under results/runs/shortcut_<model>/,
so 30_evaluate.py scores them identically.

  uv run python scripts/40_shortcut_baseline.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from lungclam import paths  # noqa: E402

LABELS = {"Normal": 0, "LUAD": 1, "LUSC": 2}
FEATURES = ["tissue_area_mm2", "tissue_fraction", "n_tissue_pieces", "width_mm", "height_mm", "mpp",
            "background_rgb_mean"] + [f"{s}_{c}" for s in ("rgb_mean", "rgb_std") for c in "rgb"] \
           + [f"{s}_{c}" for s in ("hsv_mean", "hsv_std") for c in "hsv"]
MODELS = {
    "lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
    "rf": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=0),
}


def load_stats() -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in paths.STATS.glob("*.json")]
    if not rows:
        raise SystemExit(f"no slide statistics under {paths.STATS}; run scripts/run_stream.py first")
    return pd.DataFrame(rows)


def main():
    stats = load_stats()
    man = pd.read_csv(paths.MANIFEST)[["slide_id", "label", "split_group"]]
    df = man.merge(stats, on="slide_id")
    df["y"] = df.label.map(LABELS)
    cptac, ext = df[df.split_group == "cptac"], df[df.split_group == "external"]
    print(f"slides with statistics: {len(cptac)} CPTAC, {len(ext)} external")

    for mname, make in MODELS.items():
        out_dir = paths.RESULTS / "runs" / f"shortcut_{mname}" / "s1"
        out_dir.mkdir(parents=True, exist_ok=True)
        for k in range(5):
            split = pd.read_csv(paths.SPLITS_DIR / "lung3_lf100" / f"splits_{k}.csv", index_col=0)
            tr = cptac[cptac.slide_id.isin(split.train.dropna())]
            va = cptac[cptac.slide_id.isin(split.val.dropna())]
            if tr.y.nunique() < 3 or va.empty:
                print(f"  fold {k}: not enough slides yet, skipping")
                continue
            clf = make().fit(tr[FEATURES], tr.y)
            preds = []
            for name, part in (("val", va), ("external", ext)):
                if part.empty:
                    continue
                p = clf.predict_proba(part[FEATURES])
                preds.append(pd.DataFrame({"split": name, "slide_id": part.slide_id.values, "label": part.y.values,
                                           **{f"p_{c}": p[:, j] for c, j in LABELS.items()}}))
            pd.concat(preds).to_csv(out_dir / f"preds_fold{k}.csv", index=False)
        if mname == "rf" and len(cptac):
            imp = pd.Series(make().fit(cptac[FEATURES], cptac.y).feature_importances_, FEATURES)
            print("  random-forest feature importance (all CPTAC):\n" +
                  imp.sort_values(ascending=False).head(8).round(3).to_string())
    print("score with: uv run python scripts/30_evaluate.py shortcut_lr shortcut_rf")


if __name__ == "__main__":
    main()
