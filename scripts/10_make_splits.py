"""Freeze the patient-level splits used by every experiment (ROADMAP §3.5).

CPTAC patients (not slides) are split:
  * 20% → internal test set, fixed for all folds, looked at once at the very end;
  * the remaining 80% → 5 cross-validation folds (fold k is validation, the other 4 train).
A patient's tumor and normal slides always land in the same split. Patients are stratified
by which classes they contribute (LUAD only, LUAD+Normal, LUSC only, LUSC+Normal).

Label-efficiency splits keep val/test identical and subsample the *training* patients to
10/25/50% (stratified), nested so that the 10% set is inside the 25% set, and so on.

Outputs (committed, so results are reproducible):
  splits/dataset_cptac.csv, splits/dataset_external.csv    CLAM dataset tables (case_id, slide_id, label)
  splits/lung3_lf{10,25,50,100}/splits_{k}.csv              CLAM split files (train / val / test columns)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungclam import paths  # noqa: E402

LABELS = {"Normal": 0, "LUAD": 1, "LUSC": 2}
FRACTIONS = (0.10, 0.25, 0.50, 1.00)


def clam_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"patient_id": "case_id"})[["case_id", "slide_id", "label"]]


def write_split(dirpath: Path, k: int, train, val, test, slide_of) -> None:
    cols = {name: pd.Series(sorted(s for p in pats for s in slide_of[p]))
            for name, pats in (("train", train), ("val", val), ("test", test))}
    dirpath.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_csv(dirpath / f"splits_{k}.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    m = pd.read_csv(paths.MANIFEST)
    cptac, ext = m[m.split_group == "cptac"], m[m.split_group == "external"]
    paths.SPLITS_DIR.mkdir(exist_ok=True)
    clam_table(cptac).to_csv(paths.SPLITS_DIR / "dataset_cptac.csv", index=False)
    clam_table(ext).to_csv(paths.SPLITS_DIR / "dataset_external.csv", index=False)

    # one row per patient with a stratum describing what the patient contributes
    pat = cptac.groupby("patient_id").label.apply(lambda s: "+".join(sorted(s))).rename("stratum")
    slide_of = cptac.groupby("patient_id").slide_id.apply(list).to_dict()
    ids, strata = pat.index.to_numpy(), pat.to_numpy()

    dev_ids, test_ids = train_test_split(ids, test_size=args.test_frac, stratify=strata, random_state=args.seed)
    dev_strata = pat.loc[dev_ids].to_numpy()
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    rng = np.random.default_rng(args.seed)

    rows = []
    for k, (tr_idx, va_idx) in enumerate(skf.split(dev_ids, dev_strata)):
        train, val = dev_ids[tr_idx], dev_ids[va_idx]
        # nested label-fraction subsets: rank training patients randomly within each stratum once
        order = pd.DataFrame({"p": train, "s": pat.loc[train].to_numpy(), "r": rng.random(len(train))})
        order["q"] = order.groupby("s").r.rank(pct=True)
        for frac in FRACTIONS:
            sub = order.loc[order.q <= frac + 1e-9, "p"].to_numpy()
            write_split(paths.SPLITS_DIR / f"lung3_lf{int(frac * 100)}", k, sub, val, test_ids, slide_of)
            n_sl = cptac[cptac.patient_id.isin(sub)].label.value_counts()
            rows.append({"fold": k, "label_frac": frac, "train_patients": len(sub),
                         **{f"train_{c}": int(n_sl.get(c, 0)) for c in LABELS},
                         "val_patients": len(val), "test_patients": len(test_ids)})

    summary = pd.DataFrame(rows)
    print(summary[summary.label_frac == 1.0].to_string(index=False))
    print("\ntraining patients per label fraction (fold 0):",
          summary[summary.fold == 0].set_index("label_frac").train_patients.to_dict())
    test = cptac[cptac.patient_id.isin(test_ids)]
    print(f"internal test: {len(test_ids)} patients, {len(test)} slides "
          f"({test.label.value_counts().to_dict()})")
    print(f"external test: {len(ext)} slides ({ext.label.value_counts().to_dict()})")


if __name__ == "__main__":
    main()
