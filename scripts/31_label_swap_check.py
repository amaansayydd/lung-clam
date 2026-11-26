"""Sensitivity check for suspected swapped tumour/normal labels.

Finds patients whose tumour slide is predicted Normal AND whose normal slide is
predicted tumour (the signature of a label swap in the source metadata), then
re-scores the sealed test set with those two labels exchanged. The pre-registered
result in results/tables/metrics.md is unchanged; this is reported next to it.

    uv run python scripts/31_label_swap_check.py [experiment]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib  # noqa: E402

from lungclam import metrics as M  # noqa: E402

ev = importlib.import_module("30_evaluate")


def swapped_patients(df):
    df = df.assign(pred=df[M.PROB].values.argmax(1))
    out = []
    for pid, g in df.groupby("patient_id"):
        norm, tum = g[g.label == 0], g[g.label != 0]
        if len(norm) == 1 and len(tum) == 1 and norm.pred.iat[0] == tum.label.iat[0] and tum.pred.iat[0] == 0:
            out.append(pid)
    return out


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "final_clam_sb_phikon"
    p = ev.load_preds(name)
    for split, label in [("val", "cv"), ("test", "test")]:
        df = ev.average(p[p.split == split])
        pids = swapped_patients(df)
        print(f"{label}: suspected swapped patients: {pids or 'none'}")
        if label != "test" or not pids:
            continue
        fixed = df.copy()
        for pid in pids:
            m = fixed.patient_id == pid
            tum = fixed.loc[m & (fixed.label != 0), "label"].iat[0]
            fixed.loc[m, "label"] = fixed.loc[m, "label"].map({0: tum, tum: 0})
        for tag, d in [("as recorded", df), ("labels swapped back", fixed)]:
            res = M.bootstrap(d, {k: M.METRICS[k] for k in ev.CV_METRICS}, n=1000)
            print(f"  {tag}: " + ", ".join(f"{k} {v['value']:.3f} ({v['lo']:.3f}–{v['hi']:.3f})" for k, v in res.items()))


if __name__ == "__main__":
    main()
