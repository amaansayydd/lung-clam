"""Score slides with a trained experiment, averaging every checkpoint it has (seeds × folds).

Input is one or more feature files made by run_stream.py (<slide_id>.pt, one row per 128 µm
patch, from the same encoder the experiment was trained on). The final model's 15 checkpoints
are attached to the v1.0 GitHub release; unzip them into $LUNGCLAM_DATA/runs/.

  uv run python scripts/70_predict.py final_clam_sb_phikon path/to/SLIDE.pt [more.pt ...]
  uv run python scripts/70_predict.py final_clam_sb_phikon path/to/pt_files/*.pt --out preds.csv

Research and education only; not for clinical use.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from lungclam import paths  # noqa: E402
from lungclam.features import load_features  # noqa: E402

_spec = importlib.util.spec_from_file_location("train", Path(__file__).with_name("20_train.py"))
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

CLASSES = list(train.LABELS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment")
    ap.add_argument("features", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, help="write a CSV instead of printing")
    args = ap.parse_args()

    run_dir = paths.RUNS / args.experiment
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    ckpts = sorted(run_dir.glob("s*/s_*_checkpoint.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {run_dir}")
    train.use_meanpool(cfg["model"] == "meanpool")
    models = [train.eval_utils.initiate_model(train.clam_args(cfg, c.parent), str(c)) for c in ckpts]
    print(f"{args.experiment}: {len(models)} checkpoints, {cfg['model']} on {cfg['encoder']} features")

    rows = []
    with torch.inference_mode():
        for f in args.features:
            feats = load_features(f).to(train.core_utils.device)
            p = np.mean([m(feats)[1].squeeze(0).cpu().numpy() for m in models], axis=0)
            rows.append({"slide_id": f.stem, "n_patches": len(feats), "pred": CLASSES[int(p.argmax())],
                         **{f"p_{c}": round(float(p[i]), 4) for i, c in enumerate(CLASSES)}})
    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"saved {args.out}")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
