"""Train one experiment (all folds × seeds) and save its predictions.

An experiment is a YAML file in configs/experiments/. Keys (defaults in DEFAULTS below):
  encoder:      resnet50_trunc | phikon | phikon_v2 | uni_v1 | uni2_h
  model:        clam_sb | clam_mb | abmil | maxmil | meanpool
  label_frac:   10 | 25 | 50 | 100   (percent of training patients)
  folds, seeds, lr, reg, drop_out, max_epochs, early_stopping, weighted_sample,
  bag_weight, B, inst_loss, model_size

Model names map onto CLAM:
  clam_sb  = CLAM single-branch attention + instance clustering (the paper's main model)
  clam_mb  = CLAM multi-branch (one attention branch per class)
  abmil    = clam_sb with instance clustering switched off → plain gated-attention MIL
  maxmil   = CLAM's 'mil' (instance classifier + max pooling)
  meanpool = lungclam.models.MeanPoolMIL (no attention)

The internal test set stays sealed: by default the model is evaluated on validation and
external data only. Pass --with-test for the final, pre-registered configuration.

For every fold and seed, writes runs/<name>/s<seed>/preds_fold<k>.csv with columns
split, slide_id, label, p_Normal, p_LUAD, p_LUSC (split = val | test | external).

  uv run python scripts/20_train.py configs/experiments/clam_sb_resnet.yaml
  uv run python scripts/20_train.py configs/experiments/clam_sb_resnet.yaml --folds 0 --set max_epochs=5
"""
import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from lungclam import paths  # noqa: E402
from lungclam.encoders import ENCODER_DIMS  # noqa: E402

paths.add_clam_to_path()
import utils.core_utils as core_utils  # noqa: E402
import utils.eval_utils as eval_utils  # noqa: E402
import utils.utils as clam_utils  # noqa: E402

if os.environ.get("LUNGCLAM_DEVICE"):  # e.g. "cpu": on Apple Silicon the CPU beats MPS for these small MIL models
    for _mod in (core_utils, eval_utils, clam_utils):
        _mod.device = torch.device(os.environ["LUNGCLAM_DEVICE"])
from dataset_modules.dataset_generic import Generic_MIL_Dataset  # noqa: E402
from lungclam.models import MeanPoolMIL  # noqa: E402

LABELS = {"Normal": 0, "LUAD": 1, "LUSC": 2}
DEFAULTS = dict(encoder="resnet50_trunc", model="clam_sb", label_frac=100, folds=[0, 1, 2, 3, 4], seeds=[1],
                lr=2e-4, reg=1e-5, drop_out=0.25, max_epochs=200, early_stopping=True, weighted_sample=True,
                bag_weight=0.7, B=8, inst_loss="svm", model_size="small", opt="adam")


def load_config(path: Path, overrides: list[str]) -> dict:
    cfg = {**DEFAULTS, **(yaml.safe_load(path.read_text()) or {})}
    for kv in overrides:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    cfg.setdefault("name", path.stem)
    return cfg


def clam_args(cfg: dict, results_dir: Path) -> SimpleNamespace:
    model = cfg["model"]
    return SimpleNamespace(
        results_dir=str(results_dir), log_data=False, testing=False, n_classes=len(LABELS),
        embed_dim=ENCODER_DIMS[cfg["encoder"]], drop_out=cfg["drop_out"], model_size=cfg["model_size"],
        model_type={"abmil": "clam_sb", "maxmil": "mil", "meanpool": "mil"}.get(model, model),
        no_inst_cluster=(model == "abmil"), inst_loss=cfg["inst_loss"], B=cfg["B"], bag_weight=cfg["bag_weight"],
        # Normal / LUAD / LUSC are not mutually exclusive *subtypes*: tumour slides contain normal
        # tissue, so CLAM's out-of-class instance loss ("subtyping") would push it the wrong way.
        subtyping=False,
        bag_loss="ce", opt=cfg["opt"], lr=cfg["lr"], reg=cfg["reg"], max_epochs=cfg["max_epochs"],
        early_stopping=cfg["early_stopping"], weighted_sample=cfg["weighted_sample"])


def use_meanpool(enabled: bool) -> None:
    """CLAM builds model_type='mil' with MIL_fc_mc; swap in the mean-pooling baseline for 'meanpool'."""
    from models.model_mil import MIL_fc_mc
    cls = MeanPoolMIL if enabled else MIL_fc_mc
    core_utils.MIL_fc_mc = cls
    eval_utils.MIL_fc_mc = cls


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def available(table: Path, encoder: str, out: Path) -> Path:
    """Copy of a CLAM dataset table restricted to slides whose features exist."""
    df = pd.read_csv(table)
    have = df.slide_id.map(lambda s: (paths.feature_dir(encoder) / f"{s}.pt").exists())
    if not have.all():
        print(f"  note: {(~have).sum()} of {len(df)} slides in {table.name} have no {encoder} features yet; skipped")
    df[have].to_csv(out, index=False)
    return out


def make_dataset(table: Path, encoder: str, seed: int) -> Generic_MIL_Dataset:
    return Generic_MIL_Dataset(csv_path=str(table), data_dir=str(paths.FEATURES / encoder), shuffle=False,
                               seed=seed, print_info=False, label_dict=LABELS, patient_strat=False, ignore=[])


@torch.inference_mode()
def predict(model, dataset, split_name: str) -> pd.DataFrame:
    rows = []
    device = core_utils.device
    for i in range(len(dataset)):
        feats, label = dataset[i]
        _, y_prob, _, _, _ = model(feats.to(device))
        p = y_prob.squeeze(0).cpu().numpy()
        rows.append({"split": split_name, "slide_id": dataset.slide_data["slide_id"].iloc[i],
                     "label": int(label), **{f"p_{c}": float(p[j]) for c, j in LABELS.items()}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    ap.add_argument("--folds", nargs="+", type=int, help="override which folds to run")
    ap.add_argument("--with-test", action="store_true", help="also predict the sealed internal test set")
    ap.add_argument("--overwrite", action="store_true", help="retrain folds that already have predictions")
    ap.add_argument("--predict-only", action="store_true",
                    help="no training: re-score existing checkpoints (e.g. once external slides have features)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.folds:
        cfg["folds"] = args.folds
    run_dir = paths.RUNS / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"experiment {cfg['name']}: {cfg['model']} on {cfg['encoder']} features, "
          f"{cfg['label_frac']}% labels, folds {cfg['folds']}, seeds {cfg['seeds']}")

    use_meanpool(cfg["model"] == "meanpool")
    cptac_tbl = available(paths.SPLITS_DIR / "dataset_cptac.csv", cfg["encoder"], run_dir / "dataset_cptac.csv")
    ext_tbl = available(paths.SPLITS_DIR / "dataset_external.csv", cfg["encoder"], run_dir / "dataset_external.csv")
    split_dir = paths.SPLITS_DIR / f"lung3_lf{cfg['label_frac']}"

    for seed in cfg["seeds"]:
        seed_dir = run_dir / f"s{seed}"
        seed_dir.mkdir(exist_ok=True)
        dataset = make_dataset(cptac_tbl, cfg["encoder"], seed)
        external = make_dataset(ext_tbl, cfg["encoder"], seed) if len(pd.read_csv(ext_tbl)) else None
        for k in cfg["folds"]:
            out = seed_dir / f"preds_fold{k}.csv"
            ckpt = seed_dir / f"s_{k}_checkpoint.pt"
            if args.predict_only and not ckpt.exists():
                print(f"  seed {seed} fold {k}: no checkpoint to re-score, skipping")
                continue
            if out.exists() and not (args.overwrite or args.predict_only):
                print(f"  seed {seed} fold {k}: already done, skipping")
                continue
            seed_everything(seed)
            train, val, test = dataset.return_splits(from_id=False, csv_path=str(split_dir / f"splits_{k}.csv"))
            if min(len(train), len(val)) == 0:
                print(f"  seed {seed} fold {k}: no training/validation slides with features yet, skipping")
                continue
            print(f"  seed {seed} fold {k}: train {len(train)}, val {len(val)}, test {len(test)} (sealed)")
            t0 = time.time()
            ca = clam_args(cfg, seed_dir)
            if not args.predict_only:
                # CLAM evaluates and prints its third split at the end of training; give it the
                # validation set so the internal test set is never looked at during development.
                core_utils.train((train, val, val), k, ca)
            model = eval_utils.initiate_model(ca, str(ckpt))
            preds = [predict(model, val, "val")]
            if args.with_test and len(test):
                preds.append(predict(model, test, "test"))
            if external is not None:
                preds.append(predict(model, external, "external"))
            pd.concat(preds).to_csv(out, index=False)
            print(f"  seed {seed} fold {k}: done in {(time.time() - t0) / 60:.1f} min")

    # small prediction files are committed so every number in the README can be recomputed
    dest = paths.RESULTS / "runs" / cfg["name"]
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir / "config.yaml", dest / "config.yaml")
    for f in run_dir.glob("s*/preds_fold*.csv"):
        (dest / f.parent.name).mkdir(exist_ok=True)
        shutil.copy(f, dest / f.parent.name / f.name)


if __name__ == "__main__":
    main()
