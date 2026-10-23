"""Attention heatmaps and top-patch galleries for a trained attention model (CLAM-SB/MB, ABMIL).

For each chosen slide:
  <slide>_heatmap.jpg   thumbnail | attention overlay (percentile-ranked, as in CLAM)
  <slide>_patches.jpg   the 8 most- and 4 least-attended patches at full 20× resolution
Plus index.csv (label, prediction, probabilities, which model scored the slide) and one
per-class montage of top patches across slides (montage_<class>.jpg).

CPTAC slides are explained by the fold model that did NOT train on them (their out-of-fold
model); external slides by fold 0. Slides are chosen automatically unless --slides is given:
for every class, the most confident correct, the least confident correct, and every error
(capped), from CV and the external test.

  uv run python scripts/50_heatmaps.py clam_sb_phikon
  uv run python scripts/50_heatmaps.py clam_sb_phikon --slides C3L-00422-22 --no-patches
"""
import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from lungclam import paths  # noqa: E402
from lungclam.features import coords_of, load_features  # noqa: E402

_spec = importlib.util.spec_from_file_location("train", Path(__file__).with_name("20_train.py"))
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

CLASSES = ["Normal", "LUAD", "LUSC"]


def choose_slides(preds: pd.DataFrame, per_bucket: int, max_errors: int) -> pd.DataFrame:
    p = preds.copy()
    p["pred"] = p[[f"p_{c}" for c in CLASSES]].to_numpy().argmax(1)
    p["conf"] = p[[f"p_{c}" for c in CLASSES]].max(axis=1)
    if (p.split == "external").any():  # external has no Normal class: subtype call among tumour classes
        ext = p.split == "external"
        p.loc[ext, "pred"] = np.where(p.loc[ext, "p_LUSC"] > p.loc[ext, "p_LUAD"], 2, 1)
    picks = []
    for (_, label), g in p.groupby(["split", "label"]):
        ok = g[g.pred == g.label].sort_values("conf")
        picks += [ok.tail(per_bucket), ok.head(per_bucket), g[g.pred != g.label].head(max_errors)]
    return pd.concat(picks).drop_duplicates("slide_id")


def attention(model, feats: torch.Tensor, pred: int) -> np.ndarray:
    with torch.inference_mode():
        a = model(feats.to(train.core_utils.device), attention_only=True)
    a = a.cpu().float()
    a = a[pred] if a.shape[0] > 1 else a[0]  # CLAM-MB: the branch of the predicted class
    ranks = a.argsort().argsort().float()    # percentile-rank, as CLAM's heatmaps do
    return (ranks / max(len(ranks) - 1, 1)).numpy()


def render_heatmap(thumb: Image.Image, coords: np.ndarray, scores: np.ndarray, patch_px: int,
                   level0_width: int) -> Image.Image:
    s = thumb.width / level0_width
    heat = Image.new("RGBA", thumb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(heat)
    cmap = matplotlib.colormaps["jet"]
    for (x, y), v in zip(coords, scores):
        r, g, b, _ = (np.array(cmap(float(v))) * 255).astype(int)
        draw.rectangle([x * s, y * s, (x + patch_px) * s, (y + patch_px) * s], fill=(r, g, b, 110))
    over = Image.alpha_composite(thumb.convert("RGBA"), heat).convert("RGB")
    canvas = Image.new("RGB", (thumb.width * 2 + 10, thumb.height), "white")
    canvas.paste(thumb, (0, 0))
    canvas.paste(over, (thumb.width + 10, 0))
    return canvas


def patch_gallery(slide, coords, scores, patch_px, n_top=8, n_bottom=4, size=256) -> tuple[Image.Image, list]:
    order = np.argsort(-scores)
    sel = list(order[:n_top]) + list(order[-n_bottom:])
    tiles = [slide.read_region((int(coords[i][0]), int(coords[i][1])), 0, (patch_px, patch_px)).convert("RGB")
             .resize((size, size)) for i in sel]
    canvas = Image.new("RGB", (5 + 4 * (size + 10) + 2 * (size // 2 + 10), size * 2 + 30), "white")
    d = ImageDraw.Draw(canvas)
    d.text((5, 2), "most attended (top 8)", fill="black")
    d.text((5 + 4 * (size + 10), size + 17), "least attended", fill="black")
    for j, t in enumerate(tiles[:n_top]):
        canvas.paste(t, (5 + (j % 4) * (size + 10), 15 + (j // 4) * (size + 15)))
    # least-attended patches, half size, in the right-hand columns
    for j, t in enumerate(tiles[n_top:]):
        canvas.paste(t.resize((size // 2, size // 2)), (5 + 4 * (size + 10) + (j % 2) * (size // 2 + 10),
                                                        size + 35 + (j // 2) * (size // 2 + 10)))
    return canvas, tiles[:n_top]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment")
    ap.add_argument("--slides", nargs="+")
    ap.add_argument("--seed", default="s1")
    ap.add_argument("--per-bucket", type=int, default=1)
    ap.add_argument("--max-errors", type=int, default=3)
    ap.add_argument("--no-patches", action="store_true", help="skip galleries (no slide download needed)")
    args = ap.parse_args()

    run_dir = paths.RUNS / args.experiment
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    if cfg["model"] not in ("clam_sb", "clam_mb", "abmil"):
        raise SystemExit(f"{cfg['model']} has no attention to visualize")
    preds = pd.concat(pd.read_csv(f).assign(fold=int(f.stem[-1]))
                      for f in sorted((run_dir / args.seed).glob("preds_fold*.csv")))
    preds = preds[preds.split.isin(["val", "external"])]
    oof_fold = preds[preds.split == "val"].set_index("slide_id").fold.to_dict()
    preds = preds[(preds.split == "val") | (preds.fold == preds.fold.min())]
    chosen = preds[preds.slide_id.isin(args.slides)] if args.slides else \
        choose_slides(preds, args.per_bucket, args.max_errors)
    man = pd.read_csv(paths.MANIFEST).set_index("slide_id")
    out = paths.RESULTS / "heatmaps" / args.experiment
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(chosen)} slides to visualize")

    models, index, montage = {}, [], {c: [] for c in CLASSES}
    for r in chosen.itertuples():
        k = oof_fold.get(r.slide_id, preds.fold.min())
        if k not in models:
            ca = train.clam_args(cfg, run_dir / args.seed)
            train.use_meanpool(False)
            models[k] = train.eval_utils.initiate_model(ca, str(run_dir / args.seed / f"s_{k}_checkpoint.pt"))
        probs = np.array([getattr(r, f"p_{c}") for c in CLASSES])
        pred = int(probs.argmax()) if r.split == "val" else (2 if r.p_LUSC > r.p_LUAD else 1)
        feats = load_features(paths.feature_dir(cfg["encoder"]) / f"{r.slide_id}.pt")
        coords = coords_of(paths.PATCHES / f"{r.slide_id}.h5")
        scores = attention(models[k], feats, pred)
        m = man.loc[r.slide_id]
        thumb = Image.open(paths.THUMBS / f"{r.slide_id}.jpg").convert("RGB")
        render_heatmap(thumb, coords, scores, int(m.patch_px), int(m.width)).save(out / f"{r.slide_id}_heatmap.jpg",
                                                                                  quality=88)
        if not args.no_patches:
            from lungclam.slides import open_wsi
            sys.path.insert(0, str(paths.ROOT / "scripts"))
            run_stream = importlib.import_module("run_stream")
            run_stream.download(man.loc[[r.slide_id]].reset_index())
            wsi = open_wsi(paths.RAW / m.series_uid, r.slide_id)
            gallery, top = patch_gallery(wsi.getOpenSlide(), coords, scores, int(m.patch_px))
            gallery.save(out / f"{r.slide_id}_patches.jpg", quality=88)
            if pred == r.label:
                montage[CLASSES[pred]] += top[:4]
            shutil.rmtree(paths.RAW / m.series_uid, ignore_errors=True)
        index.append({"slide_id": r.slide_id, "split": r.split, "label": CLASSES[int(r.label)],
                      "pred": CLASSES[pred], "fold_model": k, **{f"p_{c}": round(float(p), 4)
                                                                 for c, p in zip(CLASSES, probs)}})
        print(f"  {r.slide_id}: {CLASSES[int(r.label)]} → {CLASSES[pred]}")

    pd.DataFrame(index).to_csv(out / "index.csv", index=False)
    for c, tiles in montage.items():
        if tiles:
            n = min(len(tiles), 16)
            canvas = Image.new("RGB", (4 * 200, ((n + 3) // 4) * 200), "white")
            for j, t in enumerate(tiles[:n]):
                canvas.paste(t.resize((200, 200)), ((j % 4) * 200, (j // 4) * 200))
            canvas.save(out / f"montage_{c}.jpg", quality=88)
    print(f"saved to {out.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
