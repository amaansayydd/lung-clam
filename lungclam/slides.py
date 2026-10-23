"""Per-slide processing: open a slide, segment tissue, grid it into patches, save QC images and stats.

Segmentation and patching are CLAM's own (WholeSlideImage). This module only adapts them:
  * IDC slides are DICOM *series* (a folder of .dcm files, one per pyramid level);
    OpenSlide 4 opens the whole series from any one file in it.
  * Patch size comes from the manifest (`patch_px`), so every patch covers ~128 µm whatever
    the scan resolution. Patches are later resized to 224 px for the encoders.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .paths import add_clam_to_path

add_clam_to_path()
from wsi_core.WholeSlideImage import WholeSlideImage  # noqa: E402

# CLAM's TCGA preset (presets/tcga.csv): saturation threshold 8, median blur 7, closing 4,
# keep tissue pieces > 16 reference patches, holes > 4 reference patches.
SEG_PARAMS = dict(sthresh=8, sthresh_up=255, mthresh=7, close=4, use_otsu=False)
FILTER_PARAMS = dict(a_t=16, a_h=4, max_n_holes=8)
SEG_TARGET_DOWNSAMPLE = 32


def find_slide_file(series_dir: Path) -> Path:
    files = sorted(series_dir.rglob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"no .dcm files under {series_dir}")
    return files[0]


def open_wsi(series_dir: Path, slide_id: str) -> WholeSlideImage:
    wsi = WholeSlideImage(str(find_slide_file(series_dir)))
    wsi.name = slide_id  # CLAM names outputs after the file; use the slide id instead
    return wsi


def seg_level(wsi: WholeSlideImage) -> int:
    return wsi.getOpenSlide().get_best_level_for_downsample(SEG_TARGET_DOWNSAMPLE)


def segment_and_patch(wsi: WholeSlideImage, patch_px: int, out_dir: Path) -> tuple[Path, int]:
    """Tissue segmentation + non-overlapping patch grid at level 0. Returns (h5 path, n patches)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wsi.segmentTissue(seg_level=seg_level(wsi), **SEG_PARAMS, filter_params=FILTER_PARAMS, ref_patch_size=patch_px)
    h5_path = out_dir / f"{wsi.name}.h5"
    h5_path.unlink(missing_ok=True)
    wsi.process_contours(str(out_dir), patch_level=0, patch_size=patch_px, step_size=patch_px,
                         contour_fn="four_pt", use_padding=True)
    if not h5_path.exists():  # no tissue found
        return h5_path, 0
    import h5py
    with h5py.File(h5_path, "r") as f:
        return h5_path, len(f["coords"])


def tissue_mask(wsi: WholeSlideImage, level: int) -> np.ndarray:
    """Binary mask of the segmented tissue (contours minus holes) at the given level."""
    w, h = wsi.level_dim[level]
    scale = 1 / np.array(wsi.level_downsamples[level])
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, wsi.scaleContourDim(wsi.contours_tissue, scale), -1, 1, thickness=cv2.FILLED)
    for holes in wsi.holes_tissue:
        cv2.drawContours(mask, wsi.scaleContourDim(holes, scale), -1, 0, thickness=cv2.FILLED)
    return mask.astype(bool)


def save_qc_and_stats(wsi: WholeSlideImage, slide_id: str, mpp: float, n_patches: int,
                      thumbs_dir: Path, stats_dir: Path, max_size: int = 2048) -> dict:
    """Thumbnail, segmentation overlay, and low-level slide statistics.

    The statistics feed the shortcut detector (ROADMAP §6.4): if colour, tissue area or
    scan geometry alone can predict the label, the dataset leaks non-biological signal.
    """
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    level = seg_level(wsi)
    img = np.asarray(wsi.getOpenSlide().read_region((0, 0), level, wsi.level_dim[level]).convert("RGB"))
    mask = tissue_mask(wsi, level)

    thumb = Image.fromarray(img)
    thumb.thumbnail((max_size, max_size))
    thumb.save(thumbs_dir / f"{slide_id}.jpg", quality=88)
    wsi.visWSI(vis_level=level, line_thickness=50, max_size=1024).save(thumbs_dir / f"{slide_id}_seg.jpg", quality=80)

    px = img[mask] if mask.any() else img.reshape(-1, 3)
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(float)
    ds = float(np.mean(wsi.level_downsamples[level]))
    um_per_px_level = mpp * ds
    stats = {
        "slide_id": slide_id,
        "n_patches": int(n_patches),
        "n_tissue_pieces": len(wsi.contours_tissue),
        "tissue_area_mm2": float(mask.sum() * (um_per_px_level / 1000) ** 2),
        "tissue_fraction": float(mask.mean()),
        "width_mm": wsi.level_dim[0][0] * mpp / 1000,
        "height_mm": wsi.level_dim[0][1] * mpp / 1000,
        "mpp": mpp,
        **{f"rgb_mean_{c}": float(px[:, i].mean()) for i, c in enumerate("rgb")},
        **{f"rgb_std_{c}": float(px[:, i].std()) for i, c in enumerate("rgb")},
        **{f"hsv_mean_{c}": float(hsv[:, i].mean()) for i, c in enumerate("hsv")},
        **{f"hsv_std_{c}": float(hsv[:, i].std()) for i, c in enumerate("hsv")},
        "background_rgb_mean": float(img[~mask].mean()) if (~mask).any() else float("nan"),
    }
    (stats_dir / f"{slide_id}.json").write_text(json.dumps(stats, indent=1))
    return stats
