"""Where everything lives.

Code and small committed files live in the repo. Heavy data (raw slides, patch coordinates,
features, thumbnails, trained models) lives under DATA_ROOT, which is `data/` in the repo
by default. `data/` is gitignored and can be a symlink to an external drive.
Override with the LUNGCLAM_DATA environment variable. Raw slides can be kept elsewhere
(e.g. a Colab machine's fast local disk while everything else goes to Google Drive) with LUNGCLAM_RAW.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAM_DIR = ROOT / "CLAM"
DATA_ROOT = Path(os.environ.get("LUNGCLAM_DATA", ROOT / "data"))

MANIFEST = ROOT / "manifests" / "slides.csv"
SPLITS_DIR = ROOT / "splits"
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"            # small, committed: tables, figures

RAW = Path(os.environ.get("LUNGCLAM_RAW", DATA_ROOT / "raw"))  # transient: downloaded DICOM series, deleted after processing
PATCHES = DATA_ROOT / "patches"       # <slide_id>.h5 with patch coordinates (CLAM format)
FEATURES = DATA_ROOT / "features"     # <encoder>/pt_files/<slide_id>.pt (fp16)
THUMBS = DATA_ROOT / "thumbs"         # <slide_id>.jpg thumbnails + <slide_id>_seg.jpg QC overlays
STATS = DATA_ROOT / "stats"           # <slide_id>.json low-level slide statistics (shortcut detector)
STATUS = DATA_ROOT / "status.csv"     # per-slide processing log
RUNS = DATA_ROOT / "runs"             # trained models + raw predictions per experiment


def add_clam_to_path() -> None:
    """CLAM is a plain repo, not a package; its modules import each other by top-level name."""
    if str(CLAM_DIR) not in sys.path:
        sys.path.insert(0, str(CLAM_DIR))


def feature_dir(encoder: str) -> Path:
    return FEATURES / encoder / "pt_files"
