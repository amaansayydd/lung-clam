"""Generates notebooks/lung_clam_colab.ipynb (kept as code so the notebook diff stays reviewable).

  python notebooks/make_colab_notebook.py
"""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# lung-clam on Google Colab

Runs the heavy parts of [lung-clam](https://github.com/amaansayydd/lung-clam) on a free Colab GPU:

1. **Extract**: download each slide from NCI Imaging Data Commons, segment tissue, cut 128 µm patches, encode them with ResNet50 and Phikon, delete the slide.
2. **Train**: CLAM and baselines, 5 patient-level folds each.
3. **Evaluate**: tables, figures, shortcut check, attention heatmaps.

**Before you start:** Runtime → Change runtime type → **T4 GPU**. Then Runtime → **Run all**.

Everything is saved to Google Drive (`MyDrive/lung_clam_data`, about 12 GB), and every step skips work that is already done. If Colab disconnects, reconnect and **Run all** again; it continues where it stopped.

**Keeping the free tier connected:** free Colab disconnects a tab that looks idle (often after about 90 minutes), well before its 5-hour limit. Keep the tab open and the computer awake and plugged in, and click into the notebook now and then. With Colab Pro, runs continue in the background.
""")

md("## 1 · Settings")
code("""
STORAGE = "drive"          # "drive": survives disconnects (recommended) · "local": Colab disk only, zip at the end
DRIVE_FOLDER = "lung_clam_data"
ENCODERS = ["resnet50_trunc", "phikon"]   # add "uni_v1" once the UNI check below says ACCESS GRANTED
RUN_EXTRACTION = True
EXPERIMENT_GROUPS = ["core", "ablations", "label_efficiency"]   # add "uni" after UNI features exist
""")

md("""
### Optional: check UNI access
Needs a Hugging Face **Read** token saved in Colab Secrets (🔑 in the left bar) as `HF_TOKEN`, with *Notebook access* switched on, plus approved access on the UNI model page.
""")
code("""
import os
from huggingface_hub import HfApi, hf_hub_download
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("Hugging Face user:", HfApi().whoami()["name"])
    for repo in ["MahmoodLab/UNI", "MahmoodLab/UNI2-h"]:
        try:
            hf_hub_download(repo, "config.json"); print(f"{repo}: ACCESS GRANTED")
        except Exception as e:
            print(f"{repo}: no access yet ({type(e).__name__})")
except Exception as e:
    print("No usable HF_TOKEN secret:", type(e).__name__, e)
""")

md("## 2 · GPU, storage and code")
code("""
import os, shutil, subprocess
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
assert gpu, "No GPU. Runtime → Change runtime type → T4 GPU, then run again."
print("GPU:", gpu)

if STORAGE == "drive":
    from google.colab import drive
    drive.mount("/content/drive", force_remount=True, timeout_ms=300_000)  # approve the Google popup; allow pop-ups for colab.research.google.com
    DATA = f"/content/drive/MyDrive/{DRIVE_FOLDER}"
else:
    DATA = "/content/lung_clam_data"
os.makedirs(DATA, exist_ok=True)
os.environ.update(LUNGCLAM_DATA=DATA, LUNGCLAM_RAW="/content/raw", LUNGCLAM_RESULTS=f"{DATA}/results",
                  PYTORCH_ENABLE_MPS_FALLBACK="1")
free = shutil.disk_usage(DATA).free / 1e9
print(f"data folder: {DATA}  ({free:.1f} GB free)")
if free < 6:
    print("⚠️  Less than 6 GB free here. Features need about 4–5 GB; free some space or set STORAGE = 'local'.")
""")
code("""
%cd /content
if not os.path.exists("/content/lung-clam"):
    !git clone -q --recurse-submodules https://github.com/amaansayydd/lung-clam.git
else:
    !cd /content/lung-clam && git pull -q && git submodule update -q --init
%cd /content/lung-clam
!git -C CLAM apply ../patches/0001-apple-mps-and-fp16-features.patch 2>/dev/null && echo "CLAM patched" || echo "CLAM patch already applied"
!pip install -q openslide-python openslide-bin idc-index "timm>=1.0.15" h5py future tabulate \\
    "topk @ git+https://github.com/oval-group/smooth-topk.git@12c1645f187e2fa0c05f47bf1fe48864d4bd2707"
!python scripts/check_env.py
""")

md("""
## 3 · Extract features

Downloads, patches and encodes all 570 slides (about 262 GB streamed; only a few slides on disk at a time). Prints one line per slide with a running ETA. Safe to interrupt and re-run.
""")
code("""
if RUN_EXTRACTION:
    !python scripts/run_stream.py --encoders {" ".join(ENCODERS)} --batch 4
""")
code("""
import pandas as pd
st = pd.read_csv(f"{DATA}/status.csv").drop_duplicates("slide_id", keep="last")
print(st.status.value_counts().to_string())
print(f"patches: {st.n_patches.sum():,.0f} total, {st.n_patches.median():,.0f} per slide (median)")
failed = st[st.status != "done"]
if len(failed):
    print("\\nfailed slides (re-run the cell above to retry):")
    print(failed[["slide_id", "detail"]].to_string(index=False))
""")

md("""
## 4 · Train and evaluate

Every experiment trains 5 folds; finished folds are skipped on re-run. The internal test set stays sealed: it is scored only by the final model, in section 6.
""")
code("""
!python scripts/run_experiments.py {" ".join(EXPERIMENT_GROUPS)}
""")
code("""
!python scripts/40_shortcut_baseline.py
!python scripts/30_evaluate.py
from IPython.display import Markdown, display
display(Markdown(open(f"{DATA}/results/tables/metrics.md").read()))
""")

md("## 5 · Figures and heatmaps")
code("""
!python scripts/60_figures.py --main clam_sb_phikon
!python scripts/50_heatmaps.py clam_sb_phikon
from IPython.display import Image, display
for f in ["model_comparison", "label_efficiency", "confusion_clam_sb_phikon", "roc_clam_sb_phikon"]:
    p = f"{DATA}/results/figures/{f}.png"
    if os.path.exists(p):
        display(Image(p, width=700))
""")

md("""
## 6 · Final model and the sealed test set (run once, at the very end)

Trains the pre-registered final configuration with 3 seeds and scores the internal test set for the first and only time. Leave `UNSEAL = False` until every other decision is final.
""")
code("""
UNSEAL = False
if UNSEAL:
    !python scripts/20_train.py configs/experiments/final_clam_sb_phikon.yaml --with-test
    !python scripts/30_evaluate.py --unseal
    display(Markdown(open(f"{DATA}/results/tables/metrics.md").read()))
""")

md("## 7 · Download the results (a few MB) to send back to the Mac")
code("""
shutil.make_archive("/content/lung_clam_results", "zip", f"{DATA}/results")
from google.colab import files
files.download("/content/lung_clam_results.zip")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).with_name("lung_clam_colab.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out)
