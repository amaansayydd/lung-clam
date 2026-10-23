# lung-clam

Weakly supervised **Normal / LUAD / LUSC** classification of lung whole-slide images with
[CLAM](https://github.com/mahmoodlab/CLAM). It's built to be honest about what the model learns, and to run on a laptop plus a free Colab GPU.

> 🚧 Work in progress: the pipeline is complete; results are being produced. See [ROADMAP.md](ROADMAP.md).
> ⚠️ Research and education only. Not a medical device, and not for clinical use.

<p align="center"><a href="https://colab.research.google.com/github/amaansayydd/lung-clam/blob/main/notebooks/lung_clam_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a></p>

## What's different about this project

| | Typical CLAM tutorial | This project |
|---|---|---|
| Normal class | TCGA normals (frozen sections) vs tumour (FFPE) → the model can learn the *preparation*, not the cancer | CPTAC normal-adjacent slides **from the same patients** as the tumour slides, same preparation and scanner |
| Splits | by slide | by **patient**, plus a sealed internal test set and a 68-hospital **external** test (TCGA) |
| Encoder | ImageNet ResNet50 | ResNet50 **vs** a pathology foundation model (Phikon), same splits |
| What CLAM adds | not measured | CLAM-SB vs CLAM-MB vs ABMIL vs max- vs mean-pooling, and **label-efficiency** curves (10–100% of labels) |
| Shortcuts | not checked | a "shortcut detector" trained on colour and size only. If it scores well, the data leaks |
| Uncertainty | single number | 95% CIs from a patient-level bootstrap |

## Data

| Set | Source | Slides | Patients | Role |
|---|---|---|---|---|
| CPTAC LUAD / LUSC / Normal | [NCI Imaging Data Commons](https://portal.imaging.datacommons.cancer.gov/) (CPTAC-LUAD, CPTAC-LSCC) | 150 / 150 / 150 | 300 | 5-fold CV (240 patients) + sealed internal test (60 patients) |
| TCGA LUAD / LUSC (FFPE diagnostic) | IDC (TCGA-LUAD, TCGA-LUSC) | 60 / 60 | 120, 68 hospitals | external test |

The list of slides is fixed in [`manifests/slides.csv`](manifests/slides.csv), built by [`scripts/01_build_manifest.py`](scripts/01_build_manifest.py) with seed 42. Findings about the data are in [`docs/data_findings.md`](docs/data_findings.md).

## Pipeline

```
IDC (cloud) ──▶ download one slide ──▶ tissue segmentation (CLAM) ──▶ 128 µm patch grid
            ──▶ ResNet50 + Phikon features (fp16) ──▶ QC thumbnail + slide stats ──▶ delete slide
features ──▶ CLAM / baselines (5 patient-level folds) ──▶ metrics + bootstrap CIs ──▶ heatmaps
```

- **Streaming:** raw slides are only on disk while they're processed, so the 262 GB dataset needs a few GB of free space.
- **Resolution-independent patches:** every patch covers 128 µm whatever the scan resolution (256 px at 20×, 512 px at 40×), resized to 224 px on the GPU.
- **Resumable:** each step skips work whose output already exists.

## Running it

### On Google Colab (recommended for the heavy steps)

Open [`notebooks/lung_clam_colab.ipynb`](https://colab.research.google.com/github/amaansayydd/lung-clam/blob/main/notebooks/lung_clam_colab.ipynb) in Colab, select a **T4 GPU**, and choose **Run all**. It stores about 4–5 GB in Google Drive and continues where it stopped if Colab disconnects.

### Locally (macOS with Apple Silicon, or Linux)

Requires git and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/amaansayydd/lung-clam.git
cd lung-clam
bash scripts/setup.sh                                 # CLAM + patches + Python 3.10 env + self-check
uv run python scripts/run_stream.py --pilot           # 30 pilot slides end to end
uv run python scripts/run_experiments.py core         # train
uv run python scripts/30_evaluate.py                  # tables → results/tables/
uv run python scripts/60_figures.py                   # figures → results/figures/
uv run python scripts/50_heatmaps.py clam_sb_phikon   # heatmaps → results/heatmaps/
uv run --group dev pytest                             # patient-leakage tests
```

| Script | Does |
|---|---|
| `00_explore_idc.py` | inventory of lung slides in IDC |
| `01_build_manifest.py` | choose the 570 slides (one per patient per class) |
| `run_stream.py` | download → segment → patch → features → QC → delete |
| `10_make_splits.py` | sealed test set + 5 patient-level folds + nested label fractions |
| `20_train.py` | one experiment (YAML in `configs/experiments/`), all folds |
| `run_experiments.py` | groups of experiments (`core`, `ablations`, `label_efficiency`) |
| `30_evaluate.py` | CV / verified-only / external / (once) internal-test tables with 95% CIs |
| `40_shortcut_baseline.py` | colour-and-size-only classifiers on the same folds |
| `50_heatmaps.py` | attention heatmaps + top-patch galleries |
| `60_figures.py` | README figures |

## Results

*Coming soon.* Every number will be generated from the prediction files committed under `results/runs/`.

## Changes to CLAM

CLAM is a pinned git submodule. Our changes are kept in [`patches/0001-apple-mps-and-fp16-features.patch`](patches/0001-apple-mps-and-fp16-features.patch):
- The device is chosen as `cuda → mps → cpu`, and the hard-coded `.cuda()` calls are replaced.
- smooth-topk's SVM losses are rebuilt after moving them to a device. They capture their label buffer when built, which breaks on MPS.
- Checkpoints load with `map_location`, and fp16 feature files load as fp32.
- Labels are mapped in one vectorized step. CLAM's cell-by-cell assignment fails on pandas 3.

On top of CLAM, this repo adds a mean-pooling baseline, a multi-encoder streaming extractor, and the evaluation code. For the 3-class task CLAM runs with `subtyping=False`: tumour slides contain normal tissue, so the "subtyping" out-of-class instance loss would push the wrong way.

## Citations

- Lu, M.Y. et al. *Data-efficient and weakly supervised computational pathology on whole-slide images.* Nature Biomedical Engineering 5, 555–570 (2021).
- Filiot, A. et al. *Scaling self-supervised learning for histopathology with masked image modeling* (Phikon), medRxiv (2023).
- Fedorov, A. et al. *National Cancer Institute Imaging Data Commons.* RadioGraphics 43(12) (2023).
- Clinical Proteomic Tumor Analysis Consortium (CPTAC) LUAD and LSCC collections; The Cancer Genome Atlas (TCGA) LUAD and LUSC.

## License

GPL-3.0, inherited from CLAM. The slide data stays with its sources and is not redistributed here.
