# lung-clam

[![tests](https://github.com/amaansayydd/lung-clam/actions/workflows/tests.yml/badge.svg)](https://github.com/amaansayydd/lung-clam/actions/workflows/tests.yml)

Weakly supervised **Normal / LUAD / LUSC** classification of lung whole-slide images with
[CLAM](https://github.com/mahmoodlab/CLAM). It's built to be honest about what the model learns, and to run on a laptop plus a free Colab GPU.

> ✅ Cross-validation, external-test and sealed internal-test results are in ([Results](#results)). Next steps: [ROADMAP.md](ROADMAP.md).
> 📄 [Model card](docs/model_card.md) · [Data card](docs/data_card.md) · [Data findings log](docs/data_findings.md)
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
| TCGA LUAD / LUSC (FFPE diagnostic) | [GDC](https://portal.gdc.cancer.gov/) original Aperio `.svs` (IDC's TCGA copies are JPEG 2000 DICOM, which OpenSlide can't read) | 60 / 60 | 120, 68 hospitals | external test |

The list of slides is fixed in [`manifests/slides.csv`](manifests/slides.csv), built by [`scripts/01_build_manifest.py`](scripts/01_build_manifest.py) with seed 42. Findings about the data are in [`docs/data_findings.md`](docs/data_findings.md).

## Pipeline

```
IDC (cloud) ──▶ download one slide ──▶ tissue segmentation (CLAM) ──▶ 128 µm patch grid
            ──▶ ResNet50 + Phikon features (fp16) ──▶ QC thumbnail + slide stats ──▶ delete slide
features ──▶ CLAM / baselines (5 patient-level folds) ──▶ metrics + bootstrap CIs ──▶ heatmaps
```

- **Streaming:** raw slides are only on disk while they're processed, so the 262 GB dataset needs a few GB of local disk (features: about 16 GB for 4.5M patches, ResNet50 + Phikon).
- **Resolution-independent patches:** every patch covers 128 µm whatever the scan resolution (256 px at 20×, 512 px at 40×), resized to 224 px on the GPU.
- **Resumable:** each step skips work whose output already exists.

## Running it

### On Google Colab (recommended for the heavy steps)

Open [`notebooks/lung_clam_colab.ipynb`](https://colab.research.google.com/github/amaansayydd/lung-clam/blob/main/notebooks/lung_clam_colab.ipynb) in Colab, select a **T4 GPU**, and choose **Run all**. It stores about 16 GB in Google Drive (more with UNI) and continues where it stopped if Colab disconnects.

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
uv run --group dev pytest                             # patient-leakage + metric tests
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
| `31_label_swap_check.py` | sensitivity check for the suspected label swap in the test set |
| `32_compare.py` | paired bootstrap tests: is model A really better than model B? |
| `40_shortcut_baseline.py` | colour-and-size-only classifiers on the same folds |
| `50_heatmaps.py` | attention heatmaps + top-patch galleries |
| `60_figures.py` | README figures |

## Results

All numbers below come from the prediction files committed under [`results/runs/`](results/runs) via [`scripts/30_evaluate.py`](scripts/30_evaluate.py) (full table: [`results/tables/metrics.md`](results/tables/metrics.md)). Brackets are 95% CIs from a patient-level bootstrap. Features: Phikon.

**In one sentence:** with all labels every MIL model is near-perfect on CPTAC, but the attention models (CLAM, ABMIL) need far fewer labels and are the ones that still work on another hospital's slides.

### Sealed internal test: 60 held-out CPTAC patients, 90 slides, scored once

The final model was fixed before the test set was opened: CLAM-SB on Phikon, 3 seeds × 5 folds, with the 15 models' probabilities averaged ([config](configs/experiments/final_clam_sb_phikon.yaml)).

| | Macro AUC | Balanced acc. | Tumour vs normal AUC | Tumour sensitivity | Normal specificity | LUAD vs LUSC AUC |
|---|---|---|---|---|---|---|
| **Test, pre-registered** | **0.980 (0.952–0.998)** | **0.900 (0.828–0.962)** | 0.987 (0.954–1.000) | 0.967 (0.917–1.000) | 0.967 (0.893–1.000) | 0.970 (0.931–0.997) |
| Same model, CV (360 slides) | 0.996 (0.990–0.999) | 0.969 (0.949–0.986) | 1.000 | 1.000 | 0.992 | 0.988 |

<p align="center"><img src="results/figures/test_final_clam_sb_phikon.png" width="720" alt="Sealed test confusion matrix and one-vs-rest ROC curves"></p>

The test score is a little below CV, as expected for a set that played no part in model choice. Part of the gap is also because CLAM's early stopping picks each fold's stopping epoch on the validation slides that CV is reported on, which makes CV slightly optimistic. Most errors are LUAD ↔ LUSC. The probabilities are well calibrated: expected calibration error is 0.040 on the test and 0.036 on TCGA ([reliability diagram](results/figures/calibration_final_clam_sb_phikon.png)). The least confident fifth of test slides is overconfident.

**A probable label swap.** For one test patient, C3L-03678, the model calls the "LUSC" slide normal (p = 0.86) and the "Normal" slide LUSC (p = 0.99). Looked at directly, the slide recorded as LUSC is airy lung tissue and the one recorded as Normal is solid tumour, so the two labels look swapped in the source metadata. This is not a pathologist's review. No other patient, in CV or test, shows this pattern. The headline above keeps the labels as recorded. With only this patient's two labels exchanged, macro AUC would be 0.990 and tumour-vs-normal AUC 1.000 ([`scripts/31_label_swap_check.py`](scripts/31_label_swap_check.py), output in [`results/tables/test_label_swap_check.txt`](results/tables/test_label_swap_check.txt)).

<p align="center"><img src="results/figures/label_swap_C3L-03678.jpg" width="640" alt="The two slides of patient C3L-03678: the one recorded as LUSC looks like normal lung, the one recorded as Normal looks like tumour"></p>

### Cross-validation: CPTAC, 360 slides, 5 patient-level folds

| Model | Macro AUC | Balanced acc. | Tumour sensitivity | Normal specificity | LUAD vs LUSC AUC |
|---|---|---|---|---|---|
| CLAM-SB | 0.993 (0.987–0.998) | 0.958 | 1.000 | 0.992 | 0.988 |
| CLAM-MB | 0.993 (0.987–0.997) | 0.953 | 1.000 | 0.992 | 0.981 |
| ABMIL (CLAM without instance clustering) | 0.995 (0.990–0.999) | 0.964 | 1.000 | 0.992 | 0.989 |
| Max-pool MIL | 0.982 (0.970–0.992) | 0.958 | 0.988 | 0.992 | 0.978 |
| Mean-pool | 0.992 (0.986–0.997) | 0.939 | 0.988 | 0.992 | 0.980 |
| *Shortcut detector (colour + size only)* | *0.909 (0.886–0.931)* | *0.775* | *0.950* | *0.850* | *0.843* |

### External test: TCGA, 120 slides from other hospitals, never used for training

TCGA has no matched normal slides here, so the external test is tumour detection and LUAD vs LUSC. Each slide is scored by the 5 fold models.

| Model | LUAD vs LUSC AUC | LUAD vs LUSC balanced acc. | Tumour sensitivity |
|---|---|---|---|
| CLAM-SB | 0.981 (0.958–0.997) | 0.883 | 0.975 (0.942–1.000) |
| CLAM-MB | 0.982 (0.962–0.996) | 0.917 | 0.967 (0.933–0.992) |
| ABMIL | 0.983 (0.962–0.997) | 0.933 | 0.983 (0.958–1.000) |
| Max-pool MIL | 0.983 (0.963–0.996) | 0.925 | 0.950 (0.908–0.983) |
| Mean-pool | 0.951 (0.914–0.980) | 0.850 | **0.800 (0.725–0.867)** |
| *Shortcut detector* | *0.627 (0.515–0.725)* | *0.508* | *—* |

### Label efficiency

<p align="center"><img src="results/figures/label_efficiency.png" width="640" alt="Macro AUC against the share of training labels for CLAM-SB, ABMIL and mean-pool"></p>

| LUAD vs LUSC AUC | 10% labels | 25% | 50% | 100% |
|---|---|---|---|---|
| CLAM-SB, CV | 0.912 | 0.965 | 0.972 | 0.988 |
| Mean-pool, CV | 0.848 | 0.921 | 0.954 | 0.980 |
| CLAM-SB, TCGA | 0.936 | 0.976 | 0.984 | 0.981 |
| Mean-pool, TCGA | 0.892 | 0.936 | 0.941 | 0.951 |

### Are the differences real?

The CIs above overlap a lot, but both models are scored on the same slides, so overlap doesn't settle the question. [`scripts/32_compare.py`](scripts/32_compare.py) bootstraps patients once per draw and scores both models on that draw, which gives a 95% CI for the *difference* ([full table](results/tables/comparisons.md)).

| CLAM-SB minus … | Labels | CV, LUAD vs LUSC AUC | TCGA, LUAD vs LUSC AUC | TCGA, tumour sensitivity |
|---|---|---|---|---|
| Mean-pool | 10% | **+0.064 (+0.042 to +0.090)** | **+0.044 (+0.001 to +0.096)** | **+0.250 (+0.175 to +0.325)** |
| Mean-pool | 25% | **+0.044 (+0.021 to +0.071)** | **+0.039 (+0.010 to +0.081)** | **+0.200 (+0.117 to +0.283)** |
| Mean-pool | 50% | **+0.018 (+0.005 to +0.033)** | **+0.043 (+0.013 to +0.083)** | **+0.217 (+0.142 to +0.292)** |
| Mean-pool | 100% | +0.008 (−0.003 to +0.020) | **+0.030 (+0.010 to +0.055)** | **+0.175 (+0.100 to +0.250)** |
| ABMIL | 100% | −0.001 (−0.007 to +0.004) | −0.002 (−0.011 to +0.004) | −0.008 (−0.025 to 0.000) |

Bold: the interval excludes 0. The intervals are not corrected for multiple comparisons.

### What this shows

1. **Attention matters for label efficiency and for transfer, not for in-distribution accuracy.** With all labels, mean-pooling Phikon features is within 0.001 macro AUC of CLAM-SB. With 10% of the labels CLAM leads by 0.06 LUAD-vs-LUSC AUC, and on TCGA mean-pool calls 1 in 5 tumours normal (sensitivity 0.80 vs 0.975).
2. **CLAM's instance clustering adds nothing measurable here.** ABMIL, which is CLAM-SB with clustering switched off, matches it within ±0.01 AUC at every label fraction, in CV and on TCGA, and none of the paired differences are clearly away from 0. With a strong pathology encoder the extra supervision seems redundant.
3. **The CPTAC data does carry a colour/size signal**, which reaches 0.91 macro AUC in CV. It does not transfer: on TCGA it falls to 0.63, while the MIL models stay at 0.98, so they are not mainly relying on it.
4. **Tumour vs normal is easy; LUAD vs LUSC is where models differ.** Every model separates tumour from normal at AUC ≥ 0.995 in CV.

### Attention heatmaps

<p align="center"><img src="results/heatmaps/clam_sb_phikon/C3N-05079-21_heatmap.jpg" width="820" alt="CPTAC LUSC slide with CLAM-SB attention overlay"></p>

CLAM-SB attention on a CPTAC LUSC slide, scored by the fold model that never saw it (red = high attention). More, including every misclassified slide: [`results/heatmaps/clam_sb_phikon/`](results/heatmaps/clam_sb_phikon) (index in `index.csv`). Heatmaps for the sealed test, including both slides of C3L-03678, are in [`results/heatmaps/final_clam_sb_phikon/`](results/heatmaps/final_clam_sb_phikon).

### Still to come

- ResNet50 (ImageNet) vs Phikon on the same splits, and UNI2-h features.

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
