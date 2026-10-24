# Data findings

## 2026-10-23: IDC v25 lung slide inventory

Produced by `scripts/00_explore_idc.py`; metadata only.

| Cohort | Tissue | Embedding | Slides | Patients | GB | µm/px |
|---|---|---|---|---|---|---|
| CPTAC-LUAD | Tumor | not recorded | 681 | 229 | 295 | 0.49 (20×) |
| CPTAC-LUAD | **Normal** | not recorded | 379 | 225 | 134 | 0.49 (20×) |
| CPTAC-LSCC | Tumor | not recorded | 662 | 203 | 309 | 0.49 (20×) |
| CPTAC-LSCC | **Normal** | not recorded | 364 | 198 | 132 | 0.49 (20×) |
| TCGA-LUAD | Tumor (DX) | Paraffin / formalin (FFPE) | 540 | 478 | 413 | 0.25 (40×) |
| TCGA-LUAD | Tumor | Frozen (OCT) | 819 | 506 | 182 | 0.25 |
| TCGA-LUAD | Normal | **Frozen only** | 244 | 198 | 34 | 0.49 |
| TCGA-LUSC | Tumor (DX) | FFPE | 512 | 478 | 410 | 0.25 |
| TCGA-LUSC | Tumor | Frozen (OCT) | 753 | 488 | 163 | 0.50 |
| TCGA-LUSC | Normal | **Frozen only** | 347 | 233 | 37 | 0.50 |

(Plus 132 CPTAC and 5 TCGA slides with no tissue-type label; these are excluded.)

### What this means
- **CPTAC has plenty of normal slides:** 743 normal slides from about 420 patients. Most patients have both tumor and normal slides.
- CPTAC tumor and normal slides share the same resolution (0.49 µm/px). The DICOM metadata does **not** record embedding or fixation, so preparation must be checked **visually** in the pilot. Frozen sections show ice-crystal holes, torn tissue and paler, smudgier staining.
- **TCGA normals are 100% frozen**, while TCGA diagnostic tumor slides are FFPE. This confirms that TCGA normals must stay out of the main results.
- TCGA DX slides are 40× (0.25 µm/px) and CPTAC slides are 20× (0.49 µm/px). Patch size has to be MPP-driven: 512 px at 40× and 256 px at 20×, both covering about 128 µm.
- Average slide size is about 0.35–0.45 GB (CPTAC) and about 0.8 GB (TCGA DX). The ~570-slide subset means streaming about 280 GB in total.

## 2026-10-23: format and preparation check (3 CPTAC slides)

- **OpenSlide 4.0.1 opens IDC DICOM directly.** Point it at any `.dcm` file in the series.
  - Pyramids have 3–4 levels (1×, 4×, 16×, 32×), and MPP and objective (20×) are read correctly.
  - The first tile read on a fresh file takes about 2–3 s; after that, 500–860 tiles/s. The encoder (about 170 patches/s) is the bottleneck, not I/O.
- **Preparation looks like FFPE for tumor *and* normal.** Crops are in `results/pilot/*_20x_crops.jpg`.
  - Nuclear detail is crisp, with no ice-crystal vacuoles.
  - LSCC (C3L-05417) shows squamous nests; LUAD (C3N-00574) is mucinous adenocarcinoma; Normal (C3N-01416) shows alveolar parenchyma.
  - This is a non-expert check on 3 slides. Re-check on all 30 pilot slides, then have the pathologist confirm.
- **Things to watch:**
  - The normal slide is much more eosinophilic (redder) than the tumors. That could be biology or a stain batch effect, which the shortcut detector (ROADMAP §6.4) will test.
  - The normal slide's thumbnail shows a dense purple region at one edge. Normal-adjacent slides may contain some tumor (label noise).
  - Some slides hold several serial sections of the same tissue (C3L-05417 has 3), so near-duplicate patches are expected.
- **No hospital/site field in the CPTAC clinical data.** Only the TCGA site code is available, for the external test.
- **Clinical diagnosis exists for 111 LUAD and 108 LSCC CPTAC patients**, and all of them match their cohort. These patients are flagged `label_verified` and sampled first.

## 2026-10-23: manifest v1 (`scripts/01_build_manifest.py`, seed 42)

| Group | Label | Slides (= patients) | Verified | Pilot | GB |
|---|---|---|---|---|---|
| CPTAC | LUAD | 150 | 107 | 10 | 53 |
| CPTAC | LUSC | 150 | 108 | 10 | 53 |
| CPTAC | Normal (75 from LUAD patients + 75 from LSCC patients) | 150 | 150 | 10 | 41 |
| TCGA external | LUAD | 60 | — | — | 60 |
| TCGA external | LUSC | 60 | — | — | 55 |

- 570 slides from 420 patients, 262 GB to stream.
- **Every normal slide comes from a patient whose tumor slide is also in the set.**
- The external test spans 68 TCGA hospitals.
- Patch size is set per slide to a constant 128 µm: 256 px at 0.5 µm/px and 512 px at 0.25 µm/px (plus 4 slides at 560 px and 1 at 400 px).

## 2026-10-23: QC on the first 146 slides processed on Colab

- **Patches per slide:** median 3,654 and mean 5,460, up to 34,652 (C3L-00445-26: two large serial sections, 537 mm²). Features for the full set are about 12 GB (ResNet50 + Phikon, fp16).
- **Segmentation** looks correct on spot checks. On a few slides, thin outlines along the coverslip edge and the scanner's "+" fiducial marks are segmented too, but they add only a handful of edge patches.
- **Colour differs by class.** Median tissue saturation is 55 (Normal) vs 81 (LUAD) vs 85 (LUSC), and median red channel is 209 vs 181 vs 167.
  - Part of this is biology: normal lung is mostly air spaces, so it's paler.
  - Part may be stain batch, e.g. patient C3L-02168's normal and tumour slides are both very dark.
  - The shortcut detector (`scripts/40_shortcut_baseline.py`) measures how much of the label this alone gives away.
- **Flagged for the pathologist review:** C3L-02168-25 (normal, very dark, dense) as possible label noise.

## 2026-10-24: TCGA slides come from the GDC, not IDC

- **The problem:** every TCGA slide failed on Colab with `OpenSlideError: Reading ...`. IDC's DICOM copies of the TCGA diagnostic slides use **JPEG 2000** (transfer syntax 1.2.840.10008.1.2.4.91), which OpenSlide's DICOM reader cannot decode. CPTAC uses baseline JPEG (…1.2.4.50), which is why all 450 CPTAC slides worked.
- **The fix:** TCGA slides are now downloaded from the **GDC** as the original Aperio `.svs` files (open access, all 120 found, 107 GB) and checked against the GDC's md5. OpenSlide reads these natively. See `scripts/02_add_gdc_ids.py`.
- **Checked:** the level-0 dimensions match IDC's (e.g. TCGA-05-5420: 10000 × 8104), so the manifest's resolution and patch size stay valid. One slide went end to end: 816 patches, finite features.
- **CPTAC extraction is complete:** 450/450 slides with ResNet50 + Phikon features.
