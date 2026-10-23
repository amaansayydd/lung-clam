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
