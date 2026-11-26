# Data card

Everything about the slides used in this project. Raw data stays with its sources and is not redistributed. The repository holds only the slide list, the splits, and per-slide predictions.

## Sources

| Collection | Where from | Format | Licence / terms |
|---|---|---|---|
| CPTAC-LUAD, CPTAC-LSCC | [NCI Imaging Data Commons](https://portal.imaging.datacommons.cancer.gov/) (v25) | DICOM WSI, baseline JPEG, 20× (0.49 µm/px) | CC BY 3.0 (TCIA/IDC) |
| TCGA-LUAD, TCGA-LUSC (diagnostic FFPE) | [GDC](https://portal.gdc.cancer.gov/), original Aperio `.svs`, md5-checked | SVS, mostly 40× (0.25 µm/px) | NIH GDC open-access terms |

IDC's TCGA copies use JPEG 2000, which OpenSlide cannot decode, so TCGA comes from the GDC ([details](data_findings.md#2026-10-24-tcga-slides-come-from-the-gdc-not-idc)).

## Composition

Fixed in [`manifests/slides.csv`](../manifests/slides.csv) by [`scripts/01_build_manifest.py`](../scripts/01_build_manifest.py) with seed 42. There is one slide per patient per class.

| Split | Normal | LUAD | LUSC | Slides | Patients | Use |
|---|---|---|---|---|---|---|
| CPTAC cross-validation (5 patient-level folds) | 120 | 120 | 120 | 360 | 240 | model development |
| CPTAC sealed internal test | 30 | 30 | 30 | 90 | 60 | scored once, by the pre-registered final model |
| TCGA external test | — | 60 | 60 | 120 | 120 (68 hospitals) | transfer to other hospitals |

- **Normal slides are normal-adjacent tissue from the same CPTAC patients as the tumour slides.** Half come from LUAD patients and half from LSCC patients. Tumour and normal slides therefore share the preparation, the scanner and the patient. A classic shortcut is avoided this way: TCGA normals are all frozen sections, while TCGA tumours are FFPE.
- **Splits are by patient.** No patient appears in two splits. The leakage tests in [`tests/test_splits.py`](../tests/test_splits.py) check this.
- **Label-fraction subsets** (10/25/50%) are nested and stratified, and are drawn from the training patients of each fold only.
- **Verified labels:** 215 of the 300 CPTAC tumour patients have a clinical diagnosis on record that matches their cohort (`label_verified`). The `cv_verified` rows in the metrics table use only these.

## Processing

- Tissue segmentation uses CLAM.
- Every patch covers 128 µm whatever the scan resolution: 256 px at 20×, 512 px at 40×.
- Features are stored in fp16.
- Per slide: median 5,278 patches, mean 7,863, maximum 43,924; 4.48 million patches in total.
- A QC thumbnail and colour/size statistics are saved for every slide. The statistics feed the shortcut detector.

## Known issues

- **Probable label swap:** CPTAC patient C3L-03678. Its tumour (-21) and normal (-22) slides look exchanged. This is a non-expert visual check, to be confirmed by a pathologist.
- **Normal-adjacent slides can contain tumour.** C3L-02168-25 is flagged for review.
- **Colour and size differ by class in CPTAC:** normal lung is paler. Some of this is biology and some may be stain batch. Colour/size alone reaches 0.91 macro AUC in CV, but it does not transfer to TCGA (0.63).
- **CPTAC metadata has no hospital or site field**, so site-stratified analysis is only possible on TCGA.
- **Some slides hold several serial sections** of the same tissue, which produces near-duplicate patches.

A dated log of everything found while building the dataset is in [`data_findings.md`](data_findings.md).
