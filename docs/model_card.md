# Model card: lung-clam final model (CLAM-SB on Phikon features)

## Summary

| | |
|---|---|
| Task | Slide-level 3-class classification of lung H&E whole-slide images: **Normal / LUAD / LUSC** |
| Architecture | [CLAM-SB](https://github.com/mahmoodlab/CLAM) (gated attention MIL with instance clustering, `subtyping=False`, small size, 661k parameters) on frozen [Phikon](https://huggingface.co/owkin/phikon) patch features (ViT-B/16, 768-dim) |
| Ensemble | 15 models (3 seeds × 5 patient-level folds); slide probabilities are averaged |
| Input | All tissue patches of a slide, each covering 128 µm (256 px at 20×, 512 px at 40×), resized to 224 px |
| Output | Probabilities for Normal, LUAD and LUSC, plus an attention score per patch |
| Config | [`configs/experiments/final_clam_sb_phikon.yaml`](../configs/experiments/final_clam_sb_phikon.yaml); training defaults in [`scripts/20_train.py`](../scripts/20_train.py) (Adam, lr 2e-4, weight decay 1e-5, dropout 0.25, bag weight 0.7, B = 8, SVM instance loss, early stopping, class-balanced sampling) |
| Status | Research and education only. **Not a medical device. Not for clinical use.** |

## Intended use

- Learning and benchmarking weakly supervised pathology methods. It shows how label efficiency, transfer to other hospitals, and shortcut checks behave.
- **Out of scope:**
  - diagnosis, triage or any decision about a patient;
  - frozen sections, cytology, biopsies from other organs;
  - tumour types other than LUAD and LUSC (small-cell, large-cell, carcinoid and metastases will be forced into one of the three classes).

## Training data

- 360 CPTAC slides from 240 patients (CPTAC-LUAD, CPTAC-LSCC via the NCI Imaging Data Commons):
  - 120 LUAD, 120 LUSC, 120 normal-adjacent;
  - the normal slides come from the same patients as the tumour slides;
  - all FFPE, 20×.
- See the [data card](data_card.md) for how the slides were chosen and what is known about them.

## Evaluation

All numbers come from committed prediction files; brackets are 95% patient-level bootstrap CIs.

| Set | Slides | Macro AUC | Balanced acc. | Tumour sensitivity | LUAD vs LUSC AUC |
|---|---|---|---|---|---|
| Sealed internal test (CPTAC, 60 patients, scored once) | 90 | 0.980 (0.952–0.998) | 0.900 (0.828–0.962) | 0.967 (0.917–1.000) | 0.970 (0.931–0.997) |
| External test (TCGA FFPE, 68 hospitals, no normals) | 120 | — | — | 0.975 (0.942–1.000) | 0.979 (0.954–0.995) |
| Cross-validation (out-of-fold, CPTAC) | 360 | 0.996 (0.990–0.999) | 0.969 (0.949–0.986) | 1.000 | 0.988 (0.973–0.998) |

**Calibration.** The expected calibration error is 0.012 in CV, 0.040 on the sealed test and 0.036 on TCGA (LUAD vs LUSC), from [`results/figures/calibration_final_clam_sb_phikon.png`](../results/figures/calibration_final_clam_sb_phikon.png). Most slides are predicted with confidence above 0.95, and those predictions are almost always right. The least confident fifth of test slides is overconfident: on average it claims about 0.75 and is right on 61% of them.

## Limitations and known issues

- **One probable label swap in the test set.**
  - The source labels for patient C3L-03678 look reversed: the tumour and normal slides appear exchanged.
  - The headline test result keeps the recorded labels. With the swap corrected it would be 0.990 macro AUC.
  - This is a non-expert visual judgement and needs a pathologist to confirm.
- **CV numbers are slightly optimistic.**
  - CLAM's early stopping picks each fold's stopping epoch on that fold's validation slides, which are also the slides that CV is reported on.
  - The sealed test and TCGA numbers do not have this bias. Use them as the estimate of performance.
- **Normal-adjacent is not healthy lung.**
  - CPTAC normals are taken next to a tumour and can contain some tumour.
  - TCGA has no FFPE normals, so normal specificity was only measured on CPTAC.
- **LUAD vs LUSC causes most of the errors.**
  - On the sealed test, 6 of 60 tumours get the wrong subtype.
  - Poorly differentiated or mixed tumours are where the model is least reliable.
- **Colour and size carry signal in CPTAC.**
  - A classifier on colour/size statistics alone reaches 0.91 macro AUC in CV, but 0.63 LUAD-vs-LUSC AUC on TCGA.
  - The MIL model holds up on TCGA, so it is not mainly using that signal, but this does not prove it ignores it.
- **Narrow population.**
  - CPTAC has no site field, and TCGA covers mostly North American hospitals.
  - Performance on other scanners, stains or populations is unknown.
- **Attention is not an explanation.**
  - Heatmaps show where the model looked, not why it decided.
  - They have not yet been reviewed by a pathologist.

## Licences

- Code is GPL-3.0, inherited from CLAM.
- Phikon weights are under Owkin's licence; see its model page.
- CPTAC and TCGA data are subject to their own terms.
- The trained checkpoints were fitted on CPTAC data and are shared for research and education only.
