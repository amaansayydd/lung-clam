# Lung Cancer WSI Classification with CLAM: Roadmap

**Goal:** a reproducible, honest, portfolio-grade project that uses CLAM for weakly supervised lung cancer classification on whole-slide images (WSIs).
**Task:** 3-class slide classification: **Normal / LUAD (adenocarcinoma) / LUSC (squamous cell carcinoma)**.
**Window:** 2026-10-23 → 2026-11-20 (4 weeks, 20+ h/week, overnight runs OK).
**Deliverable:** a polished GitHub repo with README, results tables, heatmap figures and one-command reproducibility.

> ⚠️ Research and education only. Not a medical device, and not for clinical use.

---

## 0. Decisions so far

| Topic | Decision |
|---|---|
| Task | 3-class Normal / LUAD / LUSC |
| Purpose | Portfolio and learning, so understanding plus polish matter more than state-of-the-art numbers |
| Compute | Free by default: M3 Mac (16 GB RAM, MPS). **Colab Pro is optional** and would be used only for the large encoder |
| Storage | 2 spare external drives (100 GB+ total), plus about 14 GB free internally, plus Colab VM disk if Pro |
| Pipeline | Run the full pipeline myself (download → segment → patch → features → train), on a **subset** of slides |
| Encoders | Open models first; UNI/CONCH only if Hugging Face access is approved |
| Expert | A pathology contact is available for a heatmap review |
| Experience | Intermediate (has trained CNNs; new to pathology and MIL) |

---

## 1. What makes this project stand out

Most CLAM projects run the tutorial on TCGA and report one AUC. This one aims to beat that in **six specific ways**:

1. **Confound-aware dataset design.** All three classes come from the **same cohort and the same tissue preparation**, so the model cannot pass "normal vs tumor" by learning "frozen vs FFPE" or "hospital A vs B". (See §3. This is the most common hidden flaw in public lung WSI projects.)
2. **External validation** on a separate cohort (TCGA) the model never saw.
3. **An encoder comparison:** ImageNet ResNet50 vs a pathology foundation model (and UNI if approved), all on identical splits.
4. **Ablations that explain CLAM.** Compare CLAM-SB, CLAM-MB, ABMIL (CLAM without the clustering loss), mean pooling and max pooling, so we can show what each piece of CLAM adds.
5. **Data-efficiency curves:** performance at 10%, 25%, 50% and 100% of the training labels. This was a headline claim of the CLAM paper; we reproduce it.
6. **Interpretability that is checked, not just shown:**
   - attention heatmaps plus a gallery of top-attended patches;
   - a "shortcut detector" baseline (§6.4);
   - a short structured review by a pathologist.

---

## 2. Architecture of the pipeline

```
 IDC / GDC (cloud)          Mac M3 (or Colab)                          Mac M3
┌────────────────┐   ┌──────────────────────────────────────┐   ┌──────────────────────┐
│ manifest.csv   │──▶│ stream batch of ~20 slides            │──▶│ splits (patient-level)│
│ (slide, patient│   │  ├─ tissue segmentation (CLAM)        │   │ CLAM / baselines train│
│  label, site,  │   │  ├─ patch coords @ ~20× (h5)          │   │ eval + bootstrap CIs  │
│  prep, mpp)    │   │  ├─ encoder features (fp16 .pt)       │   │ heatmaps + galleries  │
└────────────────┘   │  ├─ QC thumbnails                     │   └──────────────────────┘
                     │  └─ delete raw slide ✂                │
                     └──────────────────────────────────────┘
```

**Key idea:** raw slides are **streamed and deleted**. Only patch coordinates, features and thumbnails are kept, which is about 10–25 MB per slide per encoder instead of about 0.5 GB per slide.

---

## 3. Data plan

### 3.1 Sources

- **Primary (train / val / internal test): CPTAC-LUAD + CPTAC-LSCC**
  - Hosted on NCI Imaging Data Commons (IDC) as DICOM slide-microscopy images.
  - Older TCIA releases offered SVS files.
  - The cohorts have about 244 and 212 subjects, and include tumor **and normal-adjacent** tissue slides.
  - Download is free via `idc-index` + `s5cmd` from public buckets.
- **External test: TCGA-LUAD + TCGA-LUSC diagnostic (DX, FFPE) slides**, from IDC or the GDC portal.
  - TCGA DX slides are all tumor, so the external test checks **LUAD vs LUSC generalization**, plus how often the model wrongly calls a tumor slide "normal".
  - TCGA normals are mostly frozen sections. They are **excluded** from the main results and used only as a clearly labeled stress test (optional).

### 3.2 Verification checks (Phase 0, before any compute)

- [ ] Confirm the CPTAC normal slides exist and that their **preparation type matches** the tumor slides. Note: the README of the CLAM repo ships a CPTAC lung heatmap demo, which is a good sign.
- [ ] Find the IDC `sm_index` columns that encode tissue type (tumor/normal), anatomic site, pixel spacing (MPP) and patient ID.
- [ ] Confirm `openslide` 4.x opens IDC DICOM slides (by pointing at one `.dcm` file of a series). If it doesn't, the fallbacks are `wsidicom`, TCIA SVS files, or converting to pyramidal TIFF.
- [ ] Record each slide's scan magnification and MPP. CPTAC and TCGA mix 20× and 40× scans.

**Decision rule:** if CPTAC normals turn out to be prepared differently from the CPTAC tumors, stop and re-plan. The fallback is to train on CPTAC tumors only for subtyping, and treat "normal" as a separate, explicitly caveated experiment.

### 3.3 Subset size (fits 4 weeks on a Mac)

| Split | LUAD | LUSC | Normal | Slides |
|---|---|---|---|---|
| CPTAC (5-fold CV + held-out test) | ~150 | ~150 | ~150 | ~450 |
| TCGA external test | ~60 | ~60 | — | ~120 |
| **Total** | | | | **~570** |

- **Sampling:** at most 2 slides per patient per class; stratify by site; seed fixed and recorded.
- **Scale up later:** the pipeline is resumable, so scaling up is just "add rows to the manifest".

### 3.4 Manifest (single source of truth, committed to git)

`manifests/slides.csv` has these columns:

```
slide_id, patient_id, cohort, label, site, prep_type, mpp, magnification, source_uri, sha256, status
```

`status` tracks progress: `todo → downloaded → patched → feat_resnet → feat_phikon → … → done | failed:<reason>`.

### 3.5 Splitting rules (non-negotiable)

- **Patient-level splits:** every slide from a patient, tumor and normal alike, goes into the same fold.
- **Held-out internal test set** (about 20% of patients). It is frozen on day 1 of Week 2 and **looked at once**, at the end.
- 5-fold cross-validation on the remainder, used for model selection.
- Splits are generated once, saved to `splits/`, and reused for every experiment so all results are comparable.

---

## 4. Compute and storage plan

### 4.1 Mac M3 (default path, free)

- **Environment:**
  - Python **3.10** via `uv` (the system Python 3.14 is too new for this stack).
  - `torch` with MPS, plus `PYTORCH_ENABLE_MPS_FALLBACK=1`.
  - `openslide-python` with `openslide-bin` (or `brew install openslide`), plus `h5py`, `timm`, `scikit-learn`, `pandas`.
- **CLAM changes for Apple Silicon:** a small patch so the device is chosen as `cuda → mps → cpu`; check that nothing hard-codes `.cuda()`.
- **Storage layout:**
  - Internal SSD: the current batch of raw slides (≤ 10 GB at a time) for fast reading.
  - External drive A: patch h5 files, features and QC thumbnails.
  - External drive B: a backup of features and results.
- **Saving space:**
  - Store features as **fp16** `.pt`.
  - Drop the duplicate features that CLAM writes into its h5 files; keep only the coordinates.
  - Estimate: about 570 slides × ~15 MB ≈ **9 GB per 1024-dim encoder**.
- **Throughput (rough; measured for real in the pilot):**
  - ResNet50-trunc: ~5–10 h for the whole subset.
  - ViT-B encoder: ~12–20 h.
  - Both run overnight with `caffeinate -i`.

### 4.2 Colab Pro (optional accelerator)

- IDC data sits in Google Cloud buckets, so downloads inside Colab are **fast and free**.
- Use Colab for the **ViT-L encoder** (UNI, or Phikon-v2 as the open fallback), which is too slow on the Mac.
- Pattern:
  1. download a batch to the VM disk;
  2. patch and extract;
  3. copy the fp16 features to Drive (watch the 15 GB free quota), or zip them and pull them to the external drive;
  4. wipe the batch.
- Training stays on the Mac. CLAM training on features is light: minutes per fold.

**Decide by the end of Week 1** whether to buy Pro. Without it, the encoder comparison is ResNet50 vs Phikon (ViT-B), which is still a strong story.

---

## 5. Encoders

| ID | Model | Dims | Access | Where |
|---|---|---|---|---|
| E1 | ResNet50-trunc (ImageNet) | 1024 | Open, built into CLAM | Mac |
| E2 | Phikon (ViT-B, pathology self-supervised) | 768 | Open on Hugging Face (verify license) | Mac |
| E3 | UNI / UNI2-h (or CONCH) | 1024 / 1536 / 512 | **Gated** (requires your HF access request) | Colab |
| E3-alt | Phikon-v2 (ViT-L) | 1024 | Open | Colab |

- CLAM's `extract_features_fp.py` supports `resnet50_trunc`, `uni_v1` and `conch_v1` out of the box.
- Phikon needs about 20 lines added to the model builder.
- The Mahmood Lab's **TRIDENT** repo also supports 25+ encoders. We'll check in Phase 0 whether it saves effort (it may expect CUDA).

**Patch settings:**
- Tissue at an effective **20×** (~0.5 µm/px).
- On 40× slides, use 512 px patches at level 0 and resize them to 224.
- On 20× slides, use 256 px patches and resize them to 224.
- Patch size is driven by MPP, not by assuming every slide is 40×.

---

## 6. Experiments

**Priority labels:**
- **M** = must have; the project is incomplete without it.
- **S** = should have.
- **C** = could have; cut first if time runs short.

### 6.1 Core results (M)
- CLAM-SB on the best encoder: 5-fold patient-level CV, then the internal test once, then the TCGA external test.
- **Metrics:**
  - Macro one-vs-rest AUC and balanced accuracy.
  - Per-class sensitivity and specificity, plus a confusion matrix.
  - **Tumor-vs-normal sensitivity** called out separately, because a missed cancer is the costliest error.
  - 95% CIs by bootstrapping over **patients** (1,000 resamples).
- **External test reporting:**
  - LUAD-vs-LUSC AUC on TCGA tumor slides;
  - the fraction of TCGA tumor slides predicted "normal".

### 6.2 Encoder comparison (S, M if UNI is approved)
- E1 vs E2 (vs E3), with the same splits, same hyperparameters and same seeds.
- Present it as one table and one bar chart with CIs.

### 6.3 Model ablations (S)

| Model | How |
|---|---|
| CLAM-SB | `--model_type clam_sb --inst_loss svm --B 8 --bag_weight 0.7` |
| CLAM-MB | `--model_type clam_mb` (one attention branch per class) |
| ABMIL | CLAM-SB with `--no_inst_cluster` (isolates the effect of the clustering loss) |
| Max-pooling MIL | `--model_type mil` |
| Mean-pooling | small custom head on averaged features (a ~30-line script) |
| TransMIL | (C) only if time allows |

Light hyperparameter sweep on CV for the best model only:
- `B` ∈ {8, 16}
- `bag_weight` ∈ {0.5, 0.7, 0.9}
- `drop_out` 0.25

### 6.4 Shortcut detector (M; cheap and very convincing)
- Train a logistic regression on **trivial features**: mean RGB/HSV of the tissue, tissue area, MPP and scanner metadata.
- If it reaches a high AUC, the dataset leaks label information through non-biological signals. Report it either way.
- Also report CLAM performance **per CPTAC site**.

### 6.5 Data efficiency (S)
- `label_frac` ∈ {0.1, 0.25, 0.5, 1.0}, using CLAM's `create_splits_seq.py --label_frac`.
- Run for CLAM-SB vs ABMIL vs mean pooling on the best encoder.
- The result is a learning-curve figure with CIs.

### 6.6 Calibration (C)
- Reliability diagram plus expected calibration error, before and after temperature scaling (fitted on the validation folds).

**Training defaults** (from the CLAM paper; adjust via CV only):
- Adam, `lr 2e-4`, `reg 1e-5`
- Up to 200 epochs with early stopping (patience 20)
- `--weighted_sample` and `--drop_out 0.25`
- 3 seeds for the final model

---

## 7. Interpretability

1. **Heatmaps** (`create_heatmaps.py`):
   - 12–20 slides: correct and incorrect predictions, each class, and both cohorts.
   - Overlapping patches (overlap 0.5–0.75) for smooth maps.
   - Exported as PNGs at a readable resolution.
2. **Top-k patch galleries:** the top 16 attention patches per slide, plus a per-class montage across slides, showing "what the model thinks LUSC looks like".
3. **Pathologist review** (about 45 min of their time, booked for Week 4):
   - 20 slides, with heatmaps shown next to the plain H&E, ordered randomly.
   - For each slide they score 3 things:
     - (a) Does high attention fall on diagnostic tissue? 0 = no, 1 = partly, 2 = yes.
     - (b) Is attention on artifacts (pen, fold, blur, background)? yes/no.
     - (c) Their own diagnosis.
   - Report the agreement summary in the README.
4. **Error analysis:** for every misclassified test slide, record one line: the likely cause (artifact, ambiguous histology, label noise, mixed tumor, low tissue area).

---

## 8. Week-by-week plan

### Week 1 (Oct 23–29): Setup and a thin end-to-end slice

The goal of Week 1 is **one ugly but complete run** before anything is scaled up.

| Day | Tasks |
|---|---|
| Fri 23 | Repo skeleton, `uv` env (Python 3.10), CLAM as a pinned **git submodule**, MPS device patch. **You:** request UNI/CONCH access on HF *today*, because approval can take days. |
| Sat 24 | `idc-index` exploration notebook: list CPTAC and TCGA slides, check the tissue-type, MPP and prep columns. Run all the verification checks in §3.2. |
| Sun 25 | Build `manifests/slides.csv` with sampling, then a download + open test on 3 slides (DICOM via openslide). |
| Mon 26 | Segmentation and patching on 10 slides. Tune segmentation (`sthresh`, `mthresh`, `use_otsu`, `exclude_ids` for pen marks) by inspecting thumbnails. Save a custom preset in `configs/`. |
| Tue 27 | Feature extraction with E1 on 30 pilot slides (10 per class). **Measure patches/sec and MB/slide**, then update the timing and storage estimates. |
| Wed 28 | Train CLAM-SB on the 30 pilot slides (expected to be meaningless, but it proves the plumbing). Generate 1 heatmap. |
| Thu 29 | Write `scripts/run_stream.py`, the resumable streaming loop (download → patch → extract → verify → delete). **Gate G1 ✅. Decision: buy Colab Pro or not.** |

### Week 2 (Oct 30–Nov 5): Data at scale and first real results

| Day | Tasks |
|---|---|
| Fri 30 | Freeze the patient-level splits (5 folds + held-out test + label fractions). Start the overnight E1 stream over all ~570 slides. |
| Sat–Sun | Monitor the stream: run QC thumbnails, and log failed slides in the manifest. In parallel, build the evaluation module (metrics, bootstrap CIs, confusion matrix, plots). |
| Mon 2 | E1 done, so run the first full CLAM-SB 5-fold CV. Start the E2 (Phikon) stream overnight. |
| Tue 3 | Implement the mean-pooling baseline and the shortcut-detector baseline (§6.4). |
| Wed 4 | E3 on Colab, if Pro was bought and UNI approved; otherwise Phikon-v2 on Colab, or skip E3. |
| Thu 5 | **Gate G2 ✅:** all features extracted, first CV numbers, shortcut check done. |

### Week 3 (Nov 6–12): The experiment matrix

| Day | Tasks |
|---|---|
| Fri 6 | Encoder comparison: CLAM-SB × E1, E2 (and E3). |
| Sat–Sun | Model ablations (§6.3) on the best encoder, run overnight in batch. Then a light hyperparameter sweep. |
| Mon 9 | Data-efficiency runs (§6.5). |
| Tue 10 | Pick the final configuration and train 3 seeds. **Unseal the internal test.** Run the TCGA external test. |
| Wed 11 | Calibration (C), per-site analysis, error analysis table. |
| Thu 12 | **Gate G3 ✅:** all numbers final and every figure generated by script. Send the pathologist review pack. |

### Week 4 (Nov 13–19): Interpretability, polish, release

| Day | Tasks |
|---|---|
| Fri 13 | Production heatmaps and top-k galleries. |
| Sat–Sun | README: problem, data design (why same-cohort normals), pipeline diagram, results tables, figures, heatmaps, limitations, reproduction commands. |
| Mon 16 | Pathologist review session, then write up the findings. |
| Tue 17 | Reproducibility pass: fresh clone, follow the README on 5 slides end to end, fix whatever breaks. |
| Wed 18 | Model card and data card (`docs/`), license check (CLAM is GPL-3.0, so this repo is GPL-3.0 too), and citations. |
| Thu 19 | **Gate G4 ✅:** tag `v1.0`, make the repo public. |
| Fri 20 | Buffer day. |

### Cut list (if behind, drop in this order)
TransMIL → calibration → E3 → hyperparameter sweep → 3 seeds (use 1) → data efficiency for baselines (keep CLAM only) → reduce the TCGA external test to 60 slides.

**Never cut:** patient-level splits, the shortcut check, the external test, or heatmaps.

---

## 9. Repository layout

```
lung-clam/
├── README.md               # the portfolio front page
├── ROADMAP.md              # this file
├── LICENSE                 # GPL-3.0 (inherited from CLAM)
├── pyproject.toml / uv.lock
├── CLAM/                   # git submodule, pinned commit + patches/
├── patches/                # our small diffs to CLAM (MPS, Phikon, fp16)
├── configs/
│   ├── seg_preset_cptac.csv
│   ├── experiments/*.yaml  # one file per experiment
│   └── heatmaps/*.yaml
├── manifests/slides.csv    # committed; the single source of truth
├── splits/                 # committed; frozen patient-level splits
├── scripts/
│   ├── 00_build_manifest.py
│   ├── run_stream.py       # download → patch → extract → verify → delete
│   ├── 10_make_splits.py
│   ├── 20_train.py         # wraps CLAM main.py from a yaml config
│   ├── 30_evaluate.py      # metrics, bootstrap CIs, tables
│   ├── 40_shortcut_baseline.py
│   ├── 50_heatmaps.py
│   └── 60_figures.py
├── notebooks/              # exploration only, never required to reproduce
├── results/                # small committed tables + figures
├── docs/ model_card.md, data_card.md, pathologist_review.md
└── data/  (gitignored → symlink to external drive)
```

**Engineering rules:**
- Every number in the README comes from a script, not a notebook.
- Every run logs its git commit, config and seed.
- Get it working end to end first (Week 1); add tests only for what has bitten us: split leakage and manifest integrity.

---

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| CPTAC normals prepared differently from tumors | Medium | Checked in §3.2 on day 2; fallback design in §3.2 |
| openslide can't read IDC DICOM | Low–Med | `wsidicom`, TCIA SVS, or conversion |
| MPS op unsupported or slow | Medium | `PYTORCH_ENABLE_MPS_FALLBACK=1`, CPU for training, Colab for the encoder |
| UNI access denied or slow | Medium | Phikon / Phikon-v2 (open) |
| Disk fills mid-stream | Medium | Batches of 20, fp16 features, stream deletes raw slides, free-space guard in `run_stream.py` |
| External HDD is slow | Medium | Read raw slides from the internal SSD; only write features to external |
| Mac sleeps overnight | Low | `caffeinate -i`, resumable stream |
| Results look too good | Medium | Shortcut detector, per-site metrics, external test, pathologist review |
| Time overrun | Medium | Gates every Thursday; cut list in §8 |

---

## 11. Definition of done (v1.0)

- [ ] Fresh clone → README commands reproduce the pipeline on a 5-slide sample.
- [ ] Results table: macro AUC and balanced accuracy with 95% CIs (CV, internal test, TCGA external).
- [ ] Encoder comparison and model ablation tables.
- [ ] Shortcut-detector result reported honestly.
- [ ] Data-efficiency figure.
- [ ] ≥ 12 heatmaps and per-class top-patch montages, plus a pathologist review summary.
- [ ] Error analysis, limitations section, model card, data card, citations (CLAM, encoders, CPTAC, TCGA, IDC).

---

## 12. Action items for you (before or during Week 1)

1. **Today:** request access to **UNI** (and optionally **CONCH**) on Hugging Face, using an institutional email if you have one.
2. **By Thu Oct 29:** decide on Colab Pro, based on the pilot timings.
3. **Format the external drives by Mon Oct 26** (formatting erases them, so back up anything on them first).
   - Use **APFS** if the drive is an SSD or flash drive.
   - Use **Mac OS Extended (Journaled)** if it's a spinning hard disk.
   - Use **exFAT** only if you also need it on Windows.
   - Name them `CLAM_DATA` and `CLAM_BACKUP`.
4. **Message your pathology contact now** to book about 45 minutes around **Mon Nov 16**.
5. **GitHub:** the repo is `lung-clam`. It stays private until v1.0.

## 13. Remaining open questions

- [x] Repo name: **`lung-clam`**.
- [ ] External drive names and formats (for the `data/` symlink). Needed by **Mon Oct 26**.
- [ ] Should the pathologist be credited by name in the README, or kept anonymous?
- [x] Post-v1.0: see §14.

### Fallback if no pathologist is available
The review in §7.3 is optional. Without it:
- Score the heatmaps ourselves against published morphology criteria:
  - LUAD: glandular / acinar / lepidic / papillary growth, mucin.
  - LUSC: keratinization, keratin pearls, intercellular bridges, solid nests.
  - Normal: alveoli, bronchi, vessels.
- Run an artifact audit: what fraction of top-attended patches are pen marks, folds, blur or background.
- Count how often top-attended patches on tumor slides fall in tumor versus stroma or normal tissue.
- State plainly in the README that no expert review was done.

---

## 14. After v1.0: stretch phase (in priority order)

1. **Interactive demo**
   - A Gradio app (Hugging Face Space or local) that takes an uploaded slide, or a pick from the sample slides.
   - It returns a prediction with probabilities, a pan-and-zoom attention heatmap, and the top patches.
   - This is the highest-impact addition for a portfolio.
2. **Scale to the full CPTAC + TCGA data**
   - The same pipeline with more rows in the manifest; check whether the results hold at scale.
3. **Train on TCGA and test on CPTAC** (the reverse direction), to show the generalization result holds both ways.
4. **Mutation prediction (EGFR / KRAS / STK11) on TCGA-LUAD**
   - Reuses the extracted features, so it only needs new labels.
   - A research-flavored extension.

---

### References
- Lu et al., *Data-efficient and weakly supervised computational pathology on whole-slide images*, Nature BME 2021 (CLAM): https://github.com/mahmoodlab/CLAM
- NCI Imaging Data Commons collections: [CPTAC-LUAD](https://portal.imaging.datacommons.cancer.gov/collections/cptac_luad), [CPTAC-LSCC](https://portal.imaging.datacommons.cancer.gov/collections/cptac_lscc)
- TCIA CPTAC-LSCC collection wiki: https://wiki.cancerimagingarchive.net/x/TwiwC
