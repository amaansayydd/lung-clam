"""Build manifests/slides.csv: the fixed list of slides used in this project.

Design (see ROADMAP.md §3):
  * CPTAC (train / val / internal test), 3 classes, one slide per patient per class:
      LUAD   = tumor slide from a CPTAC-LUAD patient
      LUSC   = tumor slide from a CPTAC-LSCC patient
      Normal = normal-adjacent slide, drawn from the SAME patients as the tumor slides
               (half LUAD patients, half LSCC patients), so patient identity, hospital
               and preparation cannot separate normal from tumor.
  * TCGA diagnostic FFPE slides (external test), LUAD vs LUSC only, one slide per patient,
    spread across tissue-source sites. TCGA normals are frozen sections and are excluded.
  * Patients with a clinical diagnosis on record (label_verified) are sampled first.
  * A small stratified pilot subset is flagged for the end-to-end dry run.

Metadata only; nothing is downloaded.  Run: uv run python scripts/01_build_manifest.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from idc_index import IDCClient

ROOT = Path(__file__).resolve().parents[1]
PATCH_UM = 128  # physical patch width in µm: 256 px at 20× (0.5 µm/px), 512 px at 40×

SLIDES_SQL = """
SELECT i.collection_id AS cohort,
       i.PatientID AS patient_id,
       sm.ContainerIdentifier AS slide_id,
       i.SeriesInstanceUID AS series_uid,
       sm.primaryAnatomicStructureModifier_CodeMeaning AS tissue,
       sm.embeddingMedium_CodeMeaning AS embedding,
       sm.min_PixelSpacing_2sf * 1000 AS mpp,
       sm.ObjectiveLensPower AS magnification,
       sm.max_TotalPixelMatrixColumns AS width,
       sm.max_TotalPixelMatrixRows AS height,
       i.series_size_MB AS size_mb
FROM sm_index sm JOIN index i USING (SeriesInstanceUID)
WHERE i.collection_id IN ('cptac_luad', 'cptac_lscc', 'tcga_luad', 'tcga_lusc')
"""


def load_slides(client: IDCClient) -> pd.DataFrame:
    client.fetch_index("sm_index")
    df = client.sql_query(SLIDES_SQL)
    # list-valued DICOM code fields → plain strings
    for col in ("tissue", "embedding"):
        df[col] = df[col].map(lambda v: ";".join(v) if isinstance(v, (list, np.ndarray)) else (v or ""))
    return df


def verified_patients(client: IDCClient) -> set[str]:
    """Patients whose clinical record states the histology that matches their cohort."""
    client.fetch_index("clinical_index")
    expected = {"cptac_luad": "Adenocarcinoma", "cptac_lscc": "Squamous cell carcinoma"}
    ok = set()
    for cohort, dx in expected.items():
        clin = client.get_clinical_table(f"{cohort}_clinical")
        ok |= set(clin.loc[clin["diag__primary_diagnosis"].str.startswith(dx), "dicom_patient_id"])
    return ok


def pick_patients(df: pd.DataFrame, n: int, rng: np.random.Generator) -> list[str]:
    """n patients, label-verified ones first, random within each group."""
    pats = df.drop_duplicates("patient_id")[["patient_id", "label_verified"]]
    pats = pats.assign(r=rng.random(len(pats))).sort_values(["label_verified", "r"], ascending=[False, True])
    if len(pats) < n:
        raise SystemExit(f"only {len(pats)} patients available, asked for {n}")
    return pats["patient_id"].head(n).tolist()


def one_slide_per_patient(df: pd.DataFrame, patients: list[str], rng: np.random.Generator) -> pd.DataFrame:
    sub = df[df.patient_id.isin(patients)].assign(r=lambda d: rng.random(len(d)))
    return sub.sort_values("r").drop_duplicates("patient_id").drop(columns="r")


def build(args) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    client = IDCClient()
    df = load_slides(client)
    df["label_verified"] = df.patient_id.isin(verified_patients(client))
    df["site"] = ""

    # ---------------- CPTAC: 3 classes from the same patients ----------------
    cptac = df[df.cohort.str.startswith("cptac")]
    tumor = cptac[cptac.tissue == "Neoplasm, Primary"]
    normal = cptac[cptac.tissue == "Normal"]
    has_normal = set(normal.patient_id)

    parts = []
    for cohort, label in (("cptac_luad", "LUAD"), ("cptac_lscc", "LUSC")):
        t = tumor[tumor.cohort == cohort]
        # prefer patients who also have a normal slide, so normals can come from the same people
        t = t.assign(label_verified=t.label_verified & t.patient_id.isin(has_normal)) \
            if args.prefer_paired else t
        pats = pick_patients(t, args.n_per_class, rng)
        parts.append(one_slide_per_patient(tumor[tumor.cohort == cohort], pats, rng).assign(label=label))

        n_norm = args.n_per_class // 2 + (args.n_per_class % 2 if cohort == "cptac_luad" else 0)
        norm_pool = normal[(normal.cohort == cohort) & normal.patient_id.isin(pats)]
        norm_pats = pick_patients(norm_pool, n_norm, rng)
        parts.append(one_slide_per_patient(norm_pool, norm_pats, rng).assign(label="Normal"))

    # ---------------- TCGA: FFPE diagnostic slides, external test ----------------
    tcga = df[df.cohort.str.startswith("tcga") & (df.tissue == "Neoplasm, Primary")
              & df.embedding.str.contains("Paraffin")].copy()
    tcga["site"] = tcga.patient_id.str.split("-").str[1]  # TCGA tissue-source-site code
    for cohort, label in (("tcga_luad", "LUAD"), ("tcga_lusc", "LUSC")):
        pool = tcga[tcga.cohort == cohort].assign(r=lambda d: rng.random(len(d)))
        pool = pool.sort_values("r").drop_duplicates("patient_id")
        # round-robin over sites so no single hospital dominates the external test
        pool["site_rank"] = pool.groupby("site").cumcount()
        pick = pool.sort_values(["site_rank", "r"]).head(args.n_external_per_class)
        parts.append(pick.drop(columns=["r", "site_rank"]).assign(label=label))

    m = pd.concat(parts, ignore_index=True)
    m["split_group"] = np.where(m.cohort.str.startswith("cptac"), "cptac", "external")
    m["prep_type"] = np.where(m.cohort.str.startswith("tcga"), "FFPE", "unrecorded (FFPE on visual check)")
    m["patch_px"] = ((PATCH_UM / m.mpp) / 16).round().astype(int) * 16  # same physical size at any scan resolution
    m["source_uri"] = "idc://" + m.series_uid

    # stratified pilot subset: n_pilot per class from CPTAC, distinct patients
    m["pilot"] = False
    for label in ("LUAD", "LUSC", "Normal"):
        idx = m[(m.split_group == "cptac") & (m.label == label)].sample(args.n_pilot, random_state=args.seed).index
        m.loc[idx, "pilot"] = True

    m["status"] = "todo"
    cols = ["slide_id", "patient_id", "cohort", "label", "split_group", "pilot", "label_verified", "site",
            "prep_type", "mpp", "magnification", "patch_px", "width", "height", "size_mb", "series_uid",
            "source_uri", "status"]
    return m[cols].sort_values(["split_group", "label", "patient_id"]).reset_index(drop=True)


def check(m: pd.DataFrame) -> None:
    assert m.slide_id.is_unique, "duplicate slide ids"
    assert not m.duplicated(["patient_id", "label"]).any(), "more than one slide per patient per class"
    cptac_pats = set(m[m.split_group == "cptac"].patient_id)
    assert not cptac_pats & set(m[m.split_group == "external"].patient_id), "patient in both cohorts"
    normals = m[(m.split_group == "cptac") & (m.label == "Normal")]
    tumors = set(m[(m.split_group == "cptac") & (m.label != "Normal")].patient_id)
    assert normals.patient_id.isin(tumors).all(), "every normal must come from a sampled tumor patient"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-class", type=int, default=150)
    ap.add_argument("--n-external-per-class", type=int, default=60)
    ap.add_argument("--n-pilot", type=int, default=10, help="pilot slides per class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-prefer-paired", dest="prefer_paired", action="store_false")
    ap.add_argument("--out", type=Path, default=ROOT / "manifests" / "slides.csv")
    args = ap.parse_args()

    m = build(args)
    check(m)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.out, index=False)

    print(f"wrote {args.out.relative_to(ROOT)}: {len(m)} slides, {m.patient_id.nunique()} patients, "
          f"{m.size_mb.sum() / 1024:.0f} GB to stream\n")
    summary = m.groupby(["split_group", "cohort", "label"]).agg(
        slides=("slide_id", "size"), patients=("patient_id", "nunique"), verified=("label_verified", "sum"),
        pilot=("pilot", "sum"), GB=("size_mb", lambda s: round(s.sum() / 1024, 1)),
        patch_px=("patch_px", lambda s: "/".join(map(str, sorted(s.unique())))))
    print(summary.to_string())
    ext = m[m.split_group == "external"]
    print(f"\nexternal test spans {ext.site.nunique()} TCGA tissue-source sites")


if __name__ == "__main__":
    main()
