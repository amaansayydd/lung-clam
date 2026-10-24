"""Point the TCGA rows of the manifest at the GDC's original Aperio .svs files.

IDC's DICOM copies of the TCGA diagnostic slides are JPEG 2000 compressed, which OpenSlide's
DICOM reader cannot decode ("OpenSlideError: Reading ..."). The GDC serves the original .svs
files (open access), which OpenSlide reads natively. CPTAC rows keep their IDC source.

Adds gdc_file_id, gdc_file_name, gdc_md5 and sets source_uri = gdc://<file_id> and size_mb
for TCGA rows.  Run: uv run python scripts/02_add_gdc_ids.py
"""
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungclam import paths  # noqa: E402

GDC_FILES = "https://api.gdc.cancer.gov/files"


def gdc_diagnostic_svs() -> dict[str, dict]:
    filt = {"op": "and", "content": [
        {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
        {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}},
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-LUAD", "TCGA-LUSC"]}}]}
    r = requests.post(GDC_FILES, json={"filters": filt, "fields": "file_id,file_name,file_size,md5sum,access",
                                       "size": 10000}, timeout=120)
    r.raise_for_status()
    return {h["file_name"].split(".")[0]: h for h in r.json()["data"]["hits"]}


def main():
    m = pd.read_csv(paths.MANIFEST)
    gdc = gdc_diagnostic_svs()
    tcga = m.cohort.str.startswith("tcga")
    missing = sorted(set(m.loc[tcga, "slide_id"]) - set(gdc))
    if missing:
        raise SystemExit(f"{len(missing)} TCGA slides not found on GDC: {missing[:5]}")
    for col in ("gdc_file_id", "gdc_file_name", "gdc_md5"):
        if col not in m:
            m[col] = ""
        m[col] = m[col].astype(object)
    for i in m.index[tcga]:
        h = gdc[m.at[i, "slide_id"]]
        m.loc[i, ["gdc_file_id", "gdc_file_name", "gdc_md5"]] = [h["file_id"], h["file_name"], h["md5sum"]]
        m.at[i, "source_uri"] = f"gdc://{h['file_id']}"
        m.at[i, "size_mb"] = h["file_size"] / 1024 ** 2
    m.to_csv(paths.MANIFEST, index=False)
    t = m[tcga]
    print(f"updated {tcga.sum()} TCGA rows: {t.size_mb.sum() / 1024:.1f} GB from GDC (open access)")


if __name__ == "__main__":
    main()
