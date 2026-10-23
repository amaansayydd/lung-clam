"""Summarize lung slide-microscopy collections in NCI Imaging Data Commons (IDC).

Prints slide/patient counts, size and resolution per cohort, preparation and tissue type.
Downloads only the IDC metadata index (no images).  Run: uv run python scripts/00_explore_idc.py
"""
import pandas as pd
from idc_index import IDCClient

QUERY = """
SELECT i.collection_id,
       s.embeddingMedium_CodeMeaning AS embedding,
       s.primaryAnatomicStructureModifier_CodeMeaning AS tissue,
       COUNT(*) AS slides,
       COUNT(DISTINCT i.PatientID) AS patients,
       ROUND(SUM(i.series_size_MB) / 1024, 1) AS GB,
       ROUND(MEDIAN(s.min_PixelSpacing_2sf) * 1000, 3) AS median_um_per_px
FROM sm_index s JOIN index i USING (SeriesInstanceUID)
WHERE i.collection_id IN ('cptac_luad', 'cptac_lscc', 'tcga_luad', 'tcga_lusc')
GROUP BY ALL
ORDER BY 1, slides DESC
"""

client = IDCClient()
client.fetch_index("sm_index")
print(f"IDC data version: {client.get_idc_version()}")
pd.set_option("display.width", 200)
print(client.sql_query(QUERY).to_string(index=False))
