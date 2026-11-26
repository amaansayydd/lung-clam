# Pathologist review sheet

Thanks for looking at these. Each slide opens in a browser in the free NCI Imaging Data Commons viewer (*view*), with no login or download needed. *heatmap* shows where the model looked (red = high attention).
The viewer may show "Failed to fetch ICC profiles" pop-ups; close them, they only affect colour calibration. The left panel lists every slide from the same patient. About 30–45 minutes in total. Short answers are perfect.

## A. Are these labels right? (most important)

The model disagrees confidently with the recorded label. If the record is wrong, the reported test score is too pessimistic; if the model is wrong, it is a real error.

| Slide | Recorded | Model says | Why it is flagged | Links | Your call (tumour type / normal / unsure) |
|---|---|---|---|---|---|
| C3L-03678-21 | LUSC | Normal (0.86) | Recorded tumour slide looks like non-neoplastic lung | [view](https://viewer.imaging.datacommons.cancer.gov/slim/studies/2.25.311598809998424507957659954360300973391/series/1.3.6.1.4.1.5962.99.1.225843069.982198393.1640903382909.2.0) · [heatmap](../results/heatmaps/final_clam_sb_phikon/C3L-03678-21_heatmap.jpg) | |
| C3L-03678-22 | Normal | LUSC (0.99) | Recorded normal slide looks like squamous carcinoma | [view](https://viewer.imaging.datacommons.cancer.gov/slim/studies/2.25.311598809998424507957659954360300973391/series/1.3.6.1.4.1.5962.99.1.215111917.53065631.1640892651757.2.0) · [heatmap](../results/heatmaps/final_clam_sb_phikon/C3L-03678-22_heatmap.jpg) | |
| C3N-02435-22 | LUSC | Normal (0.61) | Little visible tumour; tumour-poor section? | [view](https://viewer.imaging.datacommons.cancer.gov/slim/studies/2.25.214924400191612718667913472214569258846/series/1.3.6.1.4.1.5962.99.1.220749468.1255384420.1640898289308.2.0) · [heatmap](../results/heatmaps/final_clam_sb_phikon/C3N-02435-22_heatmap.jpg) | |
| C3L-02168-25 | Normal | — | Very dark, dense normal-adjacent slide; possible tumour in it | [view](https://viewer.imaging.datacommons.cancer.gov/slim/studies/2.25.304502937470642449627306362815456999353/series/1.3.6.1.4.1.5962.99.1.234862924.426571652.1640912402764.2.0) | |
| C3L-02619-26 | Normal | LUSC | The only CV normal slide called tumour; tumour contamination? | [view](https://viewer.imaging.datacommons.cancer.gov/slim/studies/2.25.330144441828572805997805696076049257346/series/1.3.6.1.4.1.5962.99.1.219060349.819307316.1640896600189.2.0) · [heatmap](../results/heatmaps/clam_sb_phikon/C3L-02619-26_heatmap.jpg) | |

Especially: **are C3L-03678-21 and C3L-03678-22 swapped?** (same patient; -21 recorded LUSC, -22 recorded Normal).

## B. LUAD vs LUSC errors

Most of the model's mistakes are between the two subtypes. For each, is the recorded subtype clear on H&E, or is the tumour poorly differentiated / mixed / hard to call?

| Slide | Recorded | Model says | Links | Clear / hard / disagree with record |
|---|---|---|---|---|
| C3N-02588-22 | LUAD | LUSC | [view](https://viewer.imaging.datacommons.cancer.gov/slim/studies/2.25.166370419946104609797751884974750956842/series/1.3.6.1.4.1.5962.99.1.252191814.317228028.1640929731654.2.0) · [heatmap](../results/heatmaps/final_clam_sb_phikon/C3N-02588-22_heatmap.jpg) | |
| C3L-02170-23 | LUSC | LUAD | [view](https://viewer.imaging.datacommons.cancer.gov/slim/studies/2.25.330049697501126480084424990605239901339/series/1.3.6.1.4.1.5962.99.1.223571778.477296523.1640901111618.2.0) · [heatmap](../results/heatmaps/final_clam_sb_phikon/C3L-02170-23_heatmap.jpg) | |
| C3N-01020-22 | LUSC | LUAD | [view](https://viewer.imaging.datacommons.cancer.gov/slim/studies/2.25.281516496022280908775623771273685427647/series/1.3.6.1.4.1.5962.99.1.208682275.129649602.1640886222115.2.0) · [heatmap](../results/heatmaps/final_clam_sb_phikon/C3N-01020-22_heatmap.jpg) | |
| C3N-00247-22 | LUSC | LUAD | [view](https://viewer.imaging.datacommons.cancer.gov/slim/studies/2.25.155386925798410753995158327701570112334/series/1.3.6.1.4.1.5962.99.1.223287002.432109879.1640900826842.2.0) · [heatmap](../results/heatmaps/final_clam_sb_phikon/C3N-00247-22_heatmap.jpg) | |
| C3N-00555-22 | LUSC | LUAD | [view](https://viewer.imaging.datacommons.cancer.gov/slim/studies/2.25.41271271351062721460339920319914617399/series/1.3.6.1.4.1.5962.99.1.234718894.2126732241.1640912258734.2.0) · [heatmap](../results/heatmaps/final_clam_sb_phikon/C3N-00555-22_heatmap.jpg) | |
| C3L-02552-22 | LUSC | LUAD | [view](https://viewer.imaging.datacommons.cancer.gov/slim/studies/2.25.74444150306967826357701381050625398978/series/1.3.6.1.4.1.5962.99.1.221692780.1002639916.1640899232620.2.0) · [heatmap](../results/heatmaps/final_clam_sb_phikon/C3L-02552-22_heatmap.jpg) | |
| C3N-04168-23 | LUAD | LUSC | [view](https://viewer.imaging.datacommons.cancer.gov/slim/studies/2.25.115960460423907960394361057907054666959/series/1.3.6.1.4.1.5962.99.1.240230159.1809712223.1640917769999.2.0) · [heatmap](../results/heatmaps/clam_sb_phikon/C3N-04168-23_heatmap.jpg) | |
| C3L-00080-21 | LUAD | LUSC | [view](https://viewer.imaging.datacommons.cancer.gov/slim/studies/2.25.116585212856239497027541895138175720563/series/1.3.6.1.4.1.5962.99.1.245680070.903043566.1640923219910.2.0) · [heatmap](../results/heatmaps/clam_sb_phikon/C3L-00080-21_heatmap.jpg) | |

## C. Does the attention make sense?

Open 3–4 heatmaps of correctly classified slides, for example [C3N-05079-21](../results/heatmaps/clam_sb_phikon/C3N-05079-21_heatmap.jpg) (LUSC) or any in [`results/heatmaps/`](../results/heatmaps). Is the red attention on tumour, or on something else (stroma, blood, necrosis, artefacts, pen marks)?

## Notes

- These are research slides from the public CPTAC collections. No identifiable patient information is involved.
- Results are recorded in [`data_findings.md`](data_findings.md). The sealed test score stays as originally reported; any confirmed label errors are reported beside it.
