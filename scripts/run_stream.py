"""Stream slides through the pipeline: download → segment → patch → features → QC → delete raw.

Raw slides are only on disk while they are being processed (plus one prefetched batch),
so the whole dataset can be processed with a few GB of free space. Each step is skipped if
its output already exists, so the script can be stopped and restarted at any time.

Examples:
  uv run python scripts/run_stream.py --pilot                       # the 30 pilot slides
  uv run python scripts/run_stream.py --group cptac                 # everything for training
  uv run python scripts/run_stream.py --encoders resnet50_trunc phikon --limit 5
  caffeinate -i uv run python scripts/run_stream.py                 # all 570, keep the Mac awake
"""
import argparse
import csv
import shutil
import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from lungclam import paths  # noqa: E402
from lungclam.encoders import best_device, build_encoder  # noqa: E402
from lungclam.features import extract  # noqa: E402
from lungclam.slides import open_wsi, save_qc_and_stats, segment_and_patch  # noqa: E402

STATUS_FIELDS = ["time", "slide_id", "status", "n_patches", "seconds", "patches_per_s", "detail"]
SAFETY_GB = 2.0


def log_status(row: dict) -> None:
    paths.STATUS.parent.mkdir(parents=True, exist_ok=True)
    new = not paths.STATUS.exists()
    with open(paths.STATUS, "a", newline="") as f:
        w = csv.DictWriter(f, STATUS_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"time": datetime.now().isoformat(timespec="seconds"), **row})


def is_done(slide_id: str, encoders: list[str]) -> bool:
    return ((paths.PATCHES / f"{slide_id}.h5").exists()
            and (paths.STATS / f"{slide_id}.json").exists()
            and all((paths.feature_dir(e) / f"{slide_id}.pt").exists() for e in encoders))


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1024 ** 3


def download(rows: pd.DataFrame) -> None:
    from idc_index import IDCClient
    need = rows.size_mb.sum() / 1024
    if free_gb(paths.RAW) - need < SAFETY_GB:
        raise OSError(f"not enough disk for this batch: need {need:.1f} GB + {SAFETY_GB} GB margin, "
                      f"have {free_gb(paths.RAW):.1f} GB free at {paths.RAW}")
    IDCClient().download_from_selection(seriesInstanceUID=rows.series_uid.tolist(), downloadDir=str(paths.RAW),
                                        dirTemplate="%SeriesInstanceUID", quiet=True, show_progress_bar=False)


def process(row, encoders: dict, device, keep_raw: bool) -> dict:
    series_dir = paths.RAW / row.series_uid
    t0 = time.time()
    try:
        wsi = open_wsi(series_dir, row.slide_id)
        h5_path, n = segment_and_patch(wsi, int(row.patch_px), paths.PATCHES)
        save_qc_and_stats(wsi, row.slide_id, float(row.mpp), n, paths.THUMBS, paths.STATS)
        if n == 0:
            return {"status": "failed", "detail": "no tissue found", "n_patches": 0}
        todo = {name: m for name, m in encoders.items()
                if not (paths.feature_dir(name) / f"{row.slide_id}.pt").exists()}
        info = {"patches_per_s": ""}
        if todo:
            out = {name: paths.feature_dir(name) / f"{row.slide_id}.pt" for name in todo}
            info = extract(wsi.getOpenSlide(), h5_path, todo, out, device)
        return {"status": "done", "n_patches": n, "seconds": round(time.time() - t0, 1),
                "patches_per_s": round(info["patches_per_s"], 1) if info["patches_per_s"] != "" else ""}
    finally:
        if not keep_raw:
            shutil.rmtree(series_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+", default=["resnet50_trunc", "phikon"])
    ap.add_argument("--pilot", action="store_true", help="only the pilot slides")
    ap.add_argument("--group", choices=["cptac", "external"], help="only this split group")
    ap.add_argument("--slides", nargs="+", help="only these slide ids")
    ap.add_argument("--limit", type=int, help="process at most this many slides")
    ap.add_argument("--batch", type=int, default=3, help="slides per download batch")
    ap.add_argument("--keep-raw", action="store_true", help="do not delete raw slides after processing")
    args = ap.parse_args()

    m = pd.read_csv(paths.MANIFEST)
    if args.pilot:
        m = m[m.pilot]
    if args.group:
        m = m[m.split_group == args.group]
    if args.slides:
        m = m[m.slide_id.isin(args.slides)]
    # pilot slides first, then interleave classes so a partial run is still balanced
    m = m.assign(_o=m.groupby("label").cumcount()).sort_values(["pilot", "_o"], ascending=[False, True])
    todo = m[[not is_done(s, args.encoders) for s in m.slide_id]]
    n_already = len(m) - len(todo)
    if args.limit:
        todo = todo.head(args.limit)
    print(f"data root: {paths.DATA_ROOT}  ({free_gb(paths.DATA_ROOT):.1f} GB free)")
    print(f"{n_already} of {len(m)} selected slides already done; {len(todo)} to process "
          f"({todo.size_mb.sum() / 1024:.1f} GB to download)")
    if todo.empty:
        return

    device = best_device()
    encoders = {name: build_encoder(name).to(device) for name in args.encoders}
    if device.type != "cpu":
        encoders = {name: m.half() for name, m in encoders.items()}
    print(f"encoders on {device}: {', '.join(encoders)}")

    batches = [todo.iloc[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    t_start, n_done, n_patches = time.time(), 0, 0
    with ThreadPoolExecutor(1) as dl:
        fut: Future = dl.submit(download, batches[0])
        for b, batch in enumerate(batches):
            try:
                fut.result()
            except Exception as e:  # download failed: log every slide in the batch and move on
                for r in batch.itertuples():
                    log_status({"slide_id": r.slide_id, "status": "failed", "detail": f"download: {e}"})
                print(f"!! download failed for batch {b}: {e}")
                if isinstance(e, OSError):
                    raise
                fut = dl.submit(download, batches[b + 1]) if b + 1 < len(batches) else None
                continue
            # prefetch the next batch while this one is processed
            fut = dl.submit(download, batches[b + 1]) if b + 1 < len(batches) else None
            for r in batch.itertuples():
                try:
                    res = process(r, encoders, device, args.keep_raw)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    res = {"status": "failed", "detail": f"{type(e).__name__}: {e}"}
                    traceback.print_exc()
                log_status({"slide_id": r.slide_id, **res})
                n_done += 1
                n_patches += res.get("n_patches", 0) or 0
                elapsed = time.time() - t_start
                eta = elapsed / n_done * (len(todo) - n_done)
                print(f"[{n_done}/{len(todo)}] {r.slide_id:<22} {r.label:<6} {res['status']:<6} "
                      f"{res.get('n_patches', 0):>6} patches  {res.get('seconds', '')}s  "
                      f"| total {n_patches} patches, {elapsed / 3600:.2f} h elapsed, ETA {eta / 3600:.2f} h",
                      flush=True)
                if device.type == "mps":
                    torch.mps.empty_cache()
        if fut is not None:
            fut.result()


if __name__ == "__main__":
    main()
