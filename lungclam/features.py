"""Feature extraction: read every patch once, run it through one or more encoders.

Patches are read with a thread pool (OpenSlide handles are thread-safe and release the GIL
while decoding), resized and normalized once on the GPU, and fed to every requested encoder. Features are stored
as fp16 `.pt` files in CLAM's layout (`<encoder>/pt_files/<slide_id>.pt`); patch coordinates
stay in the CLAM patch `.h5` file, row-aligned with the features.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch

from .encoders import to_model_input


def _batches(n: int, size: int):
    for start in range(0, n, size):
        yield start, min(start + size, n)


def extract(slide, h5_path: Path, encoders: dict[str, torch.nn.Module], out_paths: dict[str, Path],
            device: torch.device, batch_size: int = 128, n_threads: int = 8) -> dict:
    """slide: an openslide.OpenSlide. encoders: name → model already on `device`."""
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]
        patch_px = int(f["coords"].attrs["patch_size"])
        level = int(f["coords"].attrs["patch_level"])
    def load(c):
        rgba = np.asarray(slide.read_region((int(c[0]), int(c[1])), level, (patch_px, patch_px)))
        return torch.from_numpy(np.ascontiguousarray(rgba[..., :3]))

    feats = {name: [] for name in encoders}
    t0, t_model = time.time(), 0.0
    with ThreadPoolExecutor(n_threads) as pool:
        # keep one batch in flight while the GPU works on the current one
        pending = None
        for start, end in _batches(len(coords), batch_size):
            nxt = pool.map(load, coords[start:end])
            if pending is not None:
                t_model += _run(pending, encoders, feats, device)
            pending = nxt
        if pending is not None:
            t_model += _run(pending, encoders, feats, device)

    for name, chunks in feats.items():
        out = torch.cat(chunks).to(torch.float16) if chunks else torch.zeros(0, 0, dtype=torch.float16)
        out_paths[name].parent.mkdir(parents=True, exist_ok=True)
        tmp = out_paths[name].with_suffix(".tmp")
        torch.save(out, tmp)
        tmp.rename(out_paths[name])  # atomic: a half-written file never looks finished
    secs = time.time() - t0
    return {"n_patches": len(coords), "seconds": secs, "model_seconds": t_model,
            "patches_per_s": len(coords) / max(secs, 1e-9)}


@torch.inference_mode()
def _run(images_iter, encoders, feats, device) -> float:
    batch = to_model_input(torch.stack(list(images_iter)), device)
    if device.type != "cpu":
        batch = batch.half()  # fp16 compute: ~25% faster on MPS, features identical (cosine 1.0000)
    t = time.time()
    for name, model in encoders.items():
        feats[name].append(model(batch).float().cpu())
    if device.type == "mps":
        torch.mps.synchronize()
    return time.time() - t


def load_features(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu").float()


def coords_of(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        return f["coords"][:]
