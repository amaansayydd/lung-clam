"""Check the environment: device, OpenSlide, and a short CLAM training step on random features.

Run from the repo root:  uv run python scripts/check_env.py
"""
import os
import sys
import time
import warnings

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "CLAM"))

import openslide
import torch
from topk.svm import SmoothTop1SVM

from models.model_clam import CLAM_MB, CLAM_SB

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"torch {torch.__version__} | device {device} | OpenSlide {openslide.__library_version__}")

for cls in (CLAM_SB, CLAM_MB):
    inst = SmoothTop1SVM(n_classes=2).to(device)
    inst.get_losses()
    model = cls(n_classes=3, embed_dim=1024, k_sample=8, instance_loss_fn=inst, subtyping=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    bag = torch.randn(8000, 1024, device=device)
    label = torch.tensor([1], device=device)
    start = time.time()
    for _ in range(5):
        logits, _, _, _, out = model(bag, label=label, instance_eval=True)
        loss = 0.7 * torch.nn.functional.cross_entropy(logits, label) + 0.3 * out["instance_loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"{cls.__name__}: ok ({(time.time() - start) / 5 * 1000:.0f} ms per 8k-patch slide step)")

print("environment OK")
