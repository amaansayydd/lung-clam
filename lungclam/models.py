"""Baseline MIL models that CLAM does not ship, written to CLAM's model interface.

CLAM's training loop expects `forward(h) -> (logits, Y_prob, Y_hat, A_raw, results_dict)`
with h of shape (n_patches, embed_dim) and logits of shape (1, n_classes).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPoolMIL(nn.Module):
    """Embed each patch, average over the slide, classify. No attention: every patch counts equally."""

    def __init__(self, n_classes=3, embed_dim=1024, dropout=0.25, **_):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(embed_dim, 512), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(512, n_classes)
        self.n_classes = n_classes

    def forward(self, h, **_):
        logits = self.classifier(self.fc(h).mean(dim=0, keepdim=True))
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        return logits, Y_prob, Y_hat, None, {}
