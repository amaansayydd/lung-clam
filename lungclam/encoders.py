"""Patch encoders: images of 128 µm tissue patches → one feature vector each.

All encoders here take 224×224 RGB input normalized with ImageNet mean/std, so one
preprocessed batch can be fed to several encoders in a single pass over the slide.

| name            | model                                  | dim  | access              |
|-----------------|----------------------------------------|------|---------------------|
| resnet50_trunc  | ImageNet ResNet50, cut after layer 3   | 1024 | open (CLAM default) |
| phikon          | Owkin Phikon, ViT-B/16 iBOT on TCGA    | 768  | open                |
| phikon_v2       | Owkin Phikon-v2, ViT-L/16 DINOv2       | 1024 | open                |
| uni_v1          | MahmoodLab UNI, ViT-L/16 DINOv2        | 1024 | gated on HF         |
| uni2_h          | MahmoodLab UNI2-h, ViT-H/14 DINOv2     | 1536 | gated on HF         |
"""
import torch
from torchvision import transforms

from .paths import add_clam_to_path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ENCODER_DIMS = {"resnet50_trunc": 1024, "phikon": 768, "phikon_v2": 1024, "uni_v1": 1024, "uni2_h": 1536}


def preprocess(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class _HFViTCLS(torch.nn.Module):
    """Hugging Face ViT / DINOv2 backbone returning the CLS token (as the Phikon authors recommend)."""

    def __init__(self, repo: str):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(repo)

    def forward(self, x):
        return self.model(pixel_values=x).last_hidden_state[:, 0, :]


def build_encoder(name: str) -> torch.nn.Module:
    if name == "resnet50_trunc":
        add_clam_to_path()
        from models.timm_wrapper import TimmCNNEncoder
        model = TimmCNNEncoder()
    elif name == "phikon":
        model = _HFViTCLS("owkin/phikon")
    elif name == "phikon_v2":
        model = _HFViTCLS("owkin/phikon-v2")
    elif name == "uni_v1":
        import timm
        model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    elif name == "uni2_h":
        import timm
        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h", pretrained=True, img_size=224, patch_size=14, depth=24, num_heads=24,
            init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
    else:
        raise ValueError(f"unknown encoder {name!r}; choose from {sorted(ENCODER_DIMS)}")
    return model.eval()


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
