# lung-clam

Weakly supervised **Normal / LUAD / LUSC** classification of lung whole-slide images with [CLAM](https://github.com/mahmoodlab/CLAM), built to run on an Apple Silicon Mac.

> 🚧 Work in progress. See [ROADMAP.md](ROADMAP.md) for the plan.
> ⚠️ Research and education only. Not a medical device, and not for clinical use.

## Setup

Requires macOS (Apple Silicon) or Linux, git and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules <repo-url> lung-clam
cd lung-clam
bash scripts/setup.sh
```

`setup.sh`:
1. fetches CLAM (a pinned git submodule);
2. applies our patches from `patches/`;
3. installs a Python 3.10 environment;
4. runs `scripts/check_env.py`.

## Changes to CLAM

[`patches/0001-apple-mps-and-fp16-features.patch`](patches/0001-apple-mps-and-fp16-features.patch):
- Selects the device as `cuda → mps → cpu`, and replaces the hard-coded `.cuda()` calls.
- Rebuilds smooth-topk's SVM losses after moving them to a device. They capture their label buffer when built, which breaks on MPS.
- Loads checkpoints with `map_location`, so a model trained on a Colab GPU loads on the Mac.
- Loads feature files with `.float()`, so features can be stored as fp16 to halve disk use.

timm is upgraded from CLAM's pinned 0.9.8 to 1.x, because 0.9.8 cannot import huggingface-hub 2.x.

## License

GPL-3.0, inherited from CLAM.
