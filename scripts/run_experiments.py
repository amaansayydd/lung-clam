"""Run groups of experiments in order, then score them. Finished folds are skipped, so this can be re-run.

  uv run python scripts/run_experiments.py core          # encoder comparison + main baselines
  uv run python scripts/run_experiments.py all
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "core": ["clam_sb_resnet", "clam_sb_phikon", "meanpool_resnet", "meanpool_phikon"],
    "ablations": ["clam_mb_phikon", "abmil_phikon", "maxmil_phikon"],
    "label_efficiency": [f"{m}_phikon_lf{f}" for f in (10, 25, 50) for m in ("clam_sb", "abmil", "meanpool")],
}
GROUPS["uni"] = ["clam_sb_uni2", "meanpool_uni2"]  # only after UNI2-h features exist (gated model)
GROUPS["all"] = GROUPS["core"] + GROUPS["ablations"] + GROUPS["label_efficiency"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="+", choices=list(GROUPS))
    ap.add_argument("--set", nargs="*", default=[], help="overrides passed to every experiment")
    args = ap.parse_args()

    names = list(dict.fromkeys(n for g in args.groups for n in GROUPS[g]))
    for i, name in enumerate(names, 1):
        print(f"\n=== [{i}/{len(names)}] {name} ===", flush=True)
        cmd = [sys.executable, str(ROOT / "scripts" / "20_train.py"), str(ROOT / "configs" / "experiments" / f"{name}.yaml")]
        if args.set:
            cmd += ["--set", *args.set]
        subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "30_evaluate.py"), *names], check=True)


if __name__ == "__main__":
    main()
