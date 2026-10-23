#!/usr/bin/env bash
# One-time setup after cloning: fetch CLAM, apply our patches, install the environment.
set -euo pipefail
cd "$(dirname "$0")/.."

git submodule update --init CLAM
for p in patches/*.patch; do
    if git -C CLAM apply --check "../$p" 2>/dev/null; then
        git -C CLAM apply "../$p" && echo "applied $p"
    else
        echo "skipped $p (already applied or does not match)"
    fi
done

uv sync
uv run python scripts/check_env.py
