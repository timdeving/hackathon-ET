#!/usr/bin/env bash
# Assemble the live demo's Hugging Face Space from this repository: the code and settings the
# demo runs, the model, the app, and the ready-made examples (python -m demo.make_examples).
#
#   bash demo/build_space.sh [folder]          # default: outputs/space
#   hf upload <user>/<space> outputs/space . --repo-type space
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:-$root/outputs/space}"
rm -rf "$out"
mkdir -p "$out/demo"
cp -r "$root/src" "$root/configs" "$root/weights" "$out/"
cp "$root/demo/__init__.py" "$root/demo/app.py" "$root/demo/pipeline.py" "$out/demo/"
if [ -d "$root/demo/examples" ]; then
    cp -r "$root/demo/examples" "$out/demo/"
else
    echo "warning: no ready-made examples; run python -m demo.make_examples samples first" >&2
fi
cp "$root/demo/space/README.md" "$root/demo/space/requirements.txt" \
   "$root/demo/space/packages.txt" "$out/"
find "$out" -name "__pycache__" -type d -prune -exec rm -rf {} +
rm -f "$out"/weights/*.pt
echo "Space folder: $out ($(du -sh "$out" | cut -f1))"
