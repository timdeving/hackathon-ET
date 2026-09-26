#!/usr/bin/env bash
# Pin every runtime dependency: resolve requirements.in once for Python 3.10 and newer (the
# oldest Python the judges allow) on every platform, and write requirements.txt with exact
# versions. A package only one platform needs carries a marker (PyTorch's CUDA libraries and
# Triton exist for Linux only), so the same file installs on Linux and on Windows; a plain
# `pip freeze` would list Linux-only packages that have no Windows wheels. Needs uv
# (https://docs.astral.sh/uv/). Run from anywhere, then commit requirements.txt.
set -euo pipefail
cd "$(dirname "$0")/.."

# unsafe-best-match: take each package's newest allowed version from either index. The extra
# index is PyTorch's own, which also mirrors some PyPI packages at older versions.
uv pip compile requirements.in --universal --python-version 3.10 \
    --index-strategy unsafe-best-match --emit-index-url \
    --custom-compile-command "bash tools/pin_requirements.sh" -o requirements.txt
echo "requirements.txt updated; commit it."
