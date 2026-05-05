#!/usr/bin/env bash
# Replaces the CUDA-11 / TF-2.9.1 binary bundled in tfjs-node-gpu with
# TF 2.15.0 (CUDA 12.3).  CUDA 12 PTX can be JIT-compiled for sm_120
# (Blackwell) by the CUDA 13+ driver; CUDA 11 PTX cannot.
#
# Called automatically by setup-project.sh.
# Safe to run again — skips download if the tarball already exists.
#
# No sudo needed — only modifies node_modules/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_VER="2.15.0"
TF_URL="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-${TF_VER}.tar.gz"
TF_TAR="/tmp/libtensorflow-gpu-${TF_VER}.tar.gz"
DEPS_LIB="${REPO_ROOT}/node_modules/@tensorflow/tfjs-node-gpu/deps/lib"
MARKER="${DEPS_LIB}/.tf-version"

# skip if already done
if [ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "$TF_VER" ]; then
  echo ">>> libtensorflow ${TF_VER} already in place — skipping."
  exit 0
fi

# download (resume-capable)
if [ ! -f "$TF_TAR" ]; then
  echo ">>> Downloading TF ${TF_VER} GPU C-API (~400 MB)..."
  wget -q --show-progress --continue "$TF_URL" -O "$TF_TAR"
else
  echo ">>> Using cached tarball: $TF_TAR"
fi

# extract
echo ">>> Extracting into deps/lib ..."
mkdir -p "$DEPS_LIB"
# tarball layout varies across TF versions — try both common structures
tar -xzf "$TF_TAR" -C "$DEPS_LIB" --strip-components=2 \
    --wildcards 'lib/libtensorflow*.so*' 2>/dev/null \
  || tar -xzf "$TF_TAR" -C "$DEPS_LIB" --wildcards 'libtensorflow*.so*'

# canonical unversioned symlinks expected by the Node.js addon
(cd "$DEPS_LIB"
  [ -f "libtensorflow.so.${TF_VER}" ] \
    && ln -sf "libtensorflow.so.${TF_VER}" libtensorflow.so
  [ -f "libtensorflow_framework.so.${TF_VER}" ] \
    && ln -sf "libtensorflow_framework.so.${TF_VER}" libtensorflow_framework.so
)

echo "$TF_VER" > "$MARKER"

echo ">>> Installed:"
ls -lh "$DEPS_LIB"/*.so* 2>/dev/null || ls -lh "$DEPS_LIB"
echo ""
