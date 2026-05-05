#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 1 of 2 — system-level prerequisites (CUDA, cuDNN)
# Run as:  sudo bash setup-system.sh
#
# Supports:
#   • WSL2 Ubuntu 24.04   (driver comes from Windows; uses wsl-ubuntu repo)
#   • Native Ubuntu 22.04 / 24.04  (installs NVIDIA driver from apt)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── detect environment ────────────────────────────────────────────────────────
if grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then
  PLATFORM="wsl2"
  CUDA_DISTRO="wsl-ubuntu"
else
  PLATFORM="native"
  # map Ubuntu release to NVIDIA repo slug
  UBUNTU_VER=$(lsb_release -rs 2>/dev/null | tr -d '.')
  case "$UBUNTU_VER" in
    2204) CUDA_DISTRO="ubuntu2204" ;;
    2404) CUDA_DISTRO="ubuntu2404" ;;
    *)    CUDA_DISTRO="ubuntu2204" ;;   # fallback
  esac
fi
echo ">>> Platform: $PLATFORM  |  CUDA repo: $CUDA_DISTRO"

# ── NVIDIA driver (native only — WSL2 inherits it from Windows) ──────────────
if [ "$PLATFORM" = "native" ]; then
  if ! command -v nvidia-smi &>/dev/null; then
    echo ">>> Installing NVIDIA driver..."
    apt-get update -qq
    apt-get install -y --no-install-recommends \
      ubuntu-drivers-common
    ubuntu-drivers autoinstall
    echo ""
    echo "!!! REBOOT REQUIRED to activate the NVIDIA driver."
    echo "!!! Run this script again after rebooting."
    exit 0
  else
    echo ">>> NVIDIA driver already present: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo '?')"
  fi
fi

# ── CUDA keyring ──────────────────────────────────────────────────────────────
CUDA_KEY=/usr/share/keyrings/cuda-archive-keyring.gpg
if [ ! -f "$CUDA_KEY" ]; then
  echo ">>> Adding CUDA keyring..."
  KEYRING=/tmp/cuda-keyring_1.1-1_all.deb
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb" \
       -O "$KEYRING"
  dpkg -i "$KEYRING"
else
  echo ">>> CUDA keyring already installed."
fi

# ── apt sources ───────────────────────────────────────────────────────────────
echo ">>> Configuring CUDA apt repositories..."
# primary (toolkit 11.8)
echo "deb [signed-by=${CUDA_KEY}] https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_DISTRO}/x86_64/ /" \
  > /etc/apt/sources.list.d/cuda-primary.list
# ubuntu2204 always needed for libcudnn8 (not in wsl-ubuntu repo)
echo "deb [signed-by=${CUDA_KEY}] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
  > /etc/apt/sources.list.d/cuda-ubuntu2204.list

apt-get update -qq

# ── CUDA 11.8 toolkit — provides the sonames TF 2.9.x links against ──────────
echo ">>> Installing CUDA 11.8 runtime libraries..."
apt-get install -y --no-install-recommends \
  cuda-cudart-11-8       \
  cuda-libraries-11-8    \
  libcublas-11-8         \
  libcufft-11-8          \
  libcurand-11-8         \
  libcusolver-11-8       \
  libcusparse-11-8       \
  cuda-nvtx-11-8

# ── cuDNN 8 — used by TF 2.9 and TF 2.15 ─────────────────────────────────────
echo ">>> Installing cuDNN 8..."
apt-get install -y --no-install-recommends \
  libcudnn8 \
  libcudnn8-dev

# ── CUDA 12 runtime — needed by TF 2.15 (CUDA 12.3) binary ──────────────────
echo ">>> Installing CUDA 12 runtime libraries..."
LATEST12() { apt-cache search "^${1}-12-" | awk '{print $1}' | sort -V | tail -1; }
apt-get install -y --no-install-recommends \
  "$(LATEST12 cuda-cudart)"  \
  "$(LATEST12 libcublas)"    \
  "$(LATEST12 libcufft)"     \
  "$(LATEST12 libcurand)"    \
  "$(LATEST12 libcusolver)"  \
  "$(LATEST12 libcusparse)"

# ── build tools ───────────────────────────────────────────────────────────────
apt-get install -y --no-install-recommends \
  build-essential python3 python3-pip

# ── LD / PATH profile ─────────────────────────────────────────────────────────
PROFILE=/etc/profile.d/cuda-wsl.sh
cat > "$PROFILE" <<'EOF'
export PATH=/usr/local/cuda-12/bin:/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:/usr/local/cuda-11.8/lib64:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOF
chmod +x "$PROFILE"
# shellcheck source=/dev/null
source "$PROFILE"

echo ""
echo "======================================================"
echo " System setup complete ($PLATFORM)."
echo " Driver:  $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'see nvidia-smi')"
echo " CUDA:    $(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' || echo 'see /usr/local/cuda-11.8')"
echo ""
echo " Next:  bash setup-project.sh"
echo "======================================================"
