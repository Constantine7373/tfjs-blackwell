#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 2 of 2 — user-level setup (Node.js, npm deps, TF 2.15 GPU binary)
# Run as normal user:  bash setup-project.sh
#
# Idempotent — safe to run again to update or repair.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── nvm + Node.js 22 ─────────────────────────────────────────────────────────
export NVM_DIR="${HOME}/.nvm"

if [ ! -s "${NVM_DIR}/nvm.sh" ]; then
  echo ">>> Installing nvm..."
  curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
fi

# shellcheck source=/dev/null
\. "${NVM_DIR}/nvm.sh"

NODE_VER="22"
if ! node --version 2>/dev/null | grep -q "^v${NODE_VER}"; then
  echo ">>> Installing Node.js ${NODE_VER} LTS..."
  nvm install "$NODE_VER"
fi
nvm use "$NODE_VER"
echo ">>> Node $(node -v) / npm $(npm -v)"

# ── make node/npm available system-wide (without sourcing nvm each time) ─────
NODE_BIN="$(dirname "$(command -v node)")"
for bin in node npm npx; do
  target="/usr/local/bin/${bin}"
  if [ ! -e "$target" ] || [ "$(readlink -f "$target")" != "${NODE_BIN}/${bin}" ]; then
    sudo ln -sf "${NODE_BIN}/${bin}" "$target"
    echo "    linked /usr/local/bin/${bin}"
  fi
done

# ── npm dependencies ─────────────────────────────────────────────────────────
echo ">>> Installing npm dependencies..."
npm install

# ── TF 2.15 GPU binary (CUDA 12.3, forward-compatible with sm_120) ────────────
echo ">>> Upgrading bundled libtensorflow → 2.15.0 (CUDA 12)..."
bash upgrade-libtf.sh

# ── smoke test ───────────────────────────────────────────────────────────────
echo ""
echo ">>> Smoke test — GPU matMul 4096×4096 (expect < 100 ms GPU-only)..."
node -e "
process.env.TF_CPP_MIN_LOG_LEVEL = '3';
process.env.TF_FORCE_GPU_ALLOW_GROWTH = 'true';
process.env.CUDA_VISIBLE_DEVICES = '0';
const tf = require('@tensorflow/tfjs-node-gpu');
(async () => {
  await tf.ready();
  const a = tf.randomNormal([4096, 4096]);
  const b = tf.randomNormal([4096, 4096]);
  const warm = tf.matMul(a, b); await warm.data(); warm.dispose();
  const ts = [];
  for (let i = 0; i < 10; i++) ts.push(tf.matMul(a, b));
  const t0 = Date.now();
  await ts[9].data();
  const ms = (Date.now() - t0) / 10;
  tf.dispose([a, b, ...ts]);
  const tflops = ((4096**3 * 2) / 1e12 / (ms / 1000)).toFixed(2);
  console.log('GPU-only matMul 4096^2:', ms.toFixed(1), 'ms →', tflops, 'TFLOPS');
  console.log(ms < 100 ? 'PASS' : 'WARN: slower than expected (WSL2 or no GPU?)');
})().catch(e => { console.error('FAIL:', e.message); process.exit(1); });
" 2>&1 | grep -v "^20[0-9][0-9]-\|DeprecationWarning\|Use \`node"

echo ""
echo "======================================================"
echo " Setup complete. Try:"
echo "   npm run benchmark   — detailed GPU benchmark"
echo "   npm run demo        — CNN training on synthetic data"
echo "======================================================"
