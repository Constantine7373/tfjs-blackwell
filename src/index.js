'use strict';
/**
 * TensorFlow.js GPU entry point — optimised for RTX 5070 (Blackwell sm_120).
 * Works on WSL2 and native Linux.
 *
 * Usage in your project:
 *   const { tf, init } = require('./src/index');
 *   await init();               // logs backend + GPU info
 *   // ... use tf normally ...
 */

const path = require('path');
const fs   = require('fs');

// ── resolve repo root regardless of where this module is required from ───────
const REPO_ROOT = path.resolve(__dirname, '..');

// ── CUDA / library search paths (set before the native addon loads) ──────────
function prependLD(...dirs) {
  const keep = dirs.filter(d => {
    try { return fs.statSync(d).isDirectory(); } catch { return false; }
  });
  if (!keep.length) return;
  const existing = process.env.LD_LIBRARY_PATH ?? '';
  process.env.LD_LIBRARY_PATH = keep.concat(existing ? [existing] : []).join(':');
}

prependLD(
  path.join(REPO_ROOT, 'node_modules/@tensorflow/tfjs-node-gpu/deps/lib'),
  '/usr/local/cuda-12/lib64',     // CUDA 12 runtime (TF 2.15 needs this)
  '/usr/local/cuda-11.8/lib64',   // CUDA 11 runtime (fallback)
  '/usr/lib/wsl/lib',             // WSL2 libcuda.so from Windows driver
);

// ptxas for the CUDA redzone allocator (non-critical, suppresses a warning)
for (const bin of ['/usr/local/cuda-12/bin', '/usr/local/cuda-11.8/bin']) {
  if (!(process.env.PATH ?? '').includes(bin) && fs.existsSync(bin)) {
    process.env.PATH = `${bin}:${process.env.PATH ?? ''}`;
    break;
  }
}

// ── TensorFlow environment flags ─────────────────────────────────────────────

// Suppress verbose C++ info logs (0=all, 1=no-INFO, 2=no-WARN, 3=ERROR only)
process.env.TF_CPP_MIN_LOG_LEVEL  ??= '1';

// Grow VRAM on demand instead of reserving everything at startup.
process.env.TF_FORCE_GPU_ALLOW_GROWTH = 'true';

// Async CUDA memory allocator — reduces fragmentation OOM on large batches.
process.env.TF_GPU_ALLOCATOR = 'cuda_malloc_async';

// XLA JIT fuses op sequences into single kernels — helps large models.
process.env.TF_XLA_FLAGS = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2';

// Persist PTX JIT-compiled kernels across process restarts.
process.env.CUDA_CACHE_DISABLE  = '0';
process.env.CUDA_CACHE_MAXSIZE  = '2147483648'; // 2 GB

// Pin to GPU 0; override with CUDA_VISIBLE_DEVICES=1 in the environment.
process.env.CUDA_VISIBLE_DEVICES ??= '0';

// ── load native backend ───────────────────────────────────────────────────────
const tf = require('@tensorflow/tfjs-node-gpu');

// ── public API ────────────────────────────────────────────────────────────────

/**
 * Await this once at the top of your script.
 * Logs backend name, GPU memory use, and TFLOPS probe.
 * Returns the `tf` instance so callers can do: const { tf } = await init();
 */
async function init() {
  await tf.ready();

  const probe = tf.zeros([64, 64]);
  const mem   = tf.memory();
  probe.dispose();

  const gpuMB = mem.numBytesInGPU != null
    ? `${(mem.numBytesInGPU / 1024 / 1024).toFixed(2)} MB`
    : 'n/a (check CUDA install)';

  console.log(`[tfjs] backend  : ${tf.getBackend()}`);
  console.log(`[tfjs] GPU mem  : ${gpuMB}`);

  return tf;
}

module.exports = { tf, init };
