/**
 * TensorFlow.js GPU entry point — optimised for RTX 5070 (Blackwell sm_120).
 * Works on WSL2 and native Linux.
 *
 * Usage:
 *   import { tf, init } from './index.js';
 *   await init();
 */

import path from 'node:path';
import fs   from 'node:fs';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');

// ── CUDA library search paths ─────────────────────────────────────────────
// Must be set before the native addon is loaded (it calls dlopen internally).
// Dynamic import below ensures this runs first.

function prependLD(...dirs: string[]): void {
  const valid = dirs.filter(d => { try { return fs.statSync(d).isDirectory(); } catch { return false; } });
  if (!valid.length) return;
  const existing = process.env.LD_LIBRARY_PATH ?? '';
  process.env.LD_LIBRARY_PATH = [...valid, ...(existing ? [existing] : [])].join(':');
}

prependLD(
  path.join(REPO_ROOT, 'node_modules/@tensorflow/tfjs-node-gpu/deps/lib'),
  '/usr/local/cuda-12/lib64',
  '/usr/local/cuda-11.8/lib64',
  '/usr/lib/wsl/lib',
);

for (const bin of ['/usr/local/cuda-12/bin', '/usr/local/cuda-11.8/bin']) {
  if (!(process.env.PATH ?? '').includes(bin) && fs.existsSync(bin)) {
    process.env.PATH = `${bin}:${process.env.PATH ?? ''}`;
    break;
  }
}

// ── TensorFlow env flags ──────────────────────────────────────────────────

process.env.TF_CPP_MIN_LOG_LEVEL     ??= '1';
process.env.TF_FORCE_GPU_ALLOW_GROWTH  = 'true';
process.env.TF_GPU_ALLOCATOR           = 'cuda_malloc_async';
process.env.TF_XLA_FLAGS               = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2';
process.env.CUDA_CACHE_DISABLE         = '0';
process.env.CUDA_CACHE_MAXSIZE         = '2147483648';
process.env.CUDA_VISIBLE_DEVICES     ??= '0';

// ── load backend (dynamic so env vars above are in place for dlopen) ──────
// Top-level await: module consumers wait for this to resolve before running.
const tf = await import('@tensorflow/tfjs-node-gpu');
export type TF = typeof tf;

// ── public API ────────────────────────────────────────────────────────────

/** Await once at the top of your script. Logs backend + GPU info. */
export async function init(): Promise<void> {
  await tf.ready();

  const probe = tf.zeros([64, 64]);
  const mem   = tf.memory();
  probe.dispose();

  // numBytesInGPU is present on the GPU memory info but not in the base type
  const gpuBytes = (mem as Record<string, unknown>)['numBytesInGPU'];
  const gpuMB = typeof gpuBytes === 'number'
    ? `${(gpuBytes / 1024 / 1024).toFixed(2)} MB`
    : 'n/a (check CUDA install)';

  console.log(`[tfjs] backend  : ${tf.getBackend()}`);
  console.log(`[tfjs] GPU mem  : ${gpuMB}`);
}

export { tf };
