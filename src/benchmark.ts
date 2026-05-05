/**
 * RTX 5070 benchmark.
 *
 * Reports two columns per operation:
 *   GPU only  — ops pipelined on GPU, one readback at the end.
 *               Reflects real inference throughput (full pass on GPU).
 *   + copy     — per-iteration GPU→CPU copy included.
 *               WSL2 caps this at ~600 MB/s; native Linux is ~32 GB/s.
 */

import type { Tensor, Tensor4D } from '@tensorflow/tfjs-node-gpu';
import { tf, init } from './index.js';

const RUNS = 30;
const WARM = 5;

// ── helpers ────────────────────────────────────────────────────────────────

async function warmup(fn: () => Tensor): Promise<void> {
  for (let i = 0; i < WARM; i++) {
    const t = tf.tidy(fn);
    await t.data();
    t.dispose();
  }
}

/** Queue RUNS ops without syncing, read back once. Pure GPU throughput. */
async function timedGPU(fn: () => Tensor): Promise<number> {
  await warmup(fn);
  const t0 = Date.now();
  const pending = Array.from({ length: RUNS }, () => tf.tidy(fn));
  await pending.at(-1)!.data();
  tf.dispose(pending);
  return (Date.now() - t0) / RUNS;
}

/** One readback per iteration — shows WSL2 copy overhead. */
async function timedFull(fn: () => Tensor): Promise<number> {
  await warmup(fn);
  const t0 = Date.now();
  for (let i = 0; i < RUNS; i++) {
    const t = tf.tidy(fn);
    await t.data();
    t.dispose();
  }
  return (Date.now() - t0) / RUNS;
}

function gflops(ops: number, ms: number): string {
  return ((ops * 2) / 1e9 / (ms / 1000)).toFixed(0);
}

async function row(label: string, fn: () => Tensor, flops?: number): Promise<void> {
  const gpu  = await timedGPU(fn);
  const full = await timedFull(fn);
  const gf   = flops ? ` (${gflops(flops, gpu)} GFLOPS)` : '';
  console.log(
    `  ${label.padEnd(36)}` +
    `GPU: ${String(gpu.toFixed(1)).padStart(7)} ms` +
    `  |  total: ${String(full.toFixed(1)).padStart(7)} ms${gf}`,
  );
}

// ── suite ──────────────────────────────────────────────────────────────────

async function run(): Promise<void> {
  await init();

  console.log('\n=== RTX 5070  TensorFlow.js  Benchmark ===');
  console.log(  '    GPU only = pipelined, no per-iter readback');
  console.log(  '    total    = includes WSL2 GPU→CPU copy\n');
  console.log(`  ${'operation'.padEnd(36)}${'GPU only'.padStart(12)}     ${'+ copy (WSL2)'.padStart(15)}`);
  console.log('  ' + '─'.repeat(72));

  // matMul — reuse tensors to exclude allocation time from the timed loop
  const mm = (n: number) => {
    const [a, b] = [tf.randomNormal([n, n]), tf.randomNormal([n, n])];
    return { a, b, fn: () => tf.matMul(a, b) };
  };
  for (const n of [512, 2048, 4096]) {
    const { a, b, fn } = mm(n);
    await row(`matMul  ${n}×${n}`, fn, n ** 3);
    tf.dispose([a, b]);
  }

  // batchMatMul
  const [bmA, bmB] = [tf.randomNormal([256, 512, 64]), tf.randomNormal([256, 64, 512])];
  await row('batchMatMul  B=256 seq=512 d=64', () => tf.matMul(bmA, bmB), 256 * 512 * 64 * 512);
  tf.dispose([bmA, bmB]);

  // conv2d
  const [cx, cw] = [tf.randomNormal([8, 56, 56, 256]), tf.randomNormal([3, 3, 256, 256])];
  await row('conv2d  8×56×56×256  3×3×256', () => tf.conv2d(cx as Tensor4D, cw as Tensor4D, 1, 'same'));
  tf.dispose([cx, cw]);

  // batchNorm
  const [bn, bnMean, bnVar] = [tf.randomNormal([8, 56, 56, 256]), tf.zeros([256]), tf.ones([256])];
  await row('batchNorm  [8,56,56,256]', () => tf.batchNorm(bn, bnMean, bnVar));
  tf.dispose([bn, bnMean, bnVar]);

  // softmax
  const sf = tf.randomNormal([512, 50257]);
  await row('softmax  [512, 50257]  GPT vocab', () => tf.softmax(sf));
  tf.dispose(sf);

  // dense model
  const model = tf.sequential({ layers: [
    tf.layers.dense({ inputShape: [256], units: 1024, activation: 'relu' }),
    tf.layers.dense({ units: 1024, activation: 'relu' }),
    tf.layers.dense({ units: 10,   activation: 'softmax' }),
  ]});
  const inp = tf.randomNormal([512, 256]);
  const p0  = model.predict(inp) as Tensor; await p0.data(); p0.dispose();
  await row('dense  256→1024→1024→10  bs=512', () => model.predict(inp) as Tensor);
  tf.dispose(inp);

  const mem = tf.memory();
  console.log(`\n  Memory  numTensors=${mem.numTensors}  numBytes=${(mem.numBytes / 1024 / 1024).toFixed(1)} MB`);
  console.log('\nBenchmark complete.\n');
}

run().catch(err => { console.error(err); process.exit(1); });
