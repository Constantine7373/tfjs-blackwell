'use strict';
/**
 * RTX 5070 benchmark — reports GPU compute throughput separately from
 * GPU→CPU readback overhead, since WSL2 paravirtualization limits
 * readback bandwidth to ~600 MB/s (vs 32 GB/s on native Linux).
 *
 * "GPU only"  — ops are pipelined; one readback at the end amortises the
 *               WSL2 transfer cost. This is what production inference looks
 *               like (keep everything on GPU, read only the final result).
 *
 * "with copy" — one readback per iteration, as the benchmark originally ran.
 *               Includes WSL2 GPU→CPU latency.
 */

const { tf, init } = require('./index');

const RUNS = 30;   // timed iterations
const WARM = 5;    // warmup iterations (not timed)

// ── helpers ────────────────────────────────────────────────────────────────

async function warmup(fn) {
  for (let i = 0; i < WARM; i++) {
    const t = tf.tidy(fn);
    await t.data();
    t.dispose();
  }
}

/**
 * GPU-only time: queue RUNS ops without inter-op sync, then one readback.
 * Measures pure GPU throughput as it would appear in a full pipeline.
 */
async function timedGPU(fn) {
  await warmup(fn);
  const t0 = Date.now();
  const pending = [];
  for (let i = 0; i < RUNS; i++) pending.push(tf.tidy(fn));
  await pending[pending.length - 1].data();  // sync GPU once
  tf.dispose(pending);                        // free all queued tensors
  return (Date.now() - t0) / RUNS;
}

/**
 * Wall-clock time including per-iteration GPU→CPU copy (WSL2 bottleneck).
 * This is what you see when each JS callback needs the result immediately.
 */
async function timedFull(fn) {
  await warmup(fn);
  const t0 = Date.now();
  for (let i = 0; i < RUNS; i++) {
    const t = tf.tidy(fn);
    await t.data();
    t.dispose();
  }
  return (Date.now() - t0) / RUNS;
}

function gflops(n3ops, ms) {
  return ((n3ops * 2) / 1e9 / (ms / 1000)).toFixed(0);
}

async function row(label, fn, flops) {
  const gpu  = await timedGPU(fn);
  const full = await timedFull(fn);
  const gf = flops ? ` (${gflops(flops, gpu)} GFLOPS)` : '';
  console.log(
    `  ${label.padEnd(36)}` +
    `GPU: ${String(gpu.toFixed(1)).padStart(7)} ms` +
    `  |  total: ${String(full.toFixed(1)).padStart(7)} ms${gf}`
  );
}

// ── benchmark suite ────────────────────────────────────────────────────────

async function run() {
  await init();

  console.log('\n=== RTX 5070  TensorFlow.js  Benchmark ===');
  console.log(  '    GPU only = pipelined, no per-iter readback');
  console.log(  '    total    = includes WSL2 GPU→CPU copy\n');
  console.log(`  ${'operation'.padEnd(36)}${'GPU only'.padStart(12)}     ${'+ copy (WSL2)'.padStart(15)}`);
  console.log('  ' + '─'.repeat(72));

  // ── matMul ──────────────────────────────────────────────────────────────
  const mm = (n) => () => {
    const a = tf.randomNormal([n, n]);
    const b = tf.randomNormal([n, n]);
    return tf.matMul(a, b);
  };
  // reuse tensors to avoid allocating inside timed loop
  const [A512,  B512]  = [tf.randomNormal([512, 512]),   tf.randomNormal([512, 512])];
  const [A2048, B2048] = [tf.randomNormal([2048, 2048]), tf.randomNormal([2048, 2048])];
  const [A4096, B4096] = [tf.randomNormal([4096, 4096]), tf.randomNormal([4096, 4096])];

  await row('matMul  512×512',   () => tf.matMul(A512,  B512),  512**3);
  await row('matMul  2048×2048', () => tf.matMul(A2048, B2048), 2048**3);
  await row('matMul  4096×4096', () => tf.matMul(A4096, B4096), 4096**3);
  tf.dispose([A512, B512, A2048, B2048, A4096, B4096]);

  // ── batchMatMul ─────────────────────────────────────────────────────────
  const [BmA, BmB] = [tf.randomNormal([256, 512, 64]), tf.randomNormal([256, 64, 512])];
  await row('batchMatMul  B=256 seq=512 d=64', () => tf.matMul(BmA, BmB), 256*512*64*512);
  tf.dispose([BmA, BmB]);

  // ── conv2d ──────────────────────────────────────────────────────────────
  const [CX, CW] = [tf.randomNormal([8, 56, 56, 256]), tf.randomNormal([3, 3, 256, 256])];
  await row('conv2d  8×56×56×256  3×3×256', () => tf.conv2d(CX, CW, 1, 'same'));
  tf.dispose([CX, CW]);

  // ── batchNorm ────────────────────────────────────────────────────────────
  const [BN, BNm, BNv] = [
    tf.randomNormal([8, 56, 56, 256]),
    tf.zeros([256]),
    tf.ones([256]),
  ];
  await row('batchNorm  [8,56,56,256]', () => tf.batchNorm(BN, BNm, BNv));
  tf.dispose([BN, BNm, BNv]);

  // ── softmax ──────────────────────────────────────────────────────────────
  const SF = tf.randomNormal([512, 50257]);
  await row('softmax  [512, 50257]  GPT vocab', () => tf.softmax(SF));
  tf.dispose([SF]);

  // ── dense model ──────────────────────────────────────────────────────────
  const model = tf.sequential({ layers: [
    tf.layers.dense({ inputShape: [256], units: 1024, activation: 'relu' }),
    tf.layers.dense({ units: 1024, activation: 'relu' }),
    tf.layers.dense({ units: 10,   activation: 'softmax' }),
  ]});
  const INP = tf.randomNormal([512, 256]);
  // single predict to build model weights
  const p0 = model.predict(INP); await p0.data(); p0.dispose();
  await row('dense  256→1024→1024→10  bs=512', () => model.predict(INP));
  tf.dispose([INP]);

  // ── memory ───────────────────────────────────────────────────────────────
  console.log('\n  Memory');
  const mem = tf.memory();
  console.log(`  numTensors : ${mem.numTensors}`);
  console.log(`  numBytes   : ${(mem.numBytes / 1024 / 1024).toFixed(1)} MB`);
  console.log('\nBenchmark complete.\n');
}

run().catch(err => { console.error(err); process.exit(1); });
