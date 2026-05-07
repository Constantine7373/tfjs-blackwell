# tensorflow-sm120

Drop-in replacement for `@tensorflow/tfjs-node-gpu` with **native sm_120 (Blackwell) kernels**. Targets NVIDIA RTX 50xx series and any other Compute Capability 12.x GPU.

Identical API to `@tensorflow/tfjs-node-gpu`. No init call, no config — just import and run.

---

## Requirements

- **Linux only** (native or WSL2) — Windows and macOS are not supported
- NVIDIA RTX 50xx (Blackwell, sm_120) or other Compute Capability 12.x GPU
- CUDA 12.8+ driver ([CUDA 12 on Ubuntu](https://developer.nvidia.com/cuda-downloads))
- Node.js ≥ 20

---

## Install

```bash
npm install @constantine7373/tensorflow-sm120
```

---

## Usage

```ts
import * as tf from '@constantine7373/tensorflow-sm120';

await tf.ready();

const a = tf.randomNormal([1024, 1024]);
const b = tf.randomNormal([1024, 1024]);
const c = tf.matMul(a, b);

console.log(await c.data());
tf.dispose([a, b, c]);
```

Env defaults applied automatically (can be overridden before import):

| Variable | Default | Effect |
|---|---|---|
| `TF_FORCE_GPU_ALLOW_GROWTH` | `true` | Allocates VRAM on demand |
| `TF_GPU_ALLOCATOR` | `cuda_malloc_async` | Reduces fragmentation OOM |
| `TF_XLA_FLAGS` | `--tf_xla_auto_jit=2` | Fuses ops into XLA kernels |
| `CUDA_CACHE_MAXSIZE` | 2 GB | Persists compiled kernels across runs |

To override:
```ts
process.env.TF_FORCE_GPU_ALLOW_GROWTH = 'false'; // must be set BEFORE import
import * as tf from '@constantine7373/tensorflow-sm120';
```

---

## Performance (RTX 5070, WSL2)

| Operation | Time | Throughput |
|---|---|---|
| matMul 4096² | ~24 ms | ~17 TFLOPS |
