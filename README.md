# tfjs-rtx5070

TensorFlow.js GPU compute optimised for **RTX 5070 (Blackwell sm\_120)**.  
Works on **WSL2 Ubuntu 24.04** and **native Ubuntu 22.04 / 24.04**.

---

## Prerequisites

| Requirement | WSL2 | Native Ubuntu |
|---|---|---|
| NVIDIA GPU driver | Installed via Windows (automatic) | Installed by `setup-system.sh` |
| CUDA toolkit | Installed by `setup-system.sh` | Installed by `setup-system.sh` |
| Node.js | Installed by `setup-project.sh` | Installed by `setup-project.sh` |

**WSL2 only:** make sure the Windows NVIDIA driver is up to date (595+) and `nvidia-smi` works inside WSL before proceeding.

---

## Setup

```bash
git clone <repo-url>
cd tfjs-rtx5070

# Step 1 — system dependencies (CUDA 11.8 + 12, cuDNN 8)
sudo bash setup-system.sh

# Native Ubuntu only: if the script exits asking for a reboot, do so and re-run step 1.

# Step 2 — Node.js 22, npm packages, TF 2.15 GPU binary
bash setup-project.sh
```

That's it. Step 2 ends with a smoke test that prints the GPU matMul throughput.

---

## Usage

```bash
npm run benchmark   # GPU throughput across matMul, conv2d, softmax, dense model
npm run demo        # trains a small CNN on synthetic data end-to-end
```

### Using in your own code

```js
const { tf, init } = require('./src/index');

(async () => {
  await init();           // logs backend + GPU info

  const a = tf.randomNormal([1024, 1024]);
  const b = tf.randomNormal([1024, 1024]);
  const c = tf.matMul(a, b);

  // keep work on GPU — only read back final results
  const result = await c.data();
  tf.dispose([a, b, c]);
})();
```

`src/index.js` is self-contained. Copy it to any Node.js project that installs `@tensorflow/tfjs-node-gpu`.

---

## Benchmark output explained

```
  operation                           GPU only       + copy (WSL2)
  matMul  4096×4096               GPU:  23 ms  |  total: 115 ms  (5.8 TFLOPS)
```

| Column | Meaning |
|---|---|
| **GPU only** | Ops pipelined on GPU, one readback at the end. Reflects real inference throughput when the full forward pass stays on GPU. |
| **+ copy (WSL2)** | One GPU→CPU copy per iteration. WSL2 paravirtualises GPU memory, limiting bandwidth to ~600 MB/s (vs 32 GB/s on native Linux). Large tensor readbacks dominate this column. |

On **native Linux** both columns will be nearly identical.

---

## Performance (RTX 5070, WSL2)

| Operation | GPU-only throughput |
|---|---|
| matMul 4096² | ~6 TFLOPS |
| matMul 2048² | ~6.7 TFLOPS |
| conv2d ResNet-style | ~2 ms |
| softmax GPT vocab (512 × 50 257) | ~6 ms |

**Why not 209 TFLOPS?**  
TF 2.15 has no precompiled kernels for Blackwell (sm\_120). The CUDA 13 driver JIT-compiles CUDA 12 PTX at runtime — this works correctly but misses Tensor Core paths. Expect a large improvement when TF ships a binary with native sm\_120 support.

**WSL2 vs native:** GPU compute numbers are the same. The `+ copy` column will drop from ~100 ms to ~2 ms on native Linux due to full PCIe bandwidth.

---

## How it works

### Why TF 2.15 instead of the bundled TF 2.9.1?

`@tensorflow/tfjs-node-gpu` 4.22 bundles TensorFlow 2.9.1 compiled against CUDA 11. CUDA 11 PTX has no forward-compatible JIT path to sm\_120 — all compute silently fell back to CPU. `upgrade-libtf.sh` replaces the bundled `.so` with TF 2.15.0 (CUDA 12.3); its PTX can be JIT-compiled for Blackwell by the CUDA 13 driver.

### Key environment flags set by `src/index.js`

| Flag | Value | Effect |
|---|---|---|
| `TF_FORCE_GPU_ALLOW_GROWTH` | `true` | Allocates VRAM on demand |
| `TF_GPU_ALLOCATOR` | `cuda_malloc_async` | Reduces fragmentation OOM |
| `TF_XLA_FLAGS` | `--tf_xla_auto_jit=2` | Fuses op sequences with XLA |
| `CUDA_CACHE_MAXSIZE` | 2 GB | Persists PTX-compiled kernels |

---

## Project structure

```
.
├── src/
│   ├── index.js        # TF.js GPU entry point — import this in your project
│   ├── benchmark.js    # throughput benchmark (GPU-only vs with-copy)
│   └── demo.js         # CNN training demo on synthetic data
├── setup-system.sh     # Step 1: CUDA + cuDNN install (needs sudo)
├── setup-project.sh    # Step 2: Node.js + npm + TF 2.15 binary (no sudo)
└── upgrade-libtf.sh    # Downloads TF 2.15 GPU binary (called by step 2)
```

---

## Troubleshooting

**`nvidia-smi` not found in WSL2**  
Update the Windows NVIDIA driver to 530+ and restart WSL: `wsl --shutdown`.

**`libcudnn8` not found during `setup-system.sh`**  
The script adds the `ubuntu2204` CUDA repo specifically for cuDNN 8. Run `sudo apt-get update` and retry.

**`npm run benchmark` uses Windows npm instead of WSL node**  
Open a new WSL terminal (nvm writes to `.bashrc` during setup). Or run `setup-project.sh` first — it symlinks `node`/`npm` into `/usr/local/bin`.

**GPU memory shows `n/a` in init log**  
TF's `numBytesInGPU` counter is only populated when CUDA memory tracking is active. The GPU is still being used — verify with `nvidia-smi` during a benchmark run.
