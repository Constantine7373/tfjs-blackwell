import { createRequire } from 'module';
import { dirname, join } from 'path';
import { existsSync } from 'fs';

const require = createRequire(import.meta.url);
const tfjsRoot = dirname(require.resolve('@tensorflow/tfjs-node-gpu/package.json'));
const depsLib  = join(tfjsRoot, 'deps/lib');

const cudaPaths = [
  depsLib,
  '/usr/local/cuda-12/lib64',
  '/usr/local/cuda-12.8/lib64',
  '/usr/local/cuda-11.8/lib64',
  '/usr/lib/wsl/lib',
].filter(existsSync);

const current = process.env.LD_LIBRARY_PATH ?? '';
const existing = new Set(current.split(':').filter(Boolean));
const toAdd = cudaPaths.filter(p => !existing.has(p)).join(':');
if (toAdd) process.env.LD_LIBRARY_PATH = `${toAdd}:${current}`.replace(/^:/, '');

process.env.TF_CPP_MIN_LOG_LEVEL    ??= '1';
process.env.TF_FORCE_GPU_ALLOW_GROWTH ??= 'true';
process.env.TF_GPU_ALLOCATOR          ??= 'cuda_malloc_async';
process.env.TF_XLA_FLAGS              ??= '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2';
process.env.CUDA_CACHE_DISABLE        ??= '0';
process.env.CUDA_CACHE_MAXSIZE        ??= '2147483648';
process.env.CUDA_VISIBLE_DEVICES      ??= '0';
