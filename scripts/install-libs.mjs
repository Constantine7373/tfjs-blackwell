#!/usr/bin/env node
/**
 * postinstall — downloads the sm_120 libtensorflow build and installs it into
 * @tensorflow/tfjs-node-gpu/deps/lib so the Node.js addon can dlopen it.
 */

import { execSync, spawnSync } from 'child_process';
import { existsSync, readFileSync, writeFileSync, symlinkSync, unlinkSync, mkdirSync } from 'fs';
import { createRequire } from 'module';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const SM120_URL = 'https://github.com/Constantine7373/tfjs-blackwell/releases/download/v2.15.0-sm120/libtensorflow-sm120-2.15.0-linux-x86_64.tar.gz';
const PTX_URL   = 'https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.15.0.tar.gz';
const TF_VER    = '2.15.0';

// ── locate deps/lib inside @tensorflow/tfjs-node-gpu ──────────────────────
const require = createRequire(import.meta.url);
let tfjsRoot;
try {
  tfjsRoot = dirname(require.resolve('@tensorflow/tfjs-node-gpu/package.json'));
} catch {
  console.log('[install-libs] @tensorflow/tfjs-node-gpu not found — skipping.');
  process.exit(0);
}
const DEPS_LIB = join(tfjsRoot, 'deps/lib');
const MARKER   = join(DEPS_LIB, '.tf-version');

// ── detect GPU compute capability ─────────────────────────────────────────
function detectVariant() {
  try {
    const out = execSync(
      'nvidia-smi --query-gpu=compute_cap --format=csv,noheader',
      { encoding: 'utf8', timeout: 8000 },
    ).trim().split('\n')[0].trim();
    if (parseInt(out.split('.')[0], 10) >= 12) return 'sm120';
  } catch { /* no GPU or nvidia-smi absent */ }
  return 'ptx';
}

const variant  = detectVariant();
const expected = `${TF_VER}-${variant}`;

// ── idempotency check ──────────────────────────────────────────────────────
if (existsSync(MARKER) && readFileSync(MARKER, 'utf8').trim() === expected) {
  console.log(`[install-libs] libtensorflow ${TF_VER} (${variant}) already installed — skipping.`);
  process.exit(0);
}

console.log(`[install-libs] GPU: ${variant === 'sm120' ? 'sm_120 Blackwell (native kernels)' : 'PTX fallback (CUDA 12 JIT)'}`);

// ── download ───────────────────────────────────────────────────────────────
function download(url, dest) {
  const hasWget = spawnSync('wget', ['--version'], { stdio: 'ignore' }).status === 0;
  if (hasWget) {
    const r = spawnSync('wget', ['-q', '--show-progress', '-c', url, '-O', dest], { stdio: 'inherit' });
    if (r.status !== 0) throw new Error(`wget failed (exit ${r.status})`);
  } else {
    const r = spawnSync('curl', ['-L', '-#', '-C', '-', url, '-o', dest], { stdio: 'inherit' });
    if (r.status !== 0) throw new Error(`curl failed (exit ${r.status})`);
  }
}

const url    = variant === 'sm120' ? SM120_URL : PTX_URL;
const tmpFile = `/tmp/libtensorflow-${variant}-${TF_VER}.tar.gz`;

try {
  if (existsSync(tmpFile)) {
    console.log('[install-libs] Using cached tarball.');
  } else {
    download(url, tmpFile);
  }
} catch (e) {
  if (variant === 'sm120') {
    console.warn(`[install-libs] sm_120 download failed: ${e.message}`);
    console.warn('[install-libs] Falling back to PTX build...');
    const ptxTmp = `/tmp/libtensorflow-ptx-${TF_VER}.tar.gz`;
    if (!existsSync(ptxTmp)) download(PTX_URL, ptxTmp);
    extractAndLink(ptxTmp, 'ptx');
  } else {
    console.error(`[install-libs] Download failed: ${e.message}`);
    process.exit(1);
  }
}

extractAndLink(tmpFile, variant);

// ── extract + symlink ──────────────────────────────────────────────────────
function extractAndLink(tarball, resolvedVariant) {
  mkdirSync(DEPS_LIB, { recursive: true });

  const probe  = spawnSync('tar', ['-tzf', tarball], { encoding: 'utf8' });
  const files  = probe.stdout.trim().split('\n');
  const strip  = files.some(f => f.startsWith('lib/libtensorflow')) ? ['--strip-components=1'] : [];

  const r = spawnSync(
    'tar', ['-xzf', tarball, '-C', DEPS_LIB, ...strip, '--wildcards', 'libtensorflow*.so*'],
    { stdio: 'inherit' },
  );
  if (r.status !== 0) {
    const r2 = spawnSync('tar', ['-xzf', tarball, '-C', DEPS_LIB, ...strip], { stdio: 'inherit' });
    if (r2.status !== 0) throw new Error('tar extraction failed');
  }

  const links = [
    ['libtensorflow.so',             `libtensorflow.so.${TF_VER}`],
    ['libtensorflow.so.2',           `libtensorflow.so.${TF_VER}`],
    ['libtensorflow_framework.so',   `libtensorflow_framework.so.${TF_VER}`],
    ['libtensorflow_framework.so.2', `libtensorflow_framework.so.${TF_VER}`],
  ];

  for (const [link, target] of links) {
    const linkPath = join(DEPS_LIB, link);
    if (existsSync(linkPath)) { try { unlinkSync(linkPath); } catch { /* ignore */ } }
    if (existsSync(join(DEPS_LIB, target))) symlinkSync(target, linkPath);
  }

  writeFileSync(MARKER, `${TF_VER}-${resolvedVariant}`);
  console.log(`[install-libs] Done. libtensorflow ${TF_VER} (${resolvedVariant}) installed.`);
}
