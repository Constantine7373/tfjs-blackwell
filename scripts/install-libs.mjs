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

// ── helpers ────────────────────────────────────────────────────────────────
function isValidTarball(file) {
  return spawnSync('tar', ['-tzf', file], { stdio: 'ignore' }).status === 0;
}

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

function ensureTarball(url, dest) {
  if (existsSync(dest) && isValidTarball(dest)) {
    console.log('[install-libs] Using cached tarball.');
    return;
  }
  if (existsSync(dest)) {
    console.log('[install-libs] Cached tarball is corrupted — re-downloading...');
    unlinkSync(dest);
  }
  download(url, dest);
}

// ── resolve which tarball to use ───────────────────────────────────────────
const sm120Tmp = `/tmp/libtensorflow-sm120-${TF_VER}.tar.gz`;
const ptxTmp   = `/tmp/libtensorflow-ptx-${TF_VER}.tar.gz`;

let resolvedTmp     = variant === 'sm120' ? sm120Tmp : ptxTmp;
let resolvedUrl     = variant === 'sm120' ? SM120_URL : PTX_URL;
let resolvedVariant = variant;

try {
  ensureTarball(resolvedUrl, resolvedTmp);
} catch (e) {
  if (variant !== 'sm120') {
    console.error(`[install-libs] Download failed: ${e.message}`);
    process.exit(1);
  }
  console.warn(`[install-libs] sm_120 download failed: ${e.message}`);
  console.warn('[install-libs] Falling back to PTX build...');
  try {
    ensureTarball(PTX_URL, ptxTmp);
  } catch (e2) {
    console.error(`[install-libs] PTX fallback also failed: ${e2.message}`);
    process.exit(1);
  }
  resolvedTmp     = ptxTmp;
  resolvedVariant = 'ptx';
}

// ── extract + symlink ──────────────────────────────────────────────────────
function extractAndLink(tarball, resolvedVariant) {
  mkdirSync(DEPS_LIB, { recursive: true });

  const probe = spawnSync('tar', ['-tzf', tarball], { encoding: 'utf8' });
  const strip = probe.stdout.split('\n').some(f => f.startsWith('lib/libtensorflow'))
    ? ['--strip-components=1']
    : [];

  const r = spawnSync('tar', ['-xzf', tarball, '-C', DEPS_LIB, ...strip], { stdio: 'inherit' });
  if (r.status !== 0) throw new Error('tar extraction failed');

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

extractAndLink(resolvedTmp, resolvedVariant);
