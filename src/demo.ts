/**
 * CNN training demo — trains a small conv-net on synthetic data.
 * Verifies the full forward + backward pass runs on the RTX 5070.
 */

import type { Logs } from '@tensorflow/tfjs-node-gpu';
import { tf, init } from './index.js';

async function run(): Promise<void> {
  await init();
  console.log('\n=== CNN training demo (synthetic data) ===\n');

  const model = tf.sequential();
  model.add(tf.layers.conv2d({ inputShape: [28, 28, 1], filters: 32, kernelSize: 3, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.4 }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  model.summary();

  const xs = tf.randomNormal([2048, 28, 28, 1]);
  const ys = tf.oneHot(tf.cast(tf.randomUniform([2048], 0, 10), 'int32'), 10);

  console.log('\nTraining 5 epochs on 2048 synthetic samples...\n');
  const t0 = Date.now();

  await model.fit(xs, ys, {
    epochs: 5,
    batchSize: 128,
    validationSplit: 0.1,
    callbacks: {
      onEpochEnd: (epoch: number, logs?: Logs) =>
        console.log(`  Epoch ${epoch + 1}/5  loss=${logs?.['loss']?.toFixed(4) ?? '-'}  acc=${logs?.['acc']?.toFixed(4) ?? '-'}`),
    },
  });

  console.log(`\nTotal: ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  tf.dispose([xs, ys]);
}

run().catch(err => { console.error(err); process.exit(1); });
