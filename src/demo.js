'use strict';
/**
 * Short demo: trains a tiny MNIST-style CNN on random data.
 * Verifies the full training pipeline runs on the RTX 5070.
 */

const { tf, init } = require('./index');

async function run() {
  await init();

  console.log('\n=== CNN training demo (synthetic data) ===\n');

  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
  }));
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

  // synthetic data: 2048 samples of 28×28 grayscale, 10 classes
  const xs = tf.randomNormal([2048, 28, 28, 1]);
  const ys = tf.oneHot(
    tf.cast(tf.randomUniform([2048], 0, 10), 'int32'),
    10,
  );

  console.log('\nTraining for 5 epochs on 2048 synthetic samples...\n');
  const t0 = Date.now();

  await model.fit(xs, ys, {
    epochs: 5,
    batchSize: 128,
    validationSplit: 0.1,
    callbacks: {
      onEpochEnd: (epoch, logs) =>
        console.log(
          `  Epoch ${epoch + 1}/5  loss=${logs.loss.toFixed(4)}  acc=${logs.acc?.toFixed(4) ?? '-'}`,
        ),
    },
  });

  console.log(`\nTotal training time: ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  console.log('Demo complete.\n');

  tf.dispose([xs, ys]);
}

run().catch(err => { console.error(err); process.exit(1); });
