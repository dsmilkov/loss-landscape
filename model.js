'use strict';

function makeFCModel() {
  const IMAGE_SIZE = 784;
  const LABELS_SIZE = 10;

  const w1 = tf.variable(tf.zeros([IMAGE_SIZE, 30]));
  const w2 = tf.variable(tf.zeros([30, 60]));
  const w3 = tf.variable(tf.zeros([60, LABELS_SIZE]));
  return xs => {
    const layer1 = xs.matMul(w1).relu();
    const layer2 = layer1.matMul(w2).relu();
    return layer2.matMul(w3);
  };
}

function makeConvModel() {
  function filterVar(shape) {
    return tf.variable(tf.zeros(shape));
  }

  function biasVar(size) {
    return tf.variable(tf.zeros([size]));
  }

  function fcVar(shape) {
    return tf.variable(tf.zeros(shape));
  }

  function conv2d(image, filter, bias) {
    return tf.conv2d(image, filter, [1, 1, 1, 1], 'same').add(bias);
  }

  function maxPool2x2(image) {
    return tf.maxPool(image, 2, 2, 'same');
  }

  const filter1 = filterVar([3, 3, 1, 4]);
  const filter1Bias = biasVar(4);

  const filter2 = filterVar([3, 3, 4, 8]);
  const filter2Bias = biasVar(8);

  const fc1W = fcVar([7 * 7 * 8, 10]);
  const fc1Bias = biasVar(10);

  return xs => {
    const image = xs.as4D(-1, 28, 28, 1);
    const conv1 = conv2d(image, filter1, filter1Bias).relu();
    const pool1 = maxPool2x2(conv1);

    const conv2 = conv2d(pool1, filter2, filter2Bias).relu();
    const pool2 = maxPool2x2(conv2);

    const fc1 = pool2.as2D(-1, 7 * 7 * 8).matMul(fc1W).add(fc1Bias);
    return fc1;
  };
}
