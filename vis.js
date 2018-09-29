'use strict';

// Hyperparameters.
const BATCH_SIZE = 64;
const TEST_BATCH_SIZE = 64*10;

// Data constants.
const LANDSCAPE_STEPS_PER_DIR = 10;
const NUM_TRAIN_CHARTS = 8;
const NUM_BATCH_PER_TRAIN = 1;
const LEARNING_RATE = 0.1;
const MAX_ALPHA = 0.25;

const WeightInit = {
  FAN_IN: 'fan-in',
  FAN_OUT: 'fan-out',
  UNIT: 'unit'
};

class Landscape {
  init(data) {
    this.dirMap = {};
    this.varMap = tf.ENV.engine.registeredVariables;
    this.data = data;
    this.testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
    this.iter = 0;
    this.randDirs = [];
    this.optimizer = tf.train.sgd(+LEARNING_RATE);
    this.alphas = [];
    for (let i = 0; i <= LANDSCAPE_STEPS_PER_DIR; i++) {
      this.alphas.push(
          tf.scalar(2 * MAX_ALPHA * (i / LANDSCAPE_STEPS_PER_DIR) - MAX_ALPHA));
    }
  }

  setModel(modelFactory) {
    tf.disposeVariables();
    this.modelFn = tf.tidy(() => modelFactory());
    this.reinitWeights();
    this.data.reset();
    tf.dispose(this.dirMap);
    this.dirMap = {};
    this.iter = 0;
    tf.dispose(this.randDirs);
    this.randDirs = [this.genUnitNormDirs(), this.genUnitNormDirs()];
  }

  async train() {
    const start = performance.now();
    let cost;
    for (let i = 0; i < NUM_BATCH_PER_TRAIN; i++) {
      cost = this.optimizer.minimize(() => {
        const batch = this.data.nextTrainBatch(BATCH_SIZE);
        return this.loss(batch.labels, this.modelFn(batch.xs));
      }, i === NUM_BATCH_PER_TRAIN - 1 /* returnCost */);
      this.iter++;
    }
    console.log('Training took', performance.now() - start, 'ms');
    cost.dispose();
    return this.iter;
  }

  reinitWeights() {
    tf.tidy(() => {
      for (const varName in this.varMap) {
        const v = this.varMap[varName];
        const lastDim = v.shape[v.shape.length - 1];
        const w = v.as2D(-1, lastDim);
        let std = 1;
        switch (this.weightInit) {
          case WeightInit.FAN_IN:
            std = Math.sqrt(2 / w.shape[0]);
            break;
          case WeightInit.FAN_OUT:
            std = Math.sqrt(2 / w.shape[1]);
            break;
          case WeightInit.UNIT:
            std = 1;
            break;
          default:
            throw new Error(`Unknown weight init ${this.weightInit}`);
        }
        v.assign(tf.truncatedNormal(v.shape, 0, std));
      }
    });
  }

  async computeLandscape() {
    const [losses, vs] = tf.tidy(() => {
      const vs = {};
      const losses = [];
      const dirs1 = this.genDirs(0);
      const dirs2 = this.genDirs(1);
      for (const varName in this.varMap) {
        vs[varName] = this.varMap[varName].flatten();
      }
      for (let i = 0; i <= LANDSCAPE_STEPS_PER_DIR; i++) {
        for (let j = 0; j <= LANDSCAPE_STEPS_PER_DIR; j++) {
          tf.tidy(() => {
            for (const varName in this.varMap) {
              const dir1 = dirs1[varName];
              const dir2 = dirs2[varName];
              const flatV = vs[varName];
              const v = this.varMap[varName];
              const alpha1 = this.alphas[i];
              const alpha2 = this.alphas[j];
              const newFlatV =
                  flatV.add(alpha1.mul(dir1)).add(alpha2.mul(dir2));
              v.assign(newFlatV.reshapeAs(v));
            }
            const loss = this.loss(
                this.testBatch.labels, this.modelFn(this.testBatch.xs));
            losses.push(loss.get());
          });
        }
      }
      return [losses, vs];
    });

    const lossVals = await Promise.all(losses);
    const matrix = [];
    for (let i = 0; i <= LANDSCAPE_STEPS_PER_DIR; i++) {
      const row = [];
      for (let j = 0; j <= LANDSCAPE_STEPS_PER_DIR; j++) {
        const index = i * (LANDSCAPE_STEPS_PER_DIR + 1) + j;
        row.push(lossVals[index]);
      }
      matrix.push(row);
    }

    // Reset the weights.
    tf.tidy(() => {
      for (const varName in this.varMap) {
        const variable = this.varMap[varName];
        variable.assign(vs[varName].reshape(variable.shape));
        vs[varName].dispose();
      }
    });
    return matrix;
  }

  dispose() {
    this.optimizer.dispose();
    tf.dispose(this.alphas);
    tf.dispose(this.randDirs);
    tf.dispose(this.testBatch);
    tf.dispose(this.dirMap);
    tf.disposeVariables();
  }

  loss(labels, logits) {
    return tf.losses.softmaxCrossEntropy(labels, logits).mean();
  }

  genUnitNormDirs() {
    const randDirs = {};
    // For each variable, it generates a random direction. Assume the shape of
    // the variable is [a, b, out]. The random direction for that variable
    // has the shape [a*b, out], representing 'out' random vectors of unit norm.
    for (const varName in this.varMap) {
      randDirs[varName] = tf.tidy(() => {
        const v = this.varMap[varName];
        const lastDim = v.shape[v.shape.length - 1];
        const w = v.as2D(-1, lastDim);
        const dir = tf.randomNormal(w.shape, 0, 1, 'float32');
        const dirNorm = dir.norm('euclidean', 0);
        return dir.div(dirNorm);
      });
    }
    return randDirs;
  }

  genDirs(index) {
    if (index in this.dirMap) {
      return this.dirMap[index];
    }
    const dirs = {};
    for (const varName in this.varMap) {
      dirs[varName] = tf.tidy(() => {
        const v = this.varMap[varName];
        const lastDim = v.shape[v.shape.length - 1];
        const w = v.as2D(-1, lastDim);
        const wNorm = w.norm('euclidean', 0);
        const randDir = this.randDirs[index][varName];
        return tf.keep(randDir.mul(wNorm).flatten());
      });
    }
    this.dirMap[index] = dirs;
    return dirs;
  }
}

async function computeCharts(modelFn, landscape, weightInit) {
  const res = [];
  landscape.weightInit = weightInit;
  landscape.setModel(modelFn);
  let iter = 0;
  for (let i = 0; i < NUM_TRAIN_CHARTS; i++) {
    if (i > 0) {
      iter = await landscape.train();
    }
    const zData = await landscape.computeLandscape();
    const mid = Math.floor(zData.length / 2);
    console.log('iter', iter, 'loss on test batch', zData[mid][mid]);
    res.push({zData, iter});
  }
  return res;
}

function plotCharts(chartsData, container) {
  const innerDivs = chartsData.map(({zData, iter}) => {
    const data = [{
      z: zData,
      type: 'contour',
      // colorscale: [
      //   [0, 'rgb(245,147,34)'], [0.5, 'rgb(232, 234, 235)'],
      //   [1, 'rgb(8,119,189)']
      // ],
      //zmin: min,
      //zmax: max,
      showscale: false,
      ncontours: 20
    }];
    const layout = {
      autosize: false,
      showlegend: false,
      width: 100,
      height: 100,
      margin: {l: 0, r: 0, b: 0, t: 0}
    };
    const plotContainer = document.createElement('div');
    const chartContainer = document.createElement('div');
    chartContainer.style.margin = '0 5px';
    chartContainer.appendChild(plotContainer);
    const iterDiv = document.createElement('div');
    iterDiv.textContent = `Iteration: ${iter}`;
    chartContainer.appendChild(iterDiv);
    Plotly.newPlot(plotContainer, data, layout, {displayModeBar: false});
    return chartContainer;
  });
  container.style.display = 'flex';
  innerDivs.forEach(d => container.appendChild(d));
}
