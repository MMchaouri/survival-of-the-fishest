import test from 'node:test';
import assert from 'node:assert/strict';
import { createNN, forward, cloneNN, mutateNN, crossoverNN, resizeInputLayer } from '../src/nn.js';

test('createNN builds correct weight/bias shapes', () => {
  const nn = createNN([3, 4, 2]);
  assert.equal(nn.weights.length, 2);
  assert.equal(nn.weights[0].length, 4);
  assert.equal(nn.weights[0][0].length, 3);
  assert.equal(nn.weights[1].length, 2);
  assert.equal(nn.weights[1][0].length, 4);
  assert.equal(nn.biases.length, 2);
  assert.equal(nn.biases[0].length, 4);
  assert.equal(nn.biases[1].length, 2);
});

test('forward produces output of correct length and activations per layer', () => {
  const nn = createNN([3, 4, 2]);
  const { output, activations } = forward(nn, [0.1, 0.2, 0.3]);
  assert.equal(output.length, 2);
  assert.equal(activations.length, 3); // input + hidden + output
  assert.equal(activations[0].length, 3);
  assert.equal(activations[1].length, 4);
  assert.equal(activations[2].length, 2);
  output.forEach(v => assert.ok(v >= -1 && v <= 1)); // tanh-bounded
});

test('forward is deterministic for same weights and inputs', () => {
  const nn = createNN([2, 3, 1]);
  const a = forward(nn, [0.5, -0.5]);
  const b = forward(nn, [0.5, -0.5]);
  assert.deepEqual(a.output, b.output);
});

test('cloneNN produces a deep copy, not a reference', () => {
  const nn = createNN([2, 2]);
  const clone = cloneNN(nn);
  clone.weights[0][0][0] = 999;
  assert.notEqual(nn.weights[0][0][0], 999);
});

test('mutateNN changes some weights but keeps shape, does not mutate original', () => {
  const nn = createNN([2, 3, 1]);
  const before = JSON.stringify(nn.weights);
  const mutated = mutateNN(nn, 1.0, 0.5); // rate=1.0 forces every weight to mutate
  assert.equal(JSON.stringify(nn.weights), before); // original untouched
  assert.notEqual(JSON.stringify(mutated.weights), before);
  assert.deepEqual(mutated.layerSizes, nn.layerSizes);
});

test('mutateNN with rate 0 returns identical weights', () => {
  const nn = createNN([2, 3, 1]);
  const mutated = mutateNN(nn, 0, 0.5);
  assert.deepEqual(mutated.weights, nn.weights);
  assert.deepEqual(mutated.biases, nn.biases);
});

test('crossoverNN produces weights each drawn from parent a or b', () => {
  const a = createNN([2, 2]);
  const b = createNN([2, 2]);
  // force distinguishable values
  a.weights[0] = [[1, 1], [1, 1]];
  b.weights[0] = [[2, 2], [2, 2]];
  const child = crossoverNN(a, b);
  child.weights[0].flat().forEach(w => assert.ok(w === 1 || w === 2));
});

test('resizeInputLayer keeps existing input weights and adds new ones', () => {
  const nn = createNN([2, 3, 1]);
  nn.weights[0] = [[1, 2], [3, 4], [5, 6]]; // 3 hidden neurons x 2 inputs
  const resized = resizeInputLayer(nn, 4);
  assert.equal(resized.layerSizes[0], 4);
  assert.equal(resized.weights[0].length, 3); // hidden neuron count unchanged
  resized.weights[0].forEach((row, i) => {
    assert.equal(row.length, 4);
    assert.equal(row[0], nn.weights[0][i][0]); // first input weight preserved
    assert.equal(row[1], nn.weights[0][i][1]); // second input weight preserved
    assert.equal(typeof row[2], 'number'); // new input weight exists
    assert.equal(typeof row[3], 'number');
  });
});

test('resizeInputLayer truncates when shrinking input size', () => {
  const nn = createNN([4, 2]);
  nn.weights[0] = [[1, 2, 3, 4], [5, 6, 7, 8]];
  const resized = resizeInputLayer(nn, 2);
  assert.equal(resized.layerSizes[0], 2);
  assert.deepEqual(resized.weights[0], [[1, 2], [5, 6]]);
});
