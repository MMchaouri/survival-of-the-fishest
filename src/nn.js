function xavier(fanIn, fanOut) {
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  return (Math.random() * 2 - 1) * limit;
}

function tanh(x) {
  return Math.tanh(x);
}

export function createNN(layerSizes) {
  const weights = [];
  const biases = [];
  for (let l = 1; l < layerSizes.length; l++) {
    const fanIn = layerSizes[l - 1];
    const fanOut = layerSizes[l];
    const layerWeights = [];
    for (let n = 0; n < fanOut; n++) {
      const row = [];
      for (let i = 0; i < fanIn; i++) {
        row.push(xavier(fanIn, fanOut));
      }
      layerWeights.push(row);
    }
    weights.push(layerWeights);
    biases.push(Array.from({ length: fanOut }, () => (Math.random() * 2 - 1) * 0.5));
  }
  return { layerSizes: [...layerSizes], weights, biases };
}

export function forward(nn, inputs) {
  const activations = [inputs];
  let current = inputs;
  for (let l = 0; l < nn.weights.length; l++) {
    const layerWeights = nn.weights[l];
    const layerBiases = nn.biases[l];
    const next = layerWeights.map((row, n) => {
      let sum = layerBiases[n];
      for (let i = 0; i < row.length; i++) {
        sum += row[i] * current[i];
      }
      return tanh(sum);
    });
    activations.push(next);
    current = next;
  }
  return { output: current, activations };
}

export function cloneNN(nn) {
  return {
    layerSizes: [...nn.layerSizes],
    weights: nn.weights.map(layer => layer.map(row => [...row])),
    biases: nn.biases.map(layer => [...layer]),
  };
}

export function mutateNN(nn, rate, sigma) {
  const clone = cloneNN(nn);
  clone.weights = clone.weights.map(layer =>
    layer.map(row =>
      row.map(w => (Math.random() < rate ? w + (Math.random() * 2 - 1) * sigma : w))
    )
  );
  clone.biases = clone.biases.map(layer =>
    layer.map(b => (Math.random() < rate ? b + (Math.random() * 2 - 1) * sigma : b))
  );
  return clone;
}

export function crossoverNN(a, b) {
  const child = cloneNN(a);
  child.weights = a.weights.map((layer, l) =>
    layer.map((row, n) => row.map((w, i) => (Math.random() < 0.5 ? w : b.weights[l][n][i])))
  );
  child.biases = a.biases.map((layer, l) =>
    layer.map((bias, n) => (Math.random() < 0.5 ? bias : b.biases[l][n]))
  );
  return child;
}

export function resizeInputLayer(nn, newInputSize) {
  const oldInputSize = nn.layerSizes[0];
  const firstLayerWeights = nn.weights[0].map(row => {
    const newRow = row.slice(0, Math.min(oldInputSize, newInputSize));
    while (newRow.length < newInputSize) {
      newRow.push(xavier(newInputSize, nn.layerSizes[1]));
    }
    return newRow;
  });
  return {
    layerSizes: [newInputSize, ...nn.layerSizes.slice(1)],
    weights: [firstLayerWeights, ...nn.weights.slice(1).map(layer => layer.map(row => [...row]))],
    biases: nn.biases.map(layer => [...layer]),
  };
}
