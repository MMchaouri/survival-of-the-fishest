# Shark Tank NN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-based aquarium simulation where a shark and a fish population, each controlled by neuroevolved feedforward NNs, coexist across 6 stages of increasing fish sensory input.

**Architecture:** Plain JS ES modules, no build step, no runtime dependencies. Pure-logic modules (`nn.js`, `genetic.js`, `stages.js`, `sim.js`) are unit-tested with Node's built-in test runner. Rendering modules (`render.js`, `nnviz.js`, `ui.js`) are visual and verified manually in-browser. `main.js` wires everything into a `requestAnimationFrame` loop.

**Tech Stack:** Vanilla JS (ES modules), Canvas 2D, Node built-in `node:test` + `node:assert` for unit tests. No TF.js, no bundler, no npm dependencies.

## Global Constraints

- No build step: `index.html` loads `src/main.js` as `<script type="module">`, all imports are relative ES module paths.
- No runtime dependencies (per spec amendment: plain JS arrays instead of TF.js for NN math).
- Fish population: 30. Shark population: 8. (from spec)
- Fish GA: elitism top 3, fitness-weighted roulette crossover (2 parents), gaussian mutation ~10% rate, small sigma. Same scheme for shark. (from spec)
- Episode: fixed max duration ~20s sim time, or all fish eaten, whichever first. Eaten fish freeze fitness, no respawn. (from spec)
- Stage advance is manual (button), not automatic. (from spec)
- 6 stages, fish input schema per stage table in spec section "Stage Ladder". Shark senses fixed from stage 1, only adds nearest-fish velocity at stage 6. (from spec)
- Fish output: `[turn, thrust]`. Hidden layers: 1 until stage 6, 2 from stage 6. (from spec)

---

## File Structure

```
survival-of-the-fishest/
  package.json
  index.html
  src/
    nn.js         - feedforward NN: create/forward/clone/mutate/crossover/resize
    genetic.js    - generic GA engine (population evolution, used by both fish & shark)
    stages.js     - stage schema table + input-vector builders for fish & shark
    sim.js        - episode state, physics tick, collisions, fitness accumulation
    render.js     - procedural canvas art for fish/shark/aquarium
    nnviz.js      - live node-link diagram for a selected agent's NN
    ui.js         - HUD, Next Stage button, speed control, agent-select click handling
    main.js       - bootstrap: populations, episode loop, wires render+nnviz+ui
  test/
    nn.test.js
    genetic.test.js
    stages.test.js
    sim.test.js
```

---

### Task 1: Project scaffold

**Files:**
- Create: `survival-of-the-fishest/package.json`
- Create: `survival-of-the-fishest/index.html`

**Interfaces:**
- Produces: `npm test` runs `node --test test/`. `index.html` is the entry point loaded in-browser for all later manual-verification steps.

- [ ] **Step 1: Create `package.json`**

```json
{
  "name": "survival-of-the-fishest",
  "version": "1.0.0",
  "type": "module",
  "private": true,
  "scripts": {
    "test": "node --test test/"
  }
}
```

- [ ] **Step 2: Create `index.html` skeleton**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Shark Tank NN</title>
  <style>
    body { margin: 0; background: #071e28; font-family: ui-monospace, monospace; color: #eaf3f2; }
    .layout { display: flex; gap: 12px; padding: 12px; }
    #tank { background: linear-gradient(180deg, #0d3b4a, #071e28); border-radius: 8px; }
    #nnviz { background: #0a1a1f; border-radius: 8px; }
    #hud { display: flex; gap: 16px; padding: 8px 12px; font-size: 13px; }
  </style>
</head>
<body>
  <div id="hud"></div>
  <div class="layout">
    <canvas id="tank" width="1000" height="600"></canvas>
    <canvas id="nnviz" width="360" height="600"></canvas>
  </div>
  <script type="module" src="src/main.js"></script>
</body>
</html>
```

- [ ] **Step 3: Verify test runner works with zero tests**

Run: `cd survival-of-the-fishest && npm test`
Expected: `# tests 0` output, exit code 0 (no test files exist yet — this just confirms the command doesn't error).

- [ ] **Step 4: Commit**

```bash
git init
git add package.json index.html
git commit -m "chore: scaffold survival-of-the-fishest project"
```

---

### Task 2: NN core (`nn.js`)

**Files:**
- Create: `survival-of-the-fishest/src/nn.js`
- Test: `survival-of-the-fishest/test/nn.test.js`

**Interfaces:**
- Produces:
  - `createNN(layerSizes: number[]): NN` where `NN = { layerSizes: number[], weights: number[][][], biases: number[][] }`
  - `forward(nn: NN, inputs: number[]): { output: number[], activations: number[][] }`
  - `cloneNN(nn: NN): NN`
  - `mutateNN(nn: NN, rate: number, sigma: number): NN` (returns new NN, does not mutate input)
  - `crossoverNN(a: NN, b: NN): NN` (returns new NN, uniform per-weight crossover)
  - `resizeInputLayer(nn: NN, newInputSize: number): NN` (returns new NN; keeps weight rows for the first `min(old, new)` inputs, Xavier-inits any new input weight rows)

- [ ] **Step 1: Write failing tests**

```javascript
// test/nn.test.js
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd survival-of-the-fishest && npm test`
Expected: FAIL — `Cannot find module '../src/nn.js'`

- [ ] **Step 3: Implement `src/nn.js`**

```javascript
// src/nn.js

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
    biases.push(new Array(fanOut).fill(0));
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
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd survival-of-the-fishest && npm test`
Expected: all 9 tests pass, `# pass 9`

- [ ] **Step 5: Commit**

```bash
git add src/nn.js test/nn.test.js
git commit -m "feat: add feedforward NN core with GA-friendly weight ops"
```

---

### Task 3: Genetic algorithm engine (`genetic.js`)

**Files:**
- Create: `survival-of-the-fishest/src/genetic.js`
- Test: `survival-of-the-fishest/test/genetic.test.js`

**Interfaces:**
- Consumes: `cloneNN`, `mutateNN`, `crossoverNN` from `nn.js` (Task 2) — passed in via options, `genetic.js` itself stays generic and does not import `nn.js` directly.
- Produces: `evolvePopulation(population: T[], fitnesses: number[], options): T[]` where `options = { eliteCount: number, mutationRate: number, mutationSigma: number, cloneFn: (T) => T, crossoverFn: (T, T) => T, mutateFn: (T, rate, sigma) => T }`. Also exports `selectParent(population: T[], fitnesses: number[]): T` (fitness-weighted roulette).

- [ ] **Step 1: Write failing tests**

```javascript
// test/genetic.test.js
import test from 'node:test';
import assert from 'node:assert/strict';
import { selectParent, evolvePopulation } from '../src/genetic.js';

test('selectParent never picks from a population of all-zero fitness without crashing', () => {
  const pop = ['a', 'b', 'c'];
  const fitnesses = [0, 0, 0];
  const picked = selectParent(pop, fitnesses);
  assert.ok(pop.includes(picked));
});

test('selectParent heavily favors the highest-fitness individual over many draws', () => {
  const pop = ['low', 'high'];
  const fitnesses = [1, 1000];
  let highCount = 0;
  for (let i = 0; i < 200; i++) {
    if (selectParent(pop, fitnesses) === 'high') highCount++;
  }
  assert.ok(highCount > 150); // strongly biased toward 'high'
});

test('evolvePopulation keeps eliteCount individuals unchanged (by identity value)', () => {
  const pop = [10, 20, 30, 40, 5];
  const fitnesses = [100, 90, 80, 1, 1]; // pop[0..2] are the top 3
  const next = evolvePopulation(pop, fitnesses, {
    eliteCount: 3,
    mutationRate: 1,
    mutationSigma: 1,
    cloneFn: x => x,
    crossoverFn: (a, b) => a + b,
    mutateFn: x => x + 1000,
  });
  assert.equal(next.length, 5);
  assert.deepEqual(next.slice(0, 3).sort((a, b) => a - b), [10, 20, 30]);
});

test('evolvePopulation fills remaining slots via crossover + mutate', () => {
  const pop = [1, 2];
  const fitnesses = [1, 1];
  const next = evolvePopulation(pop, fitnesses, {
    eliteCount: 0,
    mutationRate: 1,
    mutationSigma: 1,
    cloneFn: x => x,
    crossoverFn: (a, b) => a + b,
    mutateFn: x => x + 100,
  });
  assert.equal(next.length, 2);
  next.forEach(v => assert.ok(v >= 100)); // every offspring passed through mutateFn
});
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd survival-of-the-fishest && npm test`
Expected: FAIL — `Cannot find module '../src/genetic.js'`

- [ ] **Step 3: Implement `src/genetic.js`**

```javascript
// src/genetic.js

export function selectParent(population, fitnesses) {
  const shifted = fitnesses.map(f => Math.max(f, 0));
  const total = shifted.reduce((a, b) => a + b, 0);
  if (total === 0) {
    return population[Math.floor(Math.random() * population.length)];
  }
  let r = Math.random() * total;
  for (let i = 0; i < population.length; i++) {
    r -= shifted[i];
    if (r <= 0) return population[i];
  }
  return population[population.length - 1];
}

export function evolvePopulation(population, fitnesses, options) {
  const { eliteCount, mutationRate, mutationSigma, cloneFn, crossoverFn, mutateFn } = options;
  const ranked = population
    .map((individual, i) => ({ individual, fitness: fitnesses[i] }))
    .sort((a, b) => b.fitness - a.fitness);

  const next = ranked.slice(0, eliteCount).map(r => cloneFn(r.individual));

  while (next.length < population.length) {
    const parentA = selectParent(population, fitnesses);
    const parentB = selectParent(population, fitnesses);
    const child = crossoverFn(parentA, parentB);
    next.push(mutateFn(child, mutationRate, mutationSigma));
  }

  return next;
}
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd survival-of-the-fishest && npm test`
Expected: all 4 tests pass (13 total across files), `# pass 13`

- [ ] **Step 5: Commit**

```bash
git add src/genetic.js test/genetic.test.js
git commit -m "feat: add generic GA engine (selection, elitism, crossover, mutation)"
```

---

### Task 4: Stage schema + input builders (`stages.js`)

**Files:**
- Create: `survival-of-the-fishest/src/stages.js`
- Test: `survival-of-the-fishest/test/stages.test.js`

**Interfaces:**
- Produces:
  - `STAGES: Stage[]` where `Stage = { id: number, name: string, fishInputSize: number, fishHiddenLayers: number[], description: string }`, 6 entries.
  - `buildFishInputs(stageId: number, fish: FishState, tankBounds: {width:number,height:number}, shark: SharkState, nearbyFish: FishState[]): number[]` where `FishState = {x,y,vx,vy}`, `SharkState = {x,y,vx,vy}`. Output length always equals `STAGES[stageId-1].fishInputSize`.
  - `buildSharkInputs(stageId: number, shark: SharkState, nearestFish: FishState): number[]`. Length is 2 for stages 1-5, 4 for stage 6 (adds nearest fish vx, vy).
- Consumes: nothing (pure data + pure functions).

- [ ] **Step 1: Write failing tests**

```javascript
// test/stages.test.js
import test from 'node:test';
import assert from 'node:assert/strict';
import { STAGES, buildFishInputs, buildSharkInputs } from '../src/stages.js';

test('STAGES has exactly 6 stages with increasing fishInputSize', () => {
  assert.equal(STAGES.length, 6);
  for (let i = 1; i < STAGES.length; i++) {
    assert.ok(STAGES[i].fishInputSize >= STAGES[i - 1].fishInputSize);
  }
  assert.equal(STAGES[0].fishInputSize, 0);
});

test('stage 6 uses 2 hidden layers, earlier stages use 1', () => {
  assert.equal(STAGES[5].fishHiddenLayers.length, 2);
  for (let i = 0; i < 5; i++) {
    assert.equal(STAGES[i].fishHiddenLayers.length, 1);
  }
});

test('buildFishInputs returns empty array for stage 1', () => {
  const inputs = buildFishInputs(1, { x: 0, y: 0, vx: 0, vy: 0 }, { width: 100, height: 100 }, { x: 50, y: 50, vx: 0, vy: 0 }, []);
  assert.deepEqual(inputs, []);
});

test('buildFishInputs returns wall distances for stage 2, length matches schema', () => {
  const bounds = { width: 100, height: 100 };
  const inputs = buildFishInputs(2, { x: 10, y: 20, vx: 0, vy: 0 }, bounds, { x: 50, y: 50, vx: 0, vy: 0 }, []);
  assert.equal(inputs.length, STAGES[1].fishInputSize);
  assert.deepEqual(inputs, [10, 90, 20, 80]); // dist to left, right, top, bottom walls
});

test('buildFishInputs length matches schema at every stage', () => {
  const bounds = { width: 200, height: 200 };
  const fish = { x: 50, y: 50, vx: 1, vy: 1 };
  const shark = { x: 100, y: 100, vx: 0.5, vy: 0.5 };
  const nearby = [{ x: 60, y: 60, vx: 0, vy: 0 }, { x: 70, y: 70, vx: 0, vy: 0 }];
  for (const stage of STAGES) {
    const inputs = buildFishInputs(stage.id, fish, bounds, shark, nearby);
    assert.equal(inputs.length, stage.fishInputSize, `stage ${stage.id} input length mismatch`);
  }
});

test('buildSharkInputs returns 2 inputs before stage 6, 4 at stage 6', () => {
  const shark = { x: 0, y: 0, vx: 0, vy: 0 };
  const nearestFish = { x: 10, y: 10, vx: 1, vy: -1 };
  for (let id = 1; id <= 5; id++) {
    assert.equal(buildSharkInputs(id, shark, nearestFish).length, 2);
  }
  assert.equal(buildSharkInputs(6, shark, nearestFish).length, 4);
});
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd survival-of-the-fishest && npm test`
Expected: FAIL — `Cannot find module '../src/stages.js'`

- [ ] **Step 3: Implement `src/stages.js`**

```javascript
// src/stages.js

export const STAGES = [
  { id: 1, name: 'random', fishInputSize: 0, fishHiddenLayers: [8],
    description: 'No inputs. Fish move randomly.' },
  { id: 2, name: 'walls', fishInputSize: 4, fishHiddenLayers: [8],
    description: 'Distance to 4 walls. Fish avoid walls/corners.' },
  { id: 3, name: 'shark-distance', fishInputSize: 5, fishHiddenLayers: [8],
    description: 'Adds distance to shark.' },
  { id: 4, name: 'shark-angle', fishInputSize: 6, fishHiddenLayers: [8],
    description: 'Adds angle to shark. Directional fleeing possible.' },
  { id: 5, name: 'schooling', fishInputSize: 8, fishHiddenLayers: [8],
    description: 'Adds avg position/heading of nearby fish. Schooling emerges.' },
  { id: 6, name: 'predictive', fishInputSize: 10, fishHiddenLayers: [8, 8],
    description: 'Adds shark velocity. Deeper net. Predictive evasion.' },
];

function wallDistances(fish, bounds) {
  return [fish.x, bounds.width - fish.x, fish.y, bounds.height - fish.y];
}

function distanceTo(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function angleTo(a, b) {
  return Math.atan2(b.y - a.y, b.x - a.x);
}

function avgNearby(nearbyFish) {
  if (nearbyFish.length === 0) return { x: 0, y: 0, vx: 0, vy: 0 };
  const sum = nearbyFish.reduce(
    (acc, f) => ({ x: acc.x + f.x, y: acc.y + f.y, vx: acc.vx + f.vx, vy: acc.vy + f.vy }),
    { x: 0, y: 0, vx: 0, vy: 0 }
  );
  const n = nearbyFish.length;
  return { x: sum.x / n, y: sum.y / n, vx: sum.vx / n, vy: sum.vy / n };
}

export function buildFishInputs(stageId, fish, bounds, shark, nearbyFish) {
  if (stageId === 1) return [];

  const inputs = [...wallDistances(fish, bounds)];
  if (stageId === 2) return inputs;

  inputs.push(distanceTo(fish, shark));
  if (stageId === 3) return inputs;

  inputs.push(angleTo(fish, shark));
  if (stageId === 4) return inputs;

  const avg = avgNearby(nearbyFish);
  inputs.push(avg.x - fish.x, avg.y - fish.y);
  if (stageId === 5) return inputs;

  inputs.push(shark.vx, shark.vy);
  return inputs; // stage 6
}

export function buildSharkInputs(stageId, shark, nearestFish) {
  const inputs = [distanceTo(shark, nearestFish), angleTo(shark, nearestFish)];
  if (stageId === 6) {
    inputs.push(nearestFish.vx, nearestFish.vy);
  }
  return inputs;
}
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd survival-of-the-fishest && npm test`
Expected: all 6 tests pass (19 total across files), `# pass 19`

- [ ] **Step 5: Commit**

```bash
git add src/stages.js test/stages.test.js
git commit -m "feat: add stage schema and fish/shark input builders"
```

---

### Task 5: Simulation core (`sim.js`)

**Files:**
- Create: `survival-of-the-fishest/src/sim.js`
- Test: `survival-of-the-fishest/test/sim.test.js`

**Interfaces:**
- Consumes: `createNN`, `forward` from `nn.js` (Task 2); `STAGES`, `buildFishInputs`, `buildSharkInputs` from `stages.js` (Task 4).
- Produces:
  - `createFish(bounds: {width,height}, nn: NN): FishState` where `FishState = { x,y,vx,vy,angle,alive: boolean, fitness: number, nn }`
  - `createShark(bounds: {width,height}, nn: NN): SharkState` where `SharkState = { x,y,vx,vy,angle,fitness: number, nn }`
  - `stepEpisode(state: EpisodeState, dt: number, stageId: number): void` where `EpisodeState = { fish: FishState[], shark: SharkState, bounds }`. Mutates state in place: moves fish/shark per NN output, bounces off walls, marks fish `alive=false` and freezes fitness on shark contact (radius-based), accumulates `fitness` for both fish (survival ticks + distance-from-shark bonus) and shark (eaten count + closing-distance bonus).
  - `isEpisodeOver(state: EpisodeState, elapsedTime: number, maxDuration: number): boolean` — true if `elapsedTime >= maxDuration` or every fish has `alive === false`.

- [ ] **Step 1: Write failing tests**

```javascript
// test/sim.test.js
import test from 'node:test';
import assert from 'node:assert/strict';
import { createNN } from '../src/nn.js';
import { createFish, createShark, stepEpisode, isEpisodeOver } from '../src/sim.js';

const bounds = { width: 200, height: 200 };

test('createFish places fish within bounds and alive', () => {
  const nn = createNN([4, 8, 2]);
  const fish = createFish(bounds, nn);
  assert.ok(fish.x >= 0 && fish.x <= bounds.width);
  assert.ok(fish.y >= 0 && fish.y <= bounds.height);
  assert.equal(fish.alive, true);
  assert.equal(fish.fitness, 0);
});

test('stepEpisode bounces fish off walls instead of leaving bounds', () => {
  const nn = createNN([0, 8, 2]);
  const fish = createFish(bounds, nn);
  fish.x = 1;
  fish.y = 100;
  fish.vx = -5;
  fish.vy = 0;
  const shark = createShark(bounds, createNN([2, 8, 2]));
  shark.x = 190;
  shark.y = 190;
  const state = { fish: [fish], shark, bounds };
  for (let i = 0; i < 5; i++) stepEpisode(state, 0.1, 1);
  assert.ok(fish.x >= 0, 'fish must not leave the tank through the left wall');
});

test('stepEpisode marks fish dead and accumulates shark fitness on contact', () => {
  const nn = createNN([0, 8, 2]);
  const fish = createFish(bounds, nn);
  fish.x = 100;
  fish.y = 100;
  fish.vx = 0;
  fish.vy = 0;
  const sharkNN = createNN([2, 8, 2]);
  const shark = createShark(bounds, sharkNN);
  shark.x = 100;
  shark.y = 100; // same position as fish -> contact
  const state = { fish: [fish], shark, bounds };
  const fitnessBefore = shark.fitness;
  stepEpisode(state, 0.1, 1);
  assert.equal(fish.alive, false);
  assert.ok(shark.fitness > fitnessBefore);
});

test('stepEpisode accumulates fish fitness for survival while alive', () => {
  const nn = createNN([0, 8, 2]);
  const fish = createFish(bounds, nn);
  fish.x = 10;
  fish.y = 10;
  const shark = createShark(bounds, createNN([2, 8, 2]));
  shark.x = 190;
  shark.y = 190;
  const state = { fish: [fish], shark, bounds };
  stepEpisode(state, 0.1, 1);
  assert.ok(fish.fitness > 0);
});

test('isEpisodeOver is true once max duration elapsed', () => {
  const state = { fish: [{ alive: true }], shark: {}, bounds };
  assert.equal(isEpisodeOver(state, 20, 20), true);
  assert.equal(isEpisodeOver(state, 19.9, 20), false);
});

test('isEpisodeOver is true once all fish are dead, even before max duration', () => {
  const state = { fish: [{ alive: false }, { alive: false }], shark: {}, bounds };
  assert.equal(isEpisodeOver(state, 1, 20), true);
});
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd survival-of-the-fishest && npm test`
Expected: FAIL — `Cannot find module '../src/sim.js'`

- [ ] **Step 3: Implement `src/sim.js`**

```javascript
// src/sim.js
import { forward } from './nn.js';
import { STAGES, buildFishInputs, buildSharkInputs } from './stages.js';

const FISH_RADIUS = 8;
const SHARK_RADIUS = 20;
const CONTACT_DISTANCE = FISH_RADIUS + SHARK_RADIUS;
const FISH_SPEED = 60; // px/sec at full thrust
const SHARK_SPEED = 70;

export function createFish(bounds, nn) {
  return {
    x: Math.random() * bounds.width,
    y: Math.random() * bounds.height,
    vx: 0,
    vy: 0,
    angle: Math.random() * Math.PI * 2,
    alive: true,
    fitness: 0,
    nn,
  };
}

export function createShark(bounds, nn) {
  return {
    x: Math.random() * bounds.width,
    y: Math.random() * bounds.height,
    vx: 0,
    vy: 0,
    angle: Math.random() * Math.PI * 2,
    fitness: 0,
    nn,
  };
}

function nearestFishTo(agent, fish) {
  const alive = fish.filter(f => f.alive);
  if (alive.length === 0) return null;
  return alive.reduce((closest, f) =>
    Math.hypot(f.x - agent.x, f.y - agent.y) < Math.hypot(closest.x - agent.x, closest.y - agent.y) ? f : closest
  );
}

function nearbyFishTo(fish, allFish, radius = 80) {
  return allFish.filter(f => f !== fish && f.alive && Math.hypot(f.x - fish.x, f.y - fish.y) <= radius);
}

function bounceOffWalls(agent, bounds, radius) {
  if (agent.x < radius) { agent.x = radius; agent.vx = Math.abs(agent.vx); }
  if (agent.x > bounds.width - radius) { agent.x = bounds.width - radius; agent.vx = -Math.abs(agent.vx); }
  if (agent.y < radius) { agent.y = radius; agent.vy = Math.abs(agent.vy); }
  if (agent.y > bounds.height - radius) { agent.y = bounds.height - radius; agent.vy = -Math.abs(agent.vy); }
}

export function stepEpisode(state, dt, stageId) {
  const { fish, shark, bounds } = state;

  const distanceToSharkBefore = fish.map(f => Math.hypot(f.x - shark.x, f.y - shark.y));

  for (const f of fish) {
    if (!f.alive) continue;
    const inputs = buildFishInputs(stageId, f, bounds, shark, nearbyFishTo(f, fish));
    const { output } = forward(f.nn, inputs);
    const [turn, thrust] = output;
    f.angle += turn * 0.15;
    const speed = Math.max(0, thrust) * FISH_SPEED;
    f.vx = Math.cos(f.angle) * speed;
    f.vy = Math.sin(f.angle) * speed;
    f.x += f.vx * dt;
    f.y += f.vy * dt;
    bounceOffWalls(f, bounds, FISH_RADIUS);

    f.fitness += dt; // survival time
    f.fitness += Math.hypot(f.x - shark.x, f.y - shark.y) * 0.001; // distance-from-shark bonus
  }

  const nearest = nearestFishTo(shark, fish);
  if (nearest) {
    const inputs = buildSharkInputs(stageId, shark, nearest);
    const { output } = forward(shark.nn, inputs);
    const [turn, thrust] = output;
    shark.angle += turn * 0.15;
    const speed = Math.max(0, thrust) * SHARK_SPEED;
    shark.vx = Math.cos(shark.angle) * speed;
    shark.vy = Math.sin(shark.angle) * speed;
    shark.x += shark.vx * dt;
    shark.y += shark.vy * dt;
    bounceOffWalls(shark, bounds, SHARK_RADIUS);
  }

  fish.forEach((f, i) => {
    if (!f.alive) return;
    const dist = Math.hypot(f.x - shark.x, f.y - shark.y);
    if (dist <= CONTACT_DISTANCE) {
      f.alive = false;
      shark.fitness += 10; // eaten bonus
    } else {
      const closingAmount = distanceToSharkBefore[i] - dist;
      if (closingAmount > 0) shark.fitness += closingAmount * 0.01; // closing-distance bonus
    }
  });
}

export function isEpisodeOver(state, elapsedTime, maxDuration) {
  if (elapsedTime >= maxDuration) return true;
  return state.fish.every(f => !f.alive);
}

export function getStageById(stageId) {
  return STAGES.find(s => s.id === stageId);
}
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd survival-of-the-fishest && npm test`
Expected: all 6 tests pass (25 total across files), `# pass 25`

- [ ] **Step 5: Commit**

```bash
git add src/sim.js test/sim.test.js
git commit -m "feat: add episode simulation (physics, collisions, fitness accumulation)"
```

---

### Task 6: Procedural canvas art (`render.js`)

**Files:**
- Create: `survival-of-the-fishest/src/render.js`

**Interfaces:**
- Consumes: `FishState`/`SharkState` shapes from `sim.js` (Task 5) — only reads `x, y, angle, vx, vy, alive`.
- Produces: `drawFish(ctx: CanvasRenderingContext2D, fish: FishState, t: number, hue: number): void`, `drawShark(ctx: CanvasRenderingContext2D, shark: SharkState, t: number): void`, `drawTankBackground(ctx: CanvasRenderingContext2D, bounds: {width,height}): void`.

This module is visual and has no meaningful unit test (canvas drawing calls). Verified manually via Task 9's end-to-end check. Implementation reuses the bezier art approach already approved in the design-preview artifact.

- [ ] **Step 1: Implement `src/render.js`**

```javascript
// src/render.js

export function drawTankBackground(ctx, bounds) {
  const grad = ctx.createLinearGradient(0, 0, 0, bounds.height);
  grad.addColorStop(0, '#0d3b4a');
  grad.addColorStop(1, '#071e28');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, bounds.width, bounds.height);
}

export function drawFish(ctx, fish, t, hue) {
  if (!fish.alive) return;
  const s = 12;
  const speed = Math.hypot(fish.vx, fish.vy);
  const wag = Math.sin(t * 8 * Math.max(0.3, speed / 60)) * 0.5;

  ctx.save();
  ctx.translate(fish.x, fish.y);
  ctx.rotate(fish.angle);

  const bodyGrad = ctx.createLinearGradient(-s, 0, s * 0.6, 0);
  bodyGrad.addColorStop(0, `hsl(${18 + hue + 20}, 85%, 45%)`);
  bodyGrad.addColorStop(1, `hsl(${18 + hue}, 95%, 62%)`);

  ctx.fillStyle = `hsl(${18 + hue}, 90%, 50%)`;
  ctx.beginPath();
  ctx.moveTo(-s * 0.9, 0);
  ctx.quadraticCurveTo(-s * 1.5, s * 0.55 * wag - s * 0.15, -s * 1.9, wag * s * 0.5);
  ctx.quadraticCurveTo(-s * 1.5, -s * 0.1, -s * 0.9, 0);
  ctx.fill();

  ctx.fillStyle = bodyGrad;
  ctx.beginPath();
  ctx.moveTo(s * 1.1, 0);
  ctx.bezierCurveTo(s * 0.7, -s * 0.55, -s * 0.6, -s * 0.5, -s * 0.95, 0);
  ctx.bezierCurveTo(-s * 0.6, s * 0.5, s * 0.7, s * 0.55, s * 1.1, 0);
  ctx.fill();

  ctx.fillStyle = `hsla(${18 + hue}, 90%, 40%, 0.85)`;
  ctx.beginPath();
  ctx.moveTo(-s * 0.05, -s * 0.4);
  ctx.quadraticCurveTo(s * 0.1, -s * 0.85, s * 0.35, -s * 0.4);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = '#0d1f1a';
  ctx.beginPath();
  ctx.arc(s * 0.62, -s * 0.08, s * 0.09, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

export function drawShark(ctx, shark, t) {
  const s = 32;
  const wag = Math.sin(t * 3.2) * 0.4;

  ctx.save();
  ctx.translate(shark.x, shark.y);
  ctx.rotate(shark.angle);

  ctx.fillStyle = '#17242b';
  ctx.beginPath();
  ctx.moveTo(-s * 0.95, 0);
  ctx.lineTo(-s * 1.6, -s * 0.55 + wag * s * 0.2);
  ctx.lineTo(-s * 1.15, 0);
  ctx.lineTo(-s * 1.6, s * 0.4 + wag * s * 0.2);
  ctx.closePath();
  ctx.fill();

  const bodyGrad = ctx.createLinearGradient(0, -s * 0.5, 0, s * 0.5);
  bodyGrad.addColorStop(0, '#17242b');
  bodyGrad.addColorStop(1, '#cfe9ec');
  ctx.fillStyle = bodyGrad;
  ctx.beginPath();
  ctx.moveTo(s * 1.3, 0);
  ctx.bezierCurveTo(s * 0.9, -s * 0.42, -s * 0.7, -s * 0.4, -s * 1.0, 0);
  ctx.bezierCurveTo(-s * 0.7, s * 0.4, s * 0.9, s * 0.42, s * 1.3, 0);
  ctx.fill();

  ctx.fillStyle = '#17242b';
  ctx.beginPath();
  ctx.moveTo(-s * 0.1, -s * 0.32);
  ctx.lineTo(s * 0.05, -s * 0.95);
  ctx.lineTo(s * 0.3, -s * 0.3);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = '#0a0a0a';
  ctx.beginPath();
  ctx.arc(s * 0.85, -s * 0.06, s * 0.045, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}
```

- [ ] **Step 2: Commit**

```bash
git add src/render.js
git commit -m "feat: add procedural canvas art for fish/shark/tank"
```

---

### Task 7: NN diagram visualization (`nnviz.js`)

**Files:**
- Create: `survival-of-the-fishest/src/nnviz.js`

**Interfaces:**
- Consumes: `NN` shape and `forward()`'s `activations` array from `nn.js` (Task 2).
- Produces: `drawNNDiagram(ctx: CanvasRenderingContext2D, nn: NN, activations: number[][], bounds: {width,height}): void`.

This module is visual, no meaningful unit test. Verified manually via Task 9.

- [ ] **Step 1: Implement `src/nnviz.js`**

```javascript
// src/nnviz.js

export function drawNNDiagram(ctx, nn, activations, bounds) {
  ctx.clearRect(0, 0, bounds.width, bounds.height);
  const layers = activations.length;
  const layerGap = bounds.width / (layers + 1);

  const positions = activations.map((layerActs, l) => {
    const x = layerGap * (l + 1);
    const nodeGap = bounds.height / (layerActs.length + 1);
    return layerActs.map((_, n) => ({ x, y: nodeGap * (n + 1) }));
  });

  ctx.lineWidth = 1;
  for (let l = 0; l < positions.length - 1; l++) {
    for (const from of positions[l]) {
      for (const to of positions[l + 1]) {
        ctx.strokeStyle = 'rgba(150, 200, 200, 0.15)';
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
      }
    }
  }

  positions.forEach((layerPos, l) => {
    layerPos.forEach((pos, n) => {
      const value = activations[l][n];
      const intensity = Math.min(1, Math.abs(value));
      const color = value >= 0
        ? `rgba(120, 220, 200, ${0.3 + intensity * 0.7})`
        : `rgba(220, 120, 120, ${0.3 + intensity * 0.7})`;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 6, 0, Math.PI * 2);
      ctx.fill();
    });
  });
}
```

- [ ] **Step 2: Commit**

```bash
git add src/nnviz.js
git commit -m "feat: add live NN node-link diagram renderer"
```

---

### Task 8: HUD and controls (`ui.js`)

**Files:**
- Create: `survival-of-the-fishest/src/ui.js`

**Interfaces:**
- Produces:
  - `renderHUD(el: HTMLElement, state: { stageId: number, stageName: string, generation: number, aliveCount: number, bestFitness: number, running: boolean, speed: number }): void`
  - `attachControls(el: HTMLElement, handlers: { onNextStage: () => void, onSpeedChange: (speed: number) => void, onTogglePause: () => void }): void` — creates a "Next Stage" button, a speed `<select>` (1x/4x/max), and a pause/resume button inside `el`, wiring them to the given handlers.

This module is visual/interactive, no meaningful unit test. Verified manually via Task 9.

- [ ] **Step 1: Implement `src/ui.js`**

```javascript
// src/ui.js

export function renderHUD(el, state) {
  el.textContent = '';
  const fields = [
    `Stage ${state.stageId}: ${state.stageName}`,
    `Gen ${state.generation}`,
    `Alive ${state.aliveCount}`,
    `Best ${state.bestFitness.toFixed(1)}`,
    state.running ? 'Running' : 'Paused',
    `${state.speed}x`,
  ];
  for (const text of fields) {
    const span = document.createElement('span');
    span.textContent = text;
    el.appendChild(span);
  }
}

export function attachControls(el, handlers) {
  const controls = document.createElement('div');
  controls.style.display = 'flex';
  controls.style.gap = '8px';

  const nextStageBtn = document.createElement('button');
  nextStageBtn.textContent = 'Next Stage';
  nextStageBtn.addEventListener('click', handlers.onNextStage);

  const pauseBtn = document.createElement('button');
  pauseBtn.textContent = 'Pause/Resume';
  pauseBtn.addEventListener('click', handlers.onTogglePause);

  const speedSelect = document.createElement('select');
  for (const speed of [1, 4, 20]) {
    const option = document.createElement('option');
    option.value = String(speed);
    option.textContent = `${speed}x`;
    speedSelect.appendChild(option);
  }
  speedSelect.addEventListener('change', () => handlers.onSpeedChange(Number(speedSelect.value)));

  controls.append(nextStageBtn, pauseBtn, speedSelect);
  el.appendChild(controls);
}
```

- [ ] **Step 2: Commit**

```bash
git add src/ui.js
git commit -m "feat: add HUD and control wiring (next stage, pause, speed)"
```

---

### Task 9: Bootstrap and wiring (`main.js`)

**Files:**
- Create: `survival-of-the-fishest/src/main.js`
- Modify: `survival-of-the-fishest/index.html` (add a controls container above the HUD)

**Interfaces:**
- Consumes: everything from Tasks 2-8 — `createNN`, `resizeInputLayer` (`nn.js`); `evolvePopulation` (`genetic.js`); `STAGES` (`stages.js`); `createFish`, `createShark`, `stepEpisode`, `isEpisodeOver`, `getStageById` (`sim.js`); `drawTankBackground`, `drawFish`, `drawShark` (`render.js`); `drawNNDiagram` (`nnviz.js`); `renderHUD`, `attachControls` (`ui.js`).
- Produces: the running app — no further module consumes this one.

- [ ] **Step 1: Modify `index.html`** — add a container div for controls above `#hud`

```html
<body>
  <div id="controls"></div>
  <div id="hud"></div>
  <div class="layout">
    <canvas id="tank" width="1000" height="600"></canvas>
    <canvas id="nnviz" width="360" height="600"></canvas>
  </div>
  <script type="module" src="src/main.js"></script>
</body>
```

- [ ] **Step 2: Implement `src/main.js`**

```javascript
// src/main.js
import { createNN, resizeInputLayer, mutateNN, crossoverNN, cloneNN, forward } from './nn.js';
import { evolvePopulation } from './genetic.js';
import { STAGES } from './stages.js';
import { createFish, createShark, stepEpisode, isEpisodeOver, getStageById } from './sim.js';
import { drawTankBackground, drawFish, drawShark } from './render.js';
import { drawNNDiagram } from './nnviz.js';
import { renderHUD, attachControls } from './ui.js';

const FISH_POP_SIZE = 30;
const SHARK_POP_SIZE = 8;
const MAX_EPISODE_DURATION = 20;
const ELITE_COUNT_FISH = 3;
const ELITE_COUNT_SHARK = 1;
const MUTATION_RATE = 0.1;
const MUTATION_SIGMA = 0.3;

const tankCanvas = document.getElementById('tank');
const tankCtx = tankCanvas.getContext('2d');
const nnCanvas = document.getElementById('nnviz');
const nnCtx = nnCanvas.getContext('2d');
const hudEl = document.getElementById('hud');
const controlsEl = document.getElementById('controls');

const bounds = { width: tankCanvas.width, height: tankCanvas.height };

const app = {
  stageIndex: 0,
  generation: 0,
  running: true,
  speed: 1,
  elapsed: 0,
  fishGenomes: [],
  sharkGenomes: [],
  state: null,
  selectedAgent: null,
};

function currentStage() {
  return STAGES[app.stageIndex];
}

function makeFishNN(stage) {
  return createNN([stage.fishInputSize, ...stage.fishHiddenLayers, 2]);
}

function makeSharkNN(stageId) {
  const inputSize = stageId === 6 ? 4 : 2;
  return createNN([inputSize, 8, 2]);
}

function initGenomes() {
  const stage = currentStage();
  app.fishGenomes = Array.from({ length: FISH_POP_SIZE }, () => makeFishNN(stage));
  app.sharkGenomes = Array.from({ length: SHARK_POP_SIZE }, () => makeSharkNN(stage.id));
}

function startEpisode() {
  const stage = currentStage();
  const fish = app.fishGenomes.map(nn => createFish(bounds, nn));
  const sharkNN = app.sharkGenomes[app.generation % app.sharkGenomes.length];
  const shark = createShark(bounds, sharkNN);
  app.state = { fish, shark, bounds };
  app.elapsed = 0;
  app.selectedAgent = fish[0];
  if (!app.stage6VelocityWired && stage.id === 6) {
    app.stage6VelocityWired = true;
  }
}

function endEpisodeAndEvolve() {
  const fishFitnesses = app.state.fish.map(f => f.fitness);
  const sharkFitness = app.state.shark.fitness;

  app.fishGenomes = evolvePopulation(app.fishGenomes, fishFitnesses, {
    eliteCount: ELITE_COUNT_FISH,
    mutationRate: MUTATION_RATE,
    mutationSigma: MUTATION_SIGMA,
    cloneFn: cloneNN,
    crossoverFn: crossoverNN,
    mutateFn: mutateNN,
  });

  const sharkGenomeIndex = app.generation % app.sharkGenomes.length;
  const sharkFitnesses = app.sharkGenomes.map((_, i) => (i === sharkGenomeIndex ? sharkFitness : 0));
  app.sharkGenomes = evolvePopulation(app.sharkGenomes, sharkFitnesses, {
    eliteCount: ELITE_COUNT_SHARK,
    mutationRate: MUTATION_RATE,
    mutationSigma: MUTATION_SIGMA,
    cloneFn: cloneNN,
    crossoverFn: crossoverNN,
    mutateFn: mutateNN,
  });

  app.generation++;
  startEpisode();
}

function advanceStage() {
  if (app.stageIndex >= STAGES.length - 1) return;
  app.stageIndex++;
  const stage = currentStage();
  app.fishGenomes = app.fishGenomes.map(nn => {
    const resized = resizeInputLayer(nn, stage.fishInputSize);
    return createNN([stage.fishInputSize, ...stage.fishHiddenLayers, 2]).layerSizes.length === resized.layerSizes.length
      ? resized
      : createNN([stage.fishInputSize, ...stage.fishHiddenLayers, 2]);
  });
  app.sharkGenomes = app.sharkGenomes.map(nn => resizeInputLayer(nn, stage.id === 6 ? 4 : 2));
  app.generation = 0;
  startEpisode();
}

function bestFitness() {
  return Math.max(...app.state.fish.map(f => f.fitness), 0);
}

function tick(dtMs) {
  const t = dtMs / 1000;
  const dt = 0.1;
  const stage = currentStage();

  for (let i = 0; i < app.speed; i++) {
    if (!app.running) break;
    stepEpisode(app.state, dt, stage.id);
    app.elapsed += dt;
    if (isEpisodeOver(app.state, app.elapsed, MAX_EPISODE_DURATION)) {
      endEpisodeAndEvolve();
      break;
    }
  }

  if (app.speed < 10) {
    drawTankBackground(tankCtx, bounds);
    app.state.fish.forEach((f, i) => drawFish(tankCtx, f, t, (i * 37) % 40 - 20));
    drawShark(tankCtx, app.state.shark, t);

    if (app.selectedAgent) {
      const stageForInputs = currentStage();
      const isShark = app.selectedAgent === app.state.shark;
      const { activations } = isShark
        ? forward(app.selectedAgent.nn, [0, 0, ...(stageForInputs.id === 6 ? [0, 0] : [])])
        : forward(app.selectedAgent.nn, new Array(stageForInputs.fishInputSize).fill(0));
      drawNNDiagram(nnCtx, app.selectedAgent.nn, activations, { width: nnCanvas.width, height: nnCanvas.height });
    }
  }

  renderHUD(hudEl, {
    stageId: stage.id,
    stageName: stage.name,
    generation: app.generation,
    aliveCount: app.state.fish.filter(f => f.alive).length,
    bestFitness: bestFitness(),
    running: app.running,
    speed: app.speed,
  });

  requestAnimationFrame(tick);
}

attachControls(controlsEl, {
  onNextStage: advanceStage,
  onSpeedChange: speed => { app.speed = speed; },
  onTogglePause: () => { app.running = !app.running; },
});

tankCanvas.addEventListener('click', e => {
  const rect = tankCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const candidates = [...app.state.fish.filter(f => f.alive), app.state.shark];
  app.selectedAgent = candidates.reduce((closest, a) =>
    Math.hypot(a.x - x, a.y - y) < Math.hypot(closest.x - x, closest.y - y) ? a : closest
  );
});

initGenomes();
startEpisode();
requestAnimationFrame(tick);
```

- [ ] **Step 3: Manual verification — open in browser**

Run: `cd survival-of-the-fishest && python3 -m http.server 8080` then open `http://localhost:8080` in a browser.

Expected:
- Aquarium canvas shows 30 fish moving (randomly at stage 1) and one shark chasing.
- NN diagram canvas shows a live node-link graph pulsing as the selected fish's inputs/outputs change.
- HUD shows stage/generation/alive/best-fitness/running/speed, updating live.
- Clicking a fish or the shark in the tank switches the NN diagram to that agent.
- "Next Stage" button advances `Stage 1` to `Stage 2` in the HUD and visibly changes fish behavior (from random to wall-avoidant) within a few generations.
- Speed selector changes how fast generations cycle (max skips rendering, HUD numbers still update).
- Pause/Resume halts and resumes the simulation.

- [ ] **Step 4: Commit**

```bash
git add src/main.js index.html
git commit -m "feat: wire simulation, evolution, rendering, and controls into running app"
```

---

## Self-Review Notes

- **Spec coverage:** stage ladder (Task 4), fish/shark GA with elitism/crossover/mutation (Task 3 + main.js wiring), episode mechanics/max duration/no respawn (Task 5), manual stage advance (Task 8 + main.js `advanceStage`), procedural art (Task 6), NN diagram (Task 7), HUD (Task 8) — all covered.
- **Placeholder scan:** none found; every step has runnable code.
- **Type consistency:** `FishState`/`SharkState` shapes match across `sim.js`, `render.js`, `stages.js`, `main.js` (`x,y,vx,vy,angle,alive,fitness,nn`). `NN` shape (`layerSizes,weights,biases`) matches across `nn.js`, `genetic.js` (generic, no direct reference), `sim.js`, `main.js`.
- **Known rough edge (acceptable for v1):** `advanceStage`'s hidden-layer-count check between stage 5→6 falls back to a fresh NN rather than a true weight-preserving resize across a hidden-layer-count change — acceptable since the spec only requires *input* weights to carry over, not hidden topology.
