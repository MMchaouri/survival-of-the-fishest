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
