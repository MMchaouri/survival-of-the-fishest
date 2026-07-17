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

test('stepEpisode steers shark toward nearest fish even with a zero-weight NN', () => {
  // Zero weights/biases -> NN output is always [0, 0] (no turn from the NN),
  // isolating the baseline pursuit term: the shark's heading should still
  // rotate toward the fish rather than stay frozen or drift arbitrarily.
  const sharkNN = createNN([2, 8, 2]);
  sharkNN.weights = sharkNN.weights.map(layer => layer.map(row => row.map(() => 0)));
  const shark = createShark(bounds, sharkNN);
  shark.x = 100;
  shark.y = 100;
  shark.angle = 0; // facing +x (away from the fish, which is above-left)

  const fish = createFish(bounds, createNN([0, 8, 2]));
  fish.x = 50;
  fish.y = 50;
  fish.vx = 0;
  fish.vy = 0;

  const state = { fish: [fish], shark, bounds };
  const angleToFish = Math.atan2(fish.y - shark.y, fish.x - shark.x);
  const initialDiff = Math.abs(Math.atan2(Math.sin(angleToFish - shark.angle), Math.cos(angleToFish - shark.angle)));

  stepEpisode(state, 0.1, 1);

  const newDiff = Math.abs(Math.atan2(Math.sin(angleToFish - shark.angle), Math.cos(angleToFish - shark.angle)));
  assert.ok(newDiff < initialDiff, 'shark heading should turn closer to the nearest fish');
});

test('stepEpisode honors custom fishSpeed/sharkSpeed overrides', () => {
  const fish = createFish(bounds, createNN([0, 8, 2]));
  fish.x = 100;
  fish.y = 100;
  const shark = createShark(bounds, createNN([2, 8, 2]));
  shark.x = 150;
  shark.y = 150;
  const state = { fish: [fish], shark, bounds };

  stepEpisode(state, 0.1, 1, { fishSpeed: 0, sharkSpeed: 0 });

  assert.equal(fish.x, 100, 'fishSpeed: 0 should stop the fish from moving regardless of NN output');
  assert.equal(fish.y, 100);
  assert.equal(shark.x, 150, 'sharkSpeed: 0 should stop the shark from moving regardless of NN output');
  assert.equal(shark.y, 150);
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
