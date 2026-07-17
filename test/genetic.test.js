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
