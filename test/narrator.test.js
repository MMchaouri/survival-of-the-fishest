import test from 'node:test';
import assert from 'node:assert/strict';
import { buildGenerationCommentary } from '../src/narrator.js';

const stage = { id: 3, name: 'shark-distance', description: 'Adds distance to shark.' };

test('buildGenerationCommentary reports a new all-time best', () => {
  const text = buildGenerationCommentary({
    generation: 5,
    genBest: 12.5,
    genAvg: 6.2,
    previousAllTimeBest: 10,
    survivorCount: 20,
    populationSize: 30,
    stage,
  });
  assert.match(text, /New all-time best, up 2\.5/);
});

test('buildGenerationCommentary reports the elite holding when best is unchanged', () => {
  const text = buildGenerationCommentary({
    generation: 5,
    genBest: 10,
    genAvg: 6.2,
    previousAllTimeBest: 10,
    survivorCount: 20,
    populationSize: 30,
    stage,
  });
  assert.match(text, /held its position unchanged/);
});

test('buildGenerationCommentary reports falling short of the all-time best', () => {
  const text = buildGenerationCommentary({
    generation: 5,
    genBest: 7,
    genAvg: 4,
    previousAllTimeBest: 10,
    survivorCount: 20,
    populationSize: 30,
    stage,
  });
  assert.match(text, /fell 3\.0 short/);
});

test('buildGenerationCommentary flags a strong shark generation on low survival', () => {
  const text = buildGenerationCommentary({
    generation: 5,
    genBest: 10,
    genAvg: 4,
    previousAllTimeBest: 10,
    survivorCount: 3,
    populationSize: 30,
    stage,
  });
  assert.match(text, /3\/30 fish survived - the shark had a strong generation/);
});

test('buildGenerationCommentary flags a struggling shark generation on high survival', () => {
  const text = buildGenerationCommentary({
    generation: 5,
    genBest: 10,
    genAvg: 4,
    previousAllTimeBest: 10,
    survivorCount: 25,
    populationSize: 30,
    stage,
  });
  assert.match(text, /25\/30 fish survived - the shark struggled/);
});

test('buildGenerationCommentary includes the stage description', () => {
  const text = buildGenerationCommentary({
    generation: 5,
    genBest: 10,
    genAvg: 4,
    previousAllTimeBest: 10,
    survivorCount: 15,
    populationSize: 30,
    stage,
  });
  assert.match(text, /Stage 3 \(shark-distance\): Adds distance to shark\./);
});
