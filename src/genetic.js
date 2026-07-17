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
