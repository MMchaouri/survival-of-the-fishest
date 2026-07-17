// src/narrator.js
//
// Turns the raw numbers from one generation into a plain-English sentence
// about what evolution actually did. Pure function so it's testable without
// touching the DOM or the sim.

export function buildGenerationCommentary({
  generation,
  genBest,
  genAvg,
  previousAllTimeBest,
  survivorCount,
  populationSize,
  stage,
}) {
  const parts = [];

  parts.push(`Generation ${generation}: best fitness ${genBest.toFixed(1)}, population average ${genAvg.toFixed(1)}.`);

  if (genBest > previousAllTimeBest) {
    const delta = genBest - previousAllTimeBest;
    parts.push(`New all-time best, up ${delta.toFixed(1)} - mutation found a genome that outperforms every one before it.`);
  } else if (genBest === previousAllTimeBest) {
    parts.push(`The best genome held its position unchanged - elitism carried it forward untouched, nothing beat it this generation.`);
  } else {
    const delta = previousAllTimeBest - genBest;
    parts.push(`This generation's best fell ${delta.toFixed(1)} short of the all-time best - normal variance, mutation doesn't always help right away.`);
  }

  const survivalRatio = populationSize > 0 ? survivorCount / populationSize : 0;
  if (survivalRatio >= 0.7) {
    parts.push(`${survivorCount}/${populationSize} fish survived - the shark struggled to catch this population.`);
  } else if (survivalRatio <= 0.3) {
    parts.push(`Only ${survivorCount}/${populationSize} fish survived - the shark had a strong generation.`);
  } else {
    parts.push(`${survivorCount}/${populationSize} fish survived.`);
  }

  if (stage) {
    parts.push(`Stage ${stage.id} (${stage.name}): ${stage.description}`);
  }

  return parts.join(' ');
}
