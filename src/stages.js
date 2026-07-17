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

export function fishInputLabels(stageId) {
  const labels = [];
  if (stageId === 1) return labels;
  labels.push('wallL', 'wallR', 'wallT', 'wallB');
  if (stageId === 2) return labels;
  labels.push('sharkDist');
  if (stageId === 3) return labels;
  labels.push('sharkAngle');
  if (stageId === 4) return labels;
  labels.push('nbrDX', 'nbrDY');
  if (stageId === 5) return labels;
  labels.push('sharkVX', 'sharkVY');
  return labels;
}

export function sharkInputLabels(stageId) {
  const labels = ['preyDist', 'preyAngle'];
  if (stageId === 6) labels.push('preyVX', 'preyVY');
  return labels;
}

export const OUTPUT_LABELS = ['turn', 'thrust'];
