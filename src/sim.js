import { forward } from './nn.js';
import { STAGES, buildFishInputs, buildSharkInputs } from './stages.js';

const FISH_RADIUS = 8;
const SHARK_RADIUS = 20;
const CONTACT_DISTANCE = FISH_RADIUS + SHARK_RADIUS;
const FISH_SPEED = 60; // px/sec at full thrust
const SHARK_SPEED = 70;
const TURN_RATE = 1.5; // rad/sec at full turn output
const FISH_WANDER_NOISE = 0.6; // rad/sec of random heading drift, so a fish with no directional signal (e.g. stage 1) wanders instead of tracing a fixed circle

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
  // Reflects the agent's heading (angle), not just its velocity — vx/vy are
  // recomputed from angle every tick before this runs, so flipping only
  // vx/vy here had no lasting effect and let agents pin against a wall
  // indefinitely while their NN kept steering back into it.
  let bounced = false;
  if (agent.x < radius) { agent.x = radius; agent.angle = Math.PI - agent.angle; bounced = true; }
  if (agent.x > bounds.width - radius) { agent.x = bounds.width - radius; agent.angle = Math.PI - agent.angle; bounced = true; }
  if (agent.y < radius) { agent.y = radius; agent.angle = -agent.angle; bounced = true; }
  if (agent.y > bounds.height - radius) { agent.y = bounds.height - radius; agent.angle = -agent.angle; bounced = true; }
  if (bounced) {
    const speed = Math.hypot(agent.vx, agent.vy);
    agent.vx = Math.cos(agent.angle) * speed;
    agent.vy = Math.sin(agent.angle) * speed;
  }
}

export function stepEpisode(state, dt, stageId) {
  const { fish, shark, bounds } = state;

  const distanceToSharkBefore = fish.map(f => Math.hypot(f.x - shark.x, f.y - shark.y));

  for (const f of fish) {
    if (!f.alive) continue;
    const inputs = buildFishInputs(stageId, f, bounds, shark, nearbyFishTo(f, fish));
    f.lastInputs = inputs;
    const { output } = forward(f.nn, inputs);
    const [turn, thrust] = output;
    f.angle += turn * TURN_RATE * dt + (Math.random() - 0.5) * FISH_WANDER_NOISE * dt;
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
    shark.lastInputs = inputs;
    const { output } = forward(shark.nn, inputs);
    const [nnTurn, thrust] = output;

    // Baseline pursuit: blend the evolved NN's turn with a direct bearing to
    // the nearest fish, so an untrained/undertrained shark still hunts
    // instead of carving a constant-curvature circle from a near-constant
    // NN output.
    const angleToNearest = Math.atan2(nearest.y - shark.y, nearest.x - shark.x);
    let angleDiff = angleToNearest - shark.angle;
    angleDiff = Math.atan2(Math.sin(angleDiff), Math.cos(angleDiff));
    const seekTurn = Math.max(-1, Math.min(1, angleDiff));
    const turn = 0.5 * nnTurn + 0.5 * seekTurn;

    shark.angle += turn * TURN_RATE * dt;
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
