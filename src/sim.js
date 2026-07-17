import { forward } from './nn.js';
import { STAGES, buildFishInputs, buildSharkInputs } from './stages.js';

const FISH_RADIUS = 8;
const SHARK_RADIUS = 20;
const CONTACT_DISTANCE = FISH_RADIUS + SHARK_RADIUS;
const FISH_SPEED = 60; // px/sec at full thrust
const SHARK_SPEED = 70;
const TURN_RATE = 1.5; // rad/sec at full turn output
const FISH_WANDER_NOISE = 0.6; // rad/sec of random heading drift, so a fish with no directional signal (e.g. stage 1) wanders instead of tracing a fixed circle
const FISH_MIN_THRUST = 0.2; // floor on forward speed so a near-zero NN thrust output doesn't read as "frozen, just spinning"
const SHARK_MIN_THRUST = 0.3;
const WALL_MARGIN = 60; // px from a wall where steer-away kicks in, well before the hard position clamp

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

// Returns a turn bias in [-1, 1] that steers the agent away from whichever
// wall(s) it's within WALL_MARGIN of, growing stronger closer to the wall,
// and 0 once it's clear. Used to turn away smoothly before ever reaching the
// edge, instead of clamping and reversing on contact.
function wallAvoidanceTurn(agent, bounds, margin) {
  let biasX = 0;
  let biasY = 0;
  if (agent.x < margin) biasX += margin - agent.x;
  if (agent.x > bounds.width - margin) biasX -= agent.x - (bounds.width - margin);
  if (agent.y < margin) biasY += margin - agent.y;
  if (agent.y > bounds.height - margin) biasY -= agent.y - (bounds.height - margin);
  if (biasX === 0 && biasY === 0) return 0;

  const desiredAngle = Math.atan2(biasY, biasX);
  let diff = desiredAngle - agent.angle;
  diff = Math.atan2(Math.sin(diff), Math.cos(diff));
  const strength = Math.min(1, Math.hypot(biasX, biasY) / margin);
  return Math.max(-1, Math.min(1, diff)) * strength;
}

// Hard safety net only — clamps position inside bounds. Heading is turned
// away from walls in advance by wallAvoidanceTurn, so this should rarely be
// the thing that actually stops an agent.
function clampPosition(agent, bounds, radius) {
  agent.x = Math.min(Math.max(agent.x, radius), bounds.width - radius);
  agent.y = Math.min(Math.max(agent.y, radius), bounds.height - radius);
}

export function stepEpisode(state, dt, stageId, speedConfig = {}) {
  const fishSpeed = speedConfig.fishSpeed ?? FISH_SPEED;
  const sharkSpeed = speedConfig.sharkSpeed ?? SHARK_SPEED;
  const { fish, shark, bounds } = state;

  const distanceToSharkBefore = fish.map(f => Math.hypot(f.x - shark.x, f.y - shark.y));

  for (const f of fish) {
    if (!f.alive) continue;
    const inputs = buildFishInputs(stageId, f, bounds, shark, nearbyFishTo(f, fish));
    f.lastInputs = inputs;
    const { output } = forward(f.nn, inputs);
    const [nnTurn, thrust] = output;
    const wallTurn = wallAvoidanceTurn(f, bounds, WALL_MARGIN);

    // Baseline flee: from stage 3 onward the fish's inputs include shark
    // distance/angle, but a partially-evolved NN doesn't reliably act on
    // that yet, so fish would swim in lazy circles regardless of how far
    // the shark is. Blend in a direct bearing away from the shark, same
    // pattern as the shark's own baseline pursuit, so distance-keeping is
    // visible immediately and gets refined by evolution rather than
    // invented from nothing.
    let turn = nnTurn + wallTurn;
    if (stageId >= 3) {
      const angleToShark = Math.atan2(shark.y - f.y, shark.x - f.x);
      let fleeDiff = (angleToShark + Math.PI) - f.angle;
      fleeDiff = Math.atan2(Math.sin(fleeDiff), Math.cos(fleeDiff));
      const fleeTurn = Math.max(-1, Math.min(1, fleeDiff));
      turn = 0.6 * turn + 0.4 * fleeTurn;
    }

    // Wander noise is scoped to stage 1 only: it exists purely to break the
    // frozen-circle problem when a fish has zero directional signal to act
    // on. Past stage 1 the fish has walls/shark/flee signal to commit to,
    // and applying the same noise there just made every fish zigzag
    // constantly, quietly erasing a chunk of any real speed advantage over
    // the shark (which gets no such noise).
    const wanderNoise = stageId === 1 ? (Math.random() - 0.5) * FISH_WANDER_NOISE * dt : 0;
    f.angle += turn * TURN_RATE * dt + wanderNoise;
    const speed = Math.max(FISH_MIN_THRUST, thrust) * fishSpeed;
    f.vx = Math.cos(f.angle) * speed;
    f.vy = Math.sin(f.angle) * speed;
    f.x += f.vx * dt;
    f.y += f.vy * dt;
    clampPosition(f, bounds, FISH_RADIUS);

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
    const wallTurn = wallAvoidanceTurn(shark, bounds, WALL_MARGIN);
    const turn = 0.5 * nnTurn + 0.5 * seekTurn + wallTurn;

    shark.angle += turn * TURN_RATE * dt;
    const speed = Math.max(SHARK_MIN_THRUST, thrust) * sharkSpeed;
    shark.vx = Math.cos(shark.angle) * speed;
    shark.vy = Math.sin(shark.angle) * speed;
    shark.x += shark.vx * dt;
    shark.y += shark.vy * dt;
    clampPosition(shark, bounds, SHARK_RADIUS);
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
