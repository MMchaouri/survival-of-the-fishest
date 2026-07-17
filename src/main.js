// src/main.js
import { createNN, resizeInputLayer, mutateNN, crossoverNN, cloneNN, forward } from './nn.js';
import { evolvePopulation } from './genetic.js';
import { STAGES, fishInputLabels, sharkInputLabels, OUTPUT_LABELS } from './stages.js';
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
const BASE_TIME_SCALE = 2; // doubles every speed option's real pace, since 1x felt too slow

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
  fishSpeed: 60,
  sharkSpeed: 70,
  untilAllDead: false,
  elapsed: 0,
  fishGenomes: [],
  sharkGenomes: [],
  sharkFitnesses: [],
  allTimeBestFitness: 0,
  lastGenBestFitness: 0,
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
  app.sharkFitnesses = new Array(app.sharkGenomes.length).fill(0);
  app.allTimeBestFitness = 0;
  app.lastGenBestFitness = 0;
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

  const genBest = Math.max(...fishFitnesses, 0);
  app.lastGenBestFitness = genBest;
  app.allTimeBestFitness = Math.max(app.allTimeBestFitness, genBest);

  app.fishGenomes = evolvePopulation(app.fishGenomes, fishFitnesses, {
    eliteCount: ELITE_COUNT_FISH,
    mutationRate: MUTATION_RATE,
    mutationSigma: MUTATION_SIGMA,
    cloneFn: cloneNN,
    crossoverFn: crossoverNN,
    mutateFn: mutateNN,
  });

  const sharkGenomeIndex = app.generation % app.sharkGenomes.length;
  app.sharkFitnesses[sharkGenomeIndex] = sharkFitness;

  if ((app.generation + 1) % app.sharkGenomes.length === 0) {
    app.sharkGenomes = evolvePopulation(app.sharkGenomes, app.sharkFitnesses, {
      eliteCount: ELITE_COUNT_SHARK,
      mutationRate: MUTATION_RATE,
      mutationSigma: MUTATION_SIGMA,
      cloneFn: cloneNN,
      crossoverFn: crossoverNN,
      mutateFn: mutateNN,
    });
    app.sharkFitnesses = new Array(app.sharkGenomes.length).fill(0);
  }

  app.generation++;
  startEpisode();
}

function goToStage(newIndex) {
  app.stageIndex = newIndex;
  const stage = currentStage();
  app.fishGenomes = app.fishGenomes.map(nn => {
    const resized = resizeInputLayer(nn, stage.fishInputSize);
    return createNN([stage.fishInputSize, ...stage.fishHiddenLayers, 2]).layerSizes.length === resized.layerSizes.length
      ? resized
      : createNN([stage.fishInputSize, ...stage.fishHiddenLayers, 2]);
  });
  app.sharkGenomes = app.sharkGenomes.map(nn => resizeInputLayer(nn, stage.id === 6 ? 4 : 2));
  app.sharkFitnesses = new Array(app.sharkGenomes.length).fill(0);
  app.generation = 0;
  app.allTimeBestFitness = 0;
  app.lastGenBestFitness = 0;
  startEpisode();
}

function advanceStage() {
  if (app.stageIndex >= STAGES.length - 1) return;
  goToStage(app.stageIndex + 1);
}

function previousStage() {
  if (app.stageIndex <= 0) return;
  goToStage(app.stageIndex - 1);
}

function bestFitness() {
  return Math.max(...app.state.fish.map(f => f.fitness), 0);
}

let lastTimestampMs = null;

function tick(timestampMs) {
  if (lastTimestampMs === null) lastTimestampMs = timestampMs;
  // Clamp so a backgrounded tab (or a slow frame) can't dump a huge sim-time
  // jump into one step; caps the physics step to 100ms of real time.
  const realDt = Math.min((timestampMs - lastTimestampMs) / 1000, 0.1);
  lastTimestampMs = timestampMs;

  const t = timestampMs / 1000;
  const stage = currentStage();

  if (app.running) {
    // dt is scaled by the speed multiplier (and a base time scale, since 1x
    // alone felt too slow) so 1x tracks a brisk real pace, 0.25x/0.5x run in
    // slow motion, and 4x/20x fast-forward.
    const dt = realDt * app.speed * BASE_TIME_SCALE;
    stepEpisode(app.state, dt, stage.id, { fishSpeed: app.fishSpeed, sharkSpeed: app.sharkSpeed });
    app.elapsed += dt;
    const maxDuration = app.untilAllDead ? Infinity : MAX_EPISODE_DURATION;
    if (isEpisodeOver(app.state, app.elapsed, maxDuration)) {
      endEpisodeAndEvolve();
    }
  }

  if (app.speed < 10) {
    drawTankBackground(tankCtx, bounds);
    app.state.fish.forEach((f, i) => drawFish(tankCtx, f, t, (i * 37) % 40 - 20));
    drawShark(tankCtx, app.state.shark, t);

    if (app.selectedAgent) {
      const stageForInputs = currentStage();
      const isShark = app.selectedAgent === app.state.shark;
      const fallbackInputs = isShark
        ? new Array(stageForInputs.id === 6 ? 4 : 2).fill(0)
        : new Array(stageForInputs.fishInputSize).fill(0);
      const inputs = app.selectedAgent.lastInputs ?? fallbackInputs;
      const { activations } = forward(app.selectedAgent.nn, inputs);
      const nnLabels = {
        inputLabels: isShark ? sharkInputLabels(stageForInputs.id) : fishInputLabels(stageForInputs.id),
        outputLabels: OUTPUT_LABELS,
      };
      drawNNDiagram(nnCtx, app.selectedAgent.nn, activations, { width: nnCanvas.width, height: nnCanvas.height }, nnLabels);
    }
  }

  renderHUD(hudEl, {
    stageId: stage.id,
    stageName: stage.name,
    generation: app.generation,
    aliveCount: app.state.fish.filter(f => f.alive).length,
    bestFitness: bestFitness(),
    lastGenBestFitness: app.lastGenBestFitness,
    allTimeBestFitness: app.allTimeBestFitness,
    running: app.running,
    speed: app.speed,
    untilAllDead: app.untilAllDead,
  });

  requestAnimationFrame(tick);
}

attachControls(controlsEl, {
  onPrevStage: previousStage,
  onNextStage: advanceStage,
  onSpeedChange: speed => { app.speed = speed; },
  onTogglePause: () => { app.running = !app.running; },
  onFishSpeedChange: speed => { app.fishSpeed = speed; },
  onSharkSpeedChange: speed => { app.sharkSpeed = speed; },
  onToggleEndlessMode: () => {
    app.untilAllDead = !app.untilAllDead;
    return app.untilAllDead;
  },
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
