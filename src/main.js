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
  speedAccumulator: 0,
  elapsed: 0,
  fishGenomes: [],
  sharkGenomes: [],
  sharkFitnesses: [],
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

function tick(dtMs) {
  const t = dtMs / 1000;
  const dt = 0.1;
  const stage = currentStage();

  if (app.running) {
    app.speedAccumulator += app.speed;
    while (app.speedAccumulator >= 1) {
      stepEpisode(app.state, dt, stage.id);
      app.elapsed += dt;
      app.speedAccumulator -= 1;
      if (isEpisodeOver(app.state, app.elapsed, MAX_EPISODE_DURATION)) {
        endEpisodeAndEvolve();
        app.speedAccumulator = 0;
        break;
      }
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
    running: app.running,
    speed: app.speed,
  });

  requestAnimationFrame(tick);
}

attachControls(controlsEl, {
  onPrevStage: previousStage,
  onNextStage: advanceStage,
  onSpeedChange: speed => { app.speed = speed; app.speedAccumulator = 0; },
  onTogglePause: () => { app.running = !app.running; },
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
