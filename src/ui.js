// src/ui.js

export function renderHUD(el, state) {
  el.textContent = '';
  const fields = [
    `Stage ${state.stageId}: ${state.stageName}`,
    `Gen ${state.generation}`,
    `Alive ${state.aliveCount}`,
    `Best (this gen) ${state.bestFitness.toFixed(1)}`,
    `Last gen best ${state.lastGenBestFitness.toFixed(1)}`,
    `All-time best ${state.allTimeBestFitness.toFixed(1)}`,
    state.running ? 'Running' : 'Paused',
    `${state.speed}x`,
    state.untilAllDead ? 'Mode: until all dead' : 'Mode: timed episodes',
  ];
  for (const text of fields) {
    const span = document.createElement('span');
    span.textContent = text;
    el.appendChild(span);
  }
}

export function renderStageExplainer(el, stage, newInputLabels) {
  el.textContent = '';

  const heading = document.createElement('div');
  heading.style.fontWeight = 'bold';
  heading.textContent = `Stage ${stage.id}: ${stage.name}`;

  const desc = document.createElement('div');
  desc.textContent = stage.description;

  el.append(heading, desc);

  if (newInputLabels.length > 0) {
    const added = document.createElement('div');
    added.textContent = `New this stage: ${newInputLabels.join(', ')} (fish can now sense this and evolution has something new to act on).`;
    el.appendChild(added);
  }
}

export function renderNarrator(el, text) {
  el.textContent = text;
}

export function attachControls(el, handlers) {
  const controls = document.createElement('div');
  controls.style.display = 'flex';
  controls.style.gap = '8px';

  const prevStageBtn = document.createElement('button');
  prevStageBtn.textContent = 'Previous Stage';
  prevStageBtn.addEventListener('click', handlers.onPrevStage);

  const nextStageBtn = document.createElement('button');
  nextStageBtn.textContent = 'Next Stage';
  nextStageBtn.addEventListener('click', handlers.onNextStage);

  const pauseBtn = document.createElement('button');
  pauseBtn.textContent = 'Pause/Resume';
  pauseBtn.addEventListener('click', handlers.onTogglePause);

  const speedSelect = document.createElement('select');
  for (const speed of [0.25, 0.5, 1, 4, 20]) {
    const option = document.createElement('option');
    option.value = String(speed);
    option.textContent = `${speed}x`;
    if (speed === 1) option.selected = true;
    speedSelect.appendChild(option);
  }
  speedSelect.addEventListener('change', () => handlers.onSpeedChange(Number(speedSelect.value)));

  const endlessModeBtn = document.createElement('button');
  endlessModeBtn.textContent = 'Mode: Timed';
  endlessModeBtn.addEventListener('click', () => {
    const enabled = handlers.onToggleEndlessMode();
    endlessModeBtn.textContent = enabled ? 'Mode: Until All Dead' : 'Mode: Timed';
  });

  const fishSpeedLabel = document.createElement('span');
  fishSpeedLabel.textContent = 'Fish speed';
  const fishSpeedSlider = document.createElement('input');
  fishSpeedSlider.type = 'range';
  fishSpeedSlider.min = '10';
  fishSpeedSlider.max = '150';
  fishSpeedSlider.value = '60';
  const fishSpeedValue = document.createElement('span');
  fishSpeedValue.textContent = fishSpeedSlider.value;
  fishSpeedSlider.addEventListener('input', () => {
    fishSpeedValue.textContent = fishSpeedSlider.value;
    handlers.onFishSpeedChange(Number(fishSpeedSlider.value));
  });

  const sharkSpeedLabel = document.createElement('span');
  sharkSpeedLabel.textContent = 'Shark speed';
  const sharkSpeedSlider = document.createElement('input');
  sharkSpeedSlider.type = 'range';
  sharkSpeedSlider.min = '10';
  sharkSpeedSlider.max = '150';
  sharkSpeedSlider.value = '70';
  const sharkSpeedValue = document.createElement('span');
  sharkSpeedValue.textContent = sharkSpeedSlider.value;
  sharkSpeedSlider.addEventListener('input', () => {
    sharkSpeedValue.textContent = sharkSpeedSlider.value;
    handlers.onSharkSpeedChange(Number(sharkSpeedSlider.value));
  });

  controls.append(
    prevStageBtn, nextStageBtn, pauseBtn, speedSelect, endlessModeBtn,
    fishSpeedLabel, fishSpeedSlider, fishSpeedValue,
    sharkSpeedLabel, sharkSpeedSlider, sharkSpeedValue,
  );
  el.appendChild(controls);
}
