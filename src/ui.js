// src/ui.js

export function renderHUD(el, state) {
  el.textContent = '';
  const fields = [
    `Stage ${state.stageId}: ${state.stageName}`,
    `Gen ${state.generation}`,
    `Alive ${state.aliveCount}`,
    `Best ${state.bestFitness.toFixed(1)}`,
    state.running ? 'Running' : 'Paused',
    `${state.speed}x`,
  ];
  for (const text of fields) {
    const span = document.createElement('span');
    span.textContent = text;
    el.appendChild(span);
  }
}

export function attachControls(el, handlers) {
  const controls = document.createElement('div');
  controls.style.display = 'flex';
  controls.style.gap = '8px';

  const nextStageBtn = document.createElement('button');
  nextStageBtn.textContent = 'Next Stage';
  nextStageBtn.addEventListener('click', handlers.onNextStage);

  const pauseBtn = document.createElement('button');
  pauseBtn.textContent = 'Pause/Resume';
  pauseBtn.addEventListener('click', handlers.onTogglePause);

  const speedSelect = document.createElement('select');
  for (const speed of [1, 4, 20]) {
    const option = document.createElement('option');
    option.value = String(speed);
    option.textContent = `${speed}x`;
    speedSelect.appendChild(option);
  }
  speedSelect.addEventListener('change', () => handlers.onSpeedChange(Number(speedSelect.value)));

  controls.append(nextStageBtn, pauseBtn, speedSelect);
  el.appendChild(controls);
}
