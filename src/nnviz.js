// src/nnviz.js

export function drawNNDiagram(ctx, nn, activations, bounds, labels = {}) {
  ctx.clearRect(0, 0, bounds.width, bounds.height);
  const layers = activations.length;
  const layerGap = bounds.width / (layers + 1);

  const positions = activations.map((layerActs, l) => {
    const x = layerGap * (l + 1);
    const nodeGap = bounds.height / (layerActs.length + 1);
    return layerActs.map((_, n) => ({ x, y: nodeGap * (n + 1) }));
  });

  ctx.lineWidth = 1;
  for (let l = 0; l < positions.length - 1; l++) {
    for (const from of positions[l]) {
      for (const to of positions[l + 1]) {
        ctx.strokeStyle = 'rgba(150, 200, 200, 0.15)';
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
      }
    }
  }

  positions.forEach((layerPos, l) => {
    layerPos.forEach((pos, n) => {
      const value = activations[l][n];
      const intensity = Math.min(1, Math.abs(value));
      const color = value >= 0
        ? `rgba(120, 220, 200, ${0.3 + intensity * 0.7})`
        : `rgba(220, 120, 120, ${0.3 + intensity * 0.7})`;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 6, 0, Math.PI * 2);
      ctx.fill();
    });
  });

  ctx.font = '10px ui-monospace, monospace';

  const inputLabels = labels.inputLabels ?? [];
  ctx.textAlign = 'right';
  ctx.fillStyle = 'rgba(230, 240, 240, 0.85)';
  positions[0]?.forEach((pos, n) => {
    const text = inputLabels[n];
    if (text) ctx.fillText(text, pos.x - 10, pos.y + 3);
  });

  const outputLabels = labels.outputLabels ?? [];
  const lastLayer = positions[positions.length - 1];
  ctx.textAlign = 'left';
  lastLayer.forEach((pos, n) => {
    const text = outputLabels[n];
    if (text) ctx.fillText(text, pos.x + 10, pos.y + 3);
  });

  ctx.fillStyle = 'rgba(200, 220, 220, 0.6)';
  ctx.fillText('green = excite, red = inhibit', 6, bounds.height - 8);
}
