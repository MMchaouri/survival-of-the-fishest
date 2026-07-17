// src/nnviz.js

export function drawNNDiagram(ctx, nn, activations, bounds) {
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
}
