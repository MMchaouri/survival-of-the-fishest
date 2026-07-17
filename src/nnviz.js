// src/nnviz.js

export function drawNNDiagram(ctx, nn, activations, bounds, labels = {}, weightDiff = null) {
  ctx.clearRect(0, 0, bounds.width, bounds.height);
  const layers = activations.length;
  const layerGap = bounds.width / (layers + 1);

  const positions = activations.map((layerActs, l) => {
    const x = layerGap * (l + 1);
    const nodeGap = bounds.height / (layerActs.length + 1);
    return layerActs.map((_, n) => ({ x, y: nodeGap * (n + 1) }));
  });

  // Edge color/width encode the actual weight (sign + magnitude), so this
  // shows which connections matter, not just which nodes are firing.
  // Edges that changed noticeably from the last generation's best genome
  // (via weightDiff) are highlighted in yellow, since nothing here is
  // trained by backprop - mutation + selection is the only thing that ever
  // changes a weight, and this is what that change looks like.
  const CHANGED_THRESHOLD = 0.05;
  for (let l = 0; l < positions.length - 1; l++) {
    positions[l].forEach((from, i) => {
      positions[l + 1].forEach((to, j) => {
        const w = nn.weights[l]?.[j]?.[i] ?? 0;
        const magnitude = Math.min(1, Math.abs(w));
        const diffVal = weightDiff?.weights?.[l]?.[j]?.[i] ?? 0;
        const changed = Math.abs(diffVal) > CHANGED_THRESHOLD;

        if (changed) {
          const changeIntensity = Math.min(1, Math.abs(diffVal));
          ctx.strokeStyle = `rgba(255, 210, 90, ${0.5 + changeIntensity * 0.5})`;
          ctx.lineWidth = 1.5 + changeIntensity * 2.5;
        } else {
          ctx.strokeStyle = w >= 0
            ? `rgba(120, 220, 200, ${0.08 + magnitude * 0.5})`
            : `rgba(220, 120, 120, ${0.08 + magnitude * 0.5})`;
          ctx.lineWidth = 0.5 + magnitude * 2;
        }

        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
      });
    });
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
  ctx.fillText('node: green = excite, red = inhibit', 6, bounds.height - 20);
  ctx.fillText('link: color/width = weight, yellow = changed since last gen', 6, bounds.height - 8);
}
