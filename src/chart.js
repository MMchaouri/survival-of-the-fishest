// src/chart.js

export function drawFitnessChart(ctx, history, bounds) {
  ctx.clearRect(0, 0, bounds.width, bounds.height);
  ctx.font = '10px ui-monospace, monospace';

  if (history.length < 2) {
    ctx.fillStyle = 'rgba(200, 220, 220, 0.6)';
    ctx.fillText('Fitness per generation (need 2+ to plot)', 6, 16);
    return;
  }

  const maxVal = Math.max(...history.map(h => Math.max(h.best, h.avg)), 1);
  const stepX = bounds.width / (history.length - 1);
  const topPad = 6;
  const bottomPad = 16;
  const toY = v => bounds.height - bottomPad - (v / maxVal) * (bounds.height - topPad - bottomPad);

  function drawLine(key, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.forEach((h, i) => {
      const x = i * stepX;
      const y = toY(h[key]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  drawLine('avg', 'rgba(150, 180, 220, 0.7)');
  drawLine('best', 'rgba(120, 220, 200, 0.9)');

  const last = history[history.length - 1];
  ctx.fillStyle = 'rgba(200, 220, 220, 0.8)';
  ctx.fillText(
    `gen ${last.gen}  best ${last.best.toFixed(1)}  avg ${last.avg.toFixed(1)}`,
    6,
    bounds.height - 4,
  );
}
