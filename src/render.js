// src/render.js

export function drawTankBackground(ctx, bounds) {
  const grad = ctx.createLinearGradient(0, 0, 0, bounds.height);
  grad.addColorStop(0, '#0d3b4a');
  grad.addColorStop(1, '#071e28');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, bounds.width, bounds.height);
}

export function drawFish(ctx, fish, t, hue) {
  if (!fish.alive) return;
  const s = 12;
  const speed = Math.hypot(fish.vx, fish.vy);
  const wag = Math.sin(t * 8 * Math.max(0.3, speed / 60)) * 0.5;

  ctx.save();
  ctx.translate(fish.x, fish.y);
  ctx.rotate(fish.angle);

  const bodyGrad = ctx.createLinearGradient(-s, 0, s * 0.6, 0);
  bodyGrad.addColorStop(0, `hsl(${18 + hue + 20}, 85%, 45%)`);
  bodyGrad.addColorStop(1, `hsl(${18 + hue}, 95%, 62%)`);

  ctx.fillStyle = `hsl(${18 + hue}, 90%, 50%)`;
  ctx.beginPath();
  ctx.moveTo(-s * 0.9, 0);
  ctx.quadraticCurveTo(-s * 1.5, s * 0.55 * wag - s * 0.15, -s * 1.9, wag * s * 0.5);
  ctx.quadraticCurveTo(-s * 1.5, -s * 0.1, -s * 0.9, 0);
  ctx.fill();

  ctx.fillStyle = bodyGrad;
  ctx.beginPath();
  ctx.moveTo(s * 1.1, 0);
  ctx.bezierCurveTo(s * 0.7, -s * 0.55, -s * 0.6, -s * 0.5, -s * 0.95, 0);
  ctx.bezierCurveTo(-s * 0.6, s * 0.5, s * 0.7, s * 0.55, s * 1.1, 0);
  ctx.fill();

  ctx.fillStyle = `hsla(${18 + hue}, 90%, 40%, 0.85)`;
  ctx.beginPath();
  ctx.moveTo(-s * 0.05, -s * 0.4);
  ctx.quadraticCurveTo(s * 0.1, -s * 0.85, s * 0.35, -s * 0.4);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = '#0d1f1a';
  ctx.beginPath();
  ctx.arc(s * 0.62, -s * 0.08, s * 0.09, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

export function drawShark(ctx, shark, t) {
  const s = 32;
  const wag = Math.sin(t * 3.2) * 0.4;

  ctx.save();
  ctx.translate(shark.x, shark.y);
  ctx.rotate(shark.angle);

  ctx.fillStyle = '#17242b';
  ctx.beginPath();
  ctx.moveTo(-s * 0.95, 0);
  ctx.lineTo(-s * 1.6, -s * 0.55 + wag * s * 0.2);
  ctx.lineTo(-s * 1.15, 0);
  ctx.lineTo(-s * 1.6, s * 0.4 + wag * s * 0.2);
  ctx.closePath();
  ctx.fill();

  const bodyGrad = ctx.createLinearGradient(0, -s * 0.5, 0, s * 0.5);
  bodyGrad.addColorStop(0, '#17242b');
  bodyGrad.addColorStop(1, '#cfe9ec');
  ctx.fillStyle = bodyGrad;
  ctx.beginPath();
  ctx.moveTo(s * 1.3, 0);
  ctx.bezierCurveTo(s * 0.9, -s * 0.42, -s * 0.7, -s * 0.4, -s * 1.0, 0);
  ctx.bezierCurveTo(-s * 0.7, s * 0.4, s * 0.9, s * 0.42, s * 1.3, 0);
  ctx.fill();

  ctx.fillStyle = '#17242b';
  ctx.beginPath();
  ctx.moveTo(-s * 0.1, -s * 0.32);
  ctx.lineTo(s * 0.05, -s * 0.95);
  ctx.lineTo(s * 0.3, -s * 0.3);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = '#0a0a0a';
  ctx.beginPath();
  ctx.arc(s * 0.85, -s * 0.06, s * 0.045, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}
