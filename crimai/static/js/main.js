/* ============================================================
   CrimAI — main.js
   ============================================================ */

// ---------- Theme toggle — apply before first paint to avoid flash ----------
(function () {
  if (localStorage.getItem('theme') === 'light') {
    document.documentElement.classList.add('light-mode');
  }
})();

function toggleTheme() {
  const isLight = document.documentElement.classList.toggle('light-mode');
  localStorage.setItem('theme', isLight ? 'light' : 'dark');
}

// ---------- KPI counter animation ----------
function animateCounter(el) {
  const target = parseInt(el.dataset.target, 10);
  const duration = 800;
  const start = performance.now();
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    el.textContent = Math.floor(progress * target);
    if (progress < 1) requestAnimationFrame(step);
    else el.textContent = target;
  }
  requestAnimationFrame(step);
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-target]').forEach(animateCounter);
});

// ---------- Live search ----------
function initLiveSearch(inputId, itemSelector, textSelector) {
  const input = document.getElementById(inputId);
  if (!input) return;
  input.addEventListener('input', () => {
    const q = input.value.toLowerCase();
    document.querySelectorAll(itemSelector).forEach(item => {
      const text = (textSelector
        ? item.querySelector(textSelector)?.textContent
        : item.textContent) || '';
      item.style.display = text.toLowerCase().includes(q) ? '' : 'none';
    });
  });
}

// ---------- Canvas chart helpers ----------

/**
 * Draw a bar chart on a 2D canvas context.
 * @param {CanvasRenderingContext2D} ctx
 * @param {string[]} labels
 * @param {number[]} data
 * @param {{ color?: string, bgColor?: string }} options
 */
function drawBarChart(ctx, labels, data, options = {}) {
  const { color = '#4f8ef7', bgColor = 'rgba(79,142,247,0.15)' } = options;
  const canvas = ctx.canvas;
  const W = canvas.width, H = canvas.height;
  const pad = { top: 20, right: 20, bottom: 40, left: 50 };
  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top - pad.bottom;
  const max = Math.max(...data, 1);

  ctx.clearRect(0, 0, W, H);

  const mutedColor = getComputedStyle(document.documentElement)
    .getPropertyValue('--text-muted').trim() || '#8892a4';

  ctx.font = '11px sans-serif';

  const barW = (chartW / labels.length) * 0.6;
  const gap = chartW / labels.length;

  data.forEach((val, i) => {
    const x = pad.left + i * gap + gap * 0.2;
    const barH = (val / max) * chartH;
    const y = pad.top + chartH - barH;

    // Background column
    ctx.fillStyle = bgColor;
    ctx.fillRect(x, pad.top, barW, chartH);

    // Value bar
    ctx.fillStyle = color;
    ctx.fillRect(x, y, barW, barH);

    // Label
    ctx.fillStyle = mutedColor;
    ctx.textAlign = 'center';
    ctx.fillText(labels[i], x + barW / 2, H - 10);
  });
}

/**
 * Draw a line chart on a 2D canvas context.
 * @param {CanvasRenderingContext2D} ctx
 * @param {string[]} labels
 * @param {number[]} data
 * @param {{ color?: string, fill?: boolean }} options
 */
function drawLineChart(ctx, labels, data, options = {}) {
  const { color = '#22c55e', fill = true } = options;
  const canvas = ctx.canvas;
  const W = canvas.width, H = canvas.height;
  const pad = { top: 20, right: 20, bottom: 40, left: 50 };
  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top - pad.bottom;
  const max = Math.max(...data, 1);

  ctx.clearRect(0, 0, W, H);
  if (data.length < 2) return;

  const pts = data.map((v, i) => ({
    x: pad.left + (i / (data.length - 1)) * chartW,
    y: pad.top + chartH - (v / max) * chartH,
  }));

  // Fill area under line
  if (fill) {
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pad.top + chartH);
    pts.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.lineTo(pts[pts.length - 1].x, pad.top + chartH);
    ctx.closePath();
    ctx.fillStyle = color.replace(')', ', 0.15)').replace('rgb', 'rgba');
    ctx.fill();
  }

  // Line
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  pts.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
  ctx.stroke();

  // Labels
  const mutedColor = getComputedStyle(document.documentElement)
    .getPropertyValue('--text-muted').trim() || '#8892a4';
  ctx.fillStyle = mutedColor;
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  labels.forEach((l, i) => ctx.fillText(l, pts[i].x, H - 10));
}

/**
 * Draw a donut chart on a 2D canvas context.
 * @param {CanvasRenderingContext2D} ctx
 * @param {string[]} labels
 * @param {number[]} data
 * @param {{ colors?: string[] }} options
 */
function drawDonutChart(ctx, labels, data, options = {}) {
  const colors = options.colors || [
    '#4f8ef7', '#22c55e', '#f59e0b', '#ef4444', '#38bdf8', '#a78bfa',
  ];
  const canvas = ctx.canvas;
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H / 2;
  const r = Math.min(W, H) / 2 - 20;
  const inner = r * 0.55;
  const total = data.reduce((a, b) => a + b, 0) || 1;

  ctx.clearRect(0, 0, W, H);

  let angle = -Math.PI / 2;
  data.forEach((val, i) => {
    const slice = (val / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, angle, angle + slice);
    ctx.closePath();
    ctx.fillStyle = colors[i % colors.length];
    ctx.fill();
    angle += slice;
  });

  // Donut hole
  ctx.beginPath();
  ctx.arc(cx, cy, inner, 0, Math.PI * 2);
  ctx.fillStyle =
    getComputedStyle(document.documentElement)
      .getPropertyValue('--surface').trim() || '#1a1d27';
  ctx.fill();
}
