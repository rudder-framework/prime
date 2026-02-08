// ORTHON Dynamical Atlas — Three-Act Flow Visualization
// Converted from flow_viz.jsx to vanilla JS for the ORTHON explorer.
// Act 1: Geometry (eigenvalue ellipsoid collapse)
// Act 2: Energy/Mass Flow (signal network)
// Act 3: State-Space Flow (particle flow field with FTLE)
// Supports simulated scenarios (demo) + real data from DuckDB-WASM.

(function () {
'use strict';

// ═══════════════════════════════════════════════════════
// SCENARIO DEFINITIONS
// ═══════════════════════════════════════════════════════

const SCENARIOS = {
  bearing: {
    name: 'Bearing Failure',
    signals: ['vibration', 'temperature', 'acoustic', 'pressure', 'speed'],
    getEigenvalues(p) {
      if (p < 0.5) return [1.8, 1.5, 1.2, 0.8, 0.4];
      if (p < 0.8) { const t = (p - 0.5) / 0.3; return [1.8 + t * 1.5, 1.5 - t * 0.6, 1.2 - t * 0.7, 0.8 - t * 0.5, 0.4 - t * 0.3]; }
      const t = (p - 0.8) / 0.2; return [3.3 + t * 1.5, 0.9 - t * 0.6, 0.5 - t * 0.35, 0.3 - t * 0.2, 0.1 - t * 0.08];
    },
    getFlows(p) {
      const b = [{ from: 0, to: 1, strength: 0.3 }, { from: 0, to: 2, strength: 0.2 }, { from: 3, to: 4, strength: 0.4 }];
      if (p > 0.5) { const t = (p - 0.5) / 0.5; b[0].strength = 0.3 + t * 0.6; b[1].strength = 0.2 + t * 0.5; b.push({ from: 1, to: 3, strength: t * 0.5 }); b.push({ from: 2, to: 1, strength: t * 0.3 }); }
      return b;
    },
    getDominant: (p) => p < 0.4 ? 0 : p < 0.7 ? 1 : 0,
    getSignalEnergy(p) {
      if (p <= 0.5) return [0.3, 0.2, 0.15, 0.25, 0.1];
      const t = (p - 0.5) / 0.5; return [0.3 + t * 0.5, 0.2 + t * 0.4, 0.15 + t * 0.3, 0.25 - t * 0.1, 0.1 - t * 0.05];
    },
    color: { h: 0, s: 70, l: 55 },
    phaseLabel: (p) => p < 0.5 ? 'healthy' : p < 0.8 ? 'degrading' : 'failing',
  },
  twitter: {
    name: 'Twitter Polarization',
    signals: ['echo_chamber', 'cross_ratio', 'bridge_frac', 'modularity', 'volume'],
    getEigenvalues(p) { const e1 = 1.5 + p * 2.5; return [e1, Math.max(0.1, 1.3 - p * 0.8), Math.max(0.08, 1.0 - p * 0.7), Math.max(0.05, 0.6 - p * 0.45), Math.max(0.03, 0.3 - p * 0.25)]; },
    getFlows(p) {
      const f = [{ from: 0, to: 1, strength: 0.5 + p * 0.4 }, { from: 2, to: 0, strength: Math.max(0, 0.4 - p * 0.35) }, { from: 4, to: 0, strength: 0.3 + p * 0.3 }, { from: 3, to: 1, strength: 0.2 }];
      if (p > 0.6) f.push({ from: 0, to: 4, strength: (p - 0.6) * 1.5 });
      return f;
    },
    getDominant: (p) => p < 0.5 ? 4 : 0,
    getSignalEnergy: (p) => [0.2 + p * 0.6, 0.3 - p * 0.15, Math.max(0.05, 0.25 - p * 0.2), 0.15, 0.2 + p * 0.3],
    color: { h: 270, s: 65, l: 55 },
    phaseLabel: (p) => p < 0.3 ? 'connected' : p < 0.7 ? 'separating' : 'locked in',
  },
  battery: {
    name: 'Battery Degradation',
    signals: ['capacity', 'voltage', 'impedance', 'temperature', 'charge_time'],
    getEigenvalues(p) { const d = Math.pow(p, 1.5); return [1.5 + d * 2, 1.3 - d * 0.5, 1.0 - d * 0.4, 0.7 - d * 0.3, 0.4 - d * 0.2]; },
    getFlows(p) {
      const f = [{ from: 0, to: 1, strength: 0.3 + p * 0.3 }, { from: 2, to: 0, strength: 0.2 + p * 0.4 }, { from: 3, to: 2, strength: 0.1 + p * 0.3 }];
      if (p > 0.7) f.push({ from: 3, to: 0, strength: (p - 0.7) * 1.5 });
      return f;
    },
    getDominant: (p) => p < 0.6 ? 0 : p < 0.85 ? 1 : 3,
    getSignalEnergy: (p) => [0.3 + p * 0.1, 0.15 + p * 0.25, 0.1 + p * 0.3, 0.05 + p * (p > 0.7 ? 0.6 : 0.1), 0.1 + p * 0.05],
    color: { h: 45, s: 75, l: 50 },
    phaseLabel: (p) => p < 0.4 ? 'healthy' : p < 0.8 ? 'degrading' : 'end of life',
  },
  turbofan: {
    name: 'Turbofan (Linear)',
    signals: ['fan_speed', 'lpc_outlet', 'hpc_outlet', 'lpt_outlet', 'bypass_ratio'],
    getEigenvalues: (p) => [1.6 + p * 0.2, 1.4 - p * 0.1, 1.1 - p * 0.05, 0.8, 0.4],
    getFlows: () => [{ from: 0, to: 1, strength: 0.3 }, { from: 1, to: 2, strength: 0.3 }, { from: 2, to: 3, strength: 0.2 }, { from: 4, to: 0, strength: 0.15 }],
    getDominant: () => 0,
    getSignalEnergy(p) { const f = 1 - p * 0.6; return [0.3 * f, 0.25 * f, 0.2 * f, 0.15 * f, 0.1 * f]; },
    color: { h: 210, s: 50, l: 55 },
    phaseLabel: () => 'linear degradation',
  },
};

// ═══════════════════════════════════════════════════════
// FLOW FIELD DEFINITIONS (Act 3 vector fields)
// ═══════════════════════════════════════════════════════

const FLOW_FIELDS = {
  bearing(x, y, t, p) {
    if (p < 0.4) { const dx = x - 0.5, dy = y - 0.5; return { vx: -dy * 0.3 + Math.sin(t * 2 + x * 5) * 0.02, vy: dx * 0.3 + Math.cos(t * 2 + y * 5) * 0.02, ftle: 0.002 }; }
    if (p < 0.75) { const cx = 0.5 + Math.sin(t) * 0.1, cy = 0.5 + Math.cos(t * 0.7) * 0.1, dx = x - cx, dy = y - cy, r = Math.sqrt(dx * dx + dy * dy) + 0.01, str = 0.2 + (p - 0.4) * 2; return { vx: -dy / r * str + dx * 0.1 * Math.sin(t * 3), vy: dx / r * str + dy * 0.1 * Math.cos(t * 3), ftle: 0.01 + (p - 0.4) * 0.15 }; }
    const v1x = 0.35 + Math.sin(t * 1.3) * 0.15, v1y = 0.4 + Math.cos(t * 0.9) * 0.15, v2x = 0.65 + Math.sin(t * 0.7) * 0.1, v2y = 0.6 + Math.cos(t * 1.1) * 0.1;
    const d1x = x - v1x, d1y = y - v1y, d2x = x - v2x, d2y = y - v2y, r1 = Math.sqrt(d1x * d1x + d1y * d1y) + 0.01, r2 = Math.sqrt(d2x * d2x + d2y * d2y) + 0.01;
    return { vx: -d1y / r1 * 0.5 + d2y / r2 * 0.3 + Math.sin(x * 10 + t * 4) * 0.1, vy: d1x / r1 * 0.5 - d2x / r2 * 0.3, ftle: 0.05 + (p - 0.75) * 0.2 };
  },
  twitter(x, y, t, p) {
    const sep = 0.15 + p * 0.35, lx = 0.5 - sep, rx = 0.5 + sep;
    const dlx = x - lx, drx = x - rx, dy = y - 0.5;
    const rl = Math.sqrt(dlx * dlx + dy * dy) + 0.01, rr = Math.sqrt(drx * drx + dy * dy) + 0.01;
    const str = 0.1 + p * 0.4;
    const lP = str / (rl * rl + 0.1), rP = str / (rr * rr + 0.1);
    const vx = x < 0.5 ? -dlx * lP : -drx * rP;
    const vy = x < 0.5 ? -dy * lP : -dy * rP;
    const dc = Math.abs(x - 0.5), rw = 0.08 * (1 - p * 0.7);
    return { vx: vx * 0.5, vy: vy * 0.5, ftle: 0.08 * Math.exp(-dc * dc / (2 * rw * rw)) };
  },
  battery(x, y, t, p) {
    const dx = x - 0.5, dy = y - 0.3, collapse = 0.1 + p * 0.3, spiral = 0.05 * Math.sin(t * 0.5);
    return { vx: (-dx * collapse + dy * spiral) * 0.3, vy: (-dy * collapse - dx * spiral - 0.02) * 0.3, ftle: Math.max(0, 0.03 * (1 - p)) };
  },
  turbofan(x, y, t, p) { return { vx: Math.sin(x * 3 + t) * 0.02, vy: -0.05 - 0.15 * p, ftle: 0.01 }; },
};

const ACT_INFO = {
  1: { title: 'Geometry', sub: 'What IS the system? Watch the eigenvalue ellipsoid collapse as dimensions vanish.' },
  2: { title: 'Energy Flow', sub: 'What DRIVES change? Watch energy transfer between signals as the dominant driver rotates.' },
  3: { title: 'Flow Field', sub: 'WHERE is it going? Watch the velocity field, FTLE ridges, and flow structure evolve.' },
};

// ═══════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════

let scenario = 'bearing';
let act = 1;
let progress = 0;
let time = 0;
let playing = true;
let speed = 1;
let animId = null;
let flowParticles = [];
let ellipsoidPoints = [];

const el = (id) => document.getElementById(id);

function initEllipsoidPoints() {
  ellipsoidPoints = [];
  for (let i = 0; i < 300; i++) {
    ellipsoidPoints.push({ theta: Math.random() * Math.PI * 2, phi: Math.acos(2 * Math.random() - 1), drift: Math.random() * 0.02 - 0.01 });
  }
}

function initFlowParticles() {
  flowParticles = [];
  for (let i = 0; i < 1500; i++) {
    flowParticles.push({ x: Math.random(), y: Math.random(), age: Math.random() * 180, maxAge: 120 + Math.random() * 80, px: [], py: [] });
  }
}

// ═══════════════════════════════════════════════════════
// ACT 1: GEOMETRY — Eigenvalue Ellipsoid
// ═══════════════════════════════════════════════════════

function renderGeometry(ctx, W, H) {
  const s = SCENARIOS[scenario];
  const eigenvalues = s.getEigenvalues(progress);
  const total = eigenvalues.reduce((a, b) => a + b, 0);
  const normalized = eigenvalues.map(e => e / total);
  const effDim = Math.exp(-normalized.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0));

  ctx.fillStyle = '#080a12';
  ctx.fillRect(0, 0, W, H);

  const cx = W * 0.45, cy = H * 0.5;
  const rotY = time * 0.3, rotX = 0.3 + Math.sin(time * 0.1) * 0.15;
  const cosY = Math.cos(rotY), sinY = Math.sin(rotY), cosX = Math.cos(rotX), sinX = Math.sin(rotX);
  const scale = Math.min(90, H * 0.22);
  const ax = Math.sqrt(eigenvalues[0]) * scale;
  const ay = Math.sqrt(Math.max(0.01, eigenvalues[1])) * scale;
  const az = Math.sqrt(Math.max(0.01, eigenvalues[2])) * scale;

  const project = (x3, y3, z3) => {
    let x = x3 * cosY - z3 * sinY, z = x3 * sinY + z3 * cosY;
    let y = y3 * cosX - z * sinX; z = y3 * sinX + z * cosX;
    return { x: cx + x, y: cy + y, depth: z / 300 + 1, z };
  };

  const { h, s: sat, l } = s.color;

  // Wireframe
  for (let lat = 0; lat < Math.PI; lat += Math.PI / 8) {
    ctx.beginPath();
    for (let lon = 0; lon <= Math.PI * 2; lon += 0.05) {
      const p = project(ax * Math.sin(lat) * Math.cos(lon), ay * Math.sin(lat) * Math.sin(lon), az * Math.cos(lat));
      lon === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
    }
    ctx.strokeStyle = `hsla(${h},${sat}%,${l}%,0.15)`; ctx.lineWidth = 0.5; ctx.stroke();
  }
  for (let lon = 0; lon < Math.PI * 2; lon += Math.PI / 6) {
    ctx.beginPath();
    for (let lat = 0; lat <= Math.PI; lat += 0.05) {
      const p = project(ax * Math.sin(lat) * Math.cos(lon), ay * Math.sin(lat) * Math.sin(lon), az * Math.cos(lat));
      lat === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
    }
    ctx.strokeStyle = `hsla(${h},${sat}%,${l}%,0.15)`; ctx.lineWidth = 0.5; ctx.stroke();
  }

  // Point cloud
  const sorted = ellipsoidPoints.map(pt => {
    const theta = pt.theta + time * pt.drift;
    return project(ax * Math.sin(pt.phi) * Math.cos(theta), ay * Math.sin(pt.phi) * Math.sin(theta), az * Math.cos(pt.phi));
  }).sort((a, b) => a.z - b.z);

  for (const pt of sorted) {
    const size = 1.2 + pt.depth * 1.5, alpha = 0.3 + pt.depth * 0.4;
    ctx.beginPath(); ctx.arc(pt.x, pt.y, size, 0, Math.PI * 2);
    ctx.fillStyle = `hsla(${h},${sat}%,${l + 15}%,${alpha})`; ctx.fill();
  }

  // Axis lines
  const axisLen = Math.min(140, H * 0.35);
  const axes = [
    { dir: [1, 0, 0], label: 'PC1', len: 1 },
    { dir: [0, 1, 0], label: 'PC2', len: Math.sqrt(Math.max(0.01, eigenvalues[1])) / Math.sqrt(eigenvalues[0]) },
    { dir: [0, 0, 1], label: 'PC3', len: Math.sqrt(Math.max(0.01, eigenvalues[2])) / Math.sqrt(eigenvalues[0]) },
  ];
  for (const a of axes) {
    const end = project(a.dir[0] * axisLen * a.len, a.dir[1] * axisLen * a.len, a.dir[2] * axisLen * a.len);
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(end.x, end.y);
    ctx.strokeStyle = `hsla(${h},40%,70%,0.4)`; ctx.lineWidth = 1; ctx.setLineDash([4, 4]); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = `hsla(${h},30%,75%,0.6)`; ctx.font = '11px monospace'; ctx.fillText(a.label, end.x + 5, end.y - 5);
  }

  // Eigenvalue bars (right side)
  const barX = W - 130, barW = 80, barH = 12, barGap = 4;
  ctx.fillStyle = '#94a3b8'; ctx.font = '11px monospace'; ctx.fillText('eigenvalues', barX, 25);
  for (let i = 0; i < Math.min(5, eigenvalues.length); i++) {
    const y = 40 + i * (barH + barGap), pct = normalized[i];
    ctx.fillStyle = 'rgba(30,41,59,0.5)'; ctx.fillRect(barX, y, barW, barH);
    ctx.fillStyle = i === 0 ? `hsla(${h},70%,60%,0.8)` : `hsla(${h},40%,50%,${0.6 - i * 0.1})`;
    ctx.fillRect(barX, y, barW * pct * (total / eigenvalues[0]), barH);
    ctx.fillStyle = '#94a3b8'; ctx.font = '9px monospace';
    ctx.fillText(`\u03BB${i + 1} ${(pct * 100).toFixed(0)}%`, barX + barW + 6, y + 10);
  }

  // eff_dim
  const edy = 40 + 5 * (barH + barGap) + 20;
  ctx.fillStyle = '#e2e8f0'; ctx.font = 'bold 22px monospace'; ctx.fillText(`d = ${effDim.toFixed(2)}`, barX, edy);
  ctx.fillStyle = '#64748b'; ctx.font = '10px monospace'; ctx.fillText('effective dim', barX, edy + 16);
  const maxD = Math.exp(-[0.2, 0.2, 0.2, 0.2, 0.2].reduce((s, p) => s + p * Math.log(p), 0));
  const cPct = (1 - effDim / maxD) * 100;
  ctx.fillStyle = cPct > 50 ? '#ef4444' : cPct > 25 ? '#f59e0b' : '#4ade80';
  ctx.font = 'bold 13px monospace'; ctx.fillText(`${cPct.toFixed(0)}% collapsed`, barX, edy + 36);

  return { effDim, collapsePct: cPct };
}

// ═══════════════════════════════════════════════════════
// ACT 2: ENERGY / MASS FLOW — Signal Network
// ═══════════════════════════════════════════════════════

function renderNetwork(ctx, W, H) {
  const s = SCENARIOS[scenario];
  const signals = s.signals, n = signals.length;
  const flows = s.getFlows(progress);
  const dominant = s.getDominant(progress);
  const energy = s.getSignalEnergy(progress);
  const totalE = energy.reduce((a, b) => a + b, 0);
  const { h, s: sat } = s.color;

  ctx.fillStyle = 'rgba(8,10,18,0.15)'; ctx.fillRect(0, 0, W, H);

  const cx = W * 0.42, cy = H * 0.5, radius = Math.min(130, H * 0.35);
  const nodes = signals.map((name, i) => {
    const angle = -Math.PI / 2 + (i / n) * Math.PI * 2;
    return { x: cx + Math.cos(angle) * radius, y: cy + Math.sin(angle) * radius, name, energy: energy[i], isDominant: i === dominant };
  });

  // Flow curves
  for (const flow of flows) {
    if (flow.strength < 0.05) continue;
    const from = nodes[flow.from], to = nodes[flow.to], str = flow.strength;
    const mx = (from.x + to.x) / 2, my = (from.y + to.y) / 2;
    const dx = to.x - from.x, dy = to.y - from.y, len = Math.sqrt(dx * dx + dy * dy) || 1;
    const cpx = mx + (-dy / len) * 30, cpy = my + (dx / len) * 30;

    ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.quadraticCurveTo(cpx, cpy, to.x, to.y);
    ctx.strokeStyle = `hsla(${h},50%,60%,${Math.min(0.6, str * 0.8)})`; ctx.lineWidth = 1 + str * 4; ctx.stroke();

    // Animated particles
    const pc = Math.floor(str * 5);
    for (let p = 0; p < pc; p++) {
      const t = ((time * 0.5 * (0.5 + str) + p / pc) % 1), it = 1 - t;
      const px = it * it * from.x + 2 * it * t * cpx + t * t * to.x;
      const py = it * it * from.y + 2 * it * t * cpy + t * t * to.y;
      ctx.beginPath(); ctx.arc(px, py, 1.5 + str * 2, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${h},70%,70%,${0.5 + str * 0.3})`; ctx.fill();
    }
  }

  // Nodes
  for (let i = 0; i < nodes.length; i++) {
    const nd = nodes[i], eNorm = nd.energy / (totalE + 0.01), ns = 14 + eNorm * 30;
    if (nd.isDominant) {
      const gr = ns + 15 + Math.sin(time * 3) * 5;
      const g = ctx.createRadialGradient(nd.x, nd.y, ns * 0.5, nd.x, nd.y, gr);
      g.addColorStop(0, `hsla(${h},80%,60%,0.3)`); g.addColorStop(1, `hsla(${h},80%,60%,0)`);
      ctx.beginPath(); ctx.arc(nd.x, nd.y, gr, 0, Math.PI * 2); ctx.fillStyle = g; ctx.fill();
    }
    ctx.beginPath(); ctx.arc(nd.x, nd.y, ns, 0, Math.PI * 2);
    ctx.fillStyle = `hsla(${h},${nd.isDominant ? 70 : 40}%,${nd.isDominant ? 55 : 35}%,${0.5 + eNorm})`;
    ctx.fill(); ctx.strokeStyle = `hsla(${h},50%,65%,0.5)`; ctx.lineWidth = nd.isDominant ? 2 : 1; ctx.stroke();
    ctx.fillStyle = nd.isDominant ? '#e2e8f0' : '#94a3b8'; ctx.font = `${nd.isDominant ? 'bold ' : ''}11px monospace`;
    ctx.textAlign = 'center'; ctx.fillText(nd.name, nd.x, nd.y + ns + 16);
    ctx.fillStyle = '#64748b'; ctx.font = '9px monospace'; ctx.fillText(`${(eNorm * 100).toFixed(0)}%`, nd.x, nd.y + 4);
    ctx.textAlign = 'left';
  }

  // Energy bars (right)
  const barX = W - 200, barW = 150, barH = 14, barGap = 6;
  ctx.fillStyle = '#94a3b8'; ctx.font = '11px monospace'; ctx.fillText('signal energy (|v|)', barX, 25);
  const maxE = Math.max(...energy);
  for (let i = 0; i < signals.length; i++) {
    const y = 40 + i * (barH + barGap), pct = energy[i] / maxE;
    ctx.fillStyle = 'rgba(30,41,59,0.4)'; ctx.fillRect(barX, y, barW, barH);
    ctx.fillStyle = `hsla(${i === dominant ? h : h + 20},${i === dominant ? 70 : 40}%,${i === dominant ? 55 : 40}%,0.8)`;
    ctx.fillRect(barX, y, barW * pct, barH);
    ctx.fillStyle = i === dominant ? '#e2e8f0' : '#64748b'; ctx.font = '9px monospace'; ctx.fillText(signals[i], barX + 4, y + 11);
  }
  ctx.fillStyle = '#e2e8f0'; ctx.font = 'bold 12px monospace';
  ctx.fillText(`driving: ${signals[dominant]}`, barX, 40 + signals.length * (barH + barGap) + 20);

  return { dominant: signals[dominant] };
}

// ═══════════════════════════════════════════════════════
// ACT 3: FLOW FIELD — State-Space Dynamics
// ═══════════════════════════════════════════════════════

function renderFlowField(ctx, W, H) {
  const s = SCENARIOS[scenario];
  const { h, s: sat } = s.color;
  const getField = FLOW_FIELDS[scenario];
  const dt = 0.016;

  ctx.fillStyle = 'rgba(8,10,18,0.12)'; ctx.fillRect(0, 0, W, H);

  // FTLE background glow
  const gs = 25;
  for (let gx = 0; gx < W; gx += gs) {
    for (let gy = 0; gy < H; gy += gs) {
      const f = getField(gx / W, gy / H, time, progress);
      const fi = Math.min(1, f.ftle * 12);
      if (fi > 0.05) {
        ctx.fillStyle = `rgba(${200 + fi * 55 | 0},${30 + fi * 40 | 0},20,${fi * 0.12})`;
        ctx.fillRect(gx, gy, gs, gs);
      }
    }
  }

  // Particles
  let maxFTLE = 0;
  for (const p of flowParticles) {
    const f = getField(p.x, p.y, time, progress);
    p.px.push(p.x); p.py.push(p.y);
    if (p.px.length > 10) { p.px.shift(); p.py.shift(); }
    p.x += f.vx * dt; p.y += f.vy * dt; p.age++;
    if (f.ftle > maxFTLE) maxFTLE = f.ftle;

    if (p.x < -0.05 || p.x > 1.05 || p.y < -0.05 || p.y > 1.05 || p.age > p.maxAge) {
      p.x = Math.random(); p.y = Math.random(); p.age = 0; p.px = []; p.py = [];
      continue;
    }

    const life = 1 - p.age / p.maxAge;
    const spd = Math.sqrt(f.vx * f.vx + f.vy * f.vy);

    if (p.px.length > 2) {
      ctx.beginPath(); ctx.moveTo(p.px[0] * W, p.py[0] * H);
      for (let j = 1; j < p.px.length; j++) ctx.lineTo(p.px[j] * W, p.py[j] * H);
      ctx.lineTo(p.x * W, p.y * H);
      ctx.strokeStyle = `hsla(${h},${60 + f.ftle * 400}%,${40 + Math.min(1, spd * 3) * 35}%,${life * 0.5})`;
      ctx.lineWidth = 1.2; ctx.stroke();
    }

    ctx.beginPath(); ctx.arc(p.x * W, p.y * H, 1.3, 0, Math.PI * 2);
    ctx.fillStyle = `hsla(${h},80%,${50 + Math.min(1, spd * 4) * 40}%,${life})`; ctx.fill();

    if (f.ftle > 0.02) {
      const gr = f.ftle * 50;
      const grd = ctx.createRadialGradient(p.x * W, p.y * H, 0, p.x * W, p.y * H, gr);
      grd.addColorStop(0, `rgba(255,80,40,${f.ftle * 2 * life})`); grd.addColorStop(1, 'rgba(255,40,20,0)');
      ctx.beginPath(); ctx.arc(p.x * W, p.y * H, gr, 0, Math.PI * 2); ctx.fillStyle = grd; ctx.fill();
    }
  }

  return { ftle: maxFTLE };
}

// ═══════════════════════════════════════════════════════
// MAIN RENDER LOOP
// ═══════════════════════════════════════════════════════

function render() {
  const canvas = el('flow-canvas');
  if (!canvas || !canvas.parentElement) return;

  const rect = canvas.parentElement.getBoundingClientRect();
  const topBar = 30 + 25 + 44; // scenario bar + subtitle + controls bar
  canvas.width = rect.width;
  canvas.height = rect.height - topBar;
  if (canvas.width < 10 || canvas.height < 10) return;

  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  let metrics = {};
  if (act === 1) metrics = renderGeometry(ctx, W, H) || {};
  else if (act === 2) metrics = renderNetwork(ctx, W, H) || {};
  else if (act === 3) metrics = renderFlowField(ctx, W, H) || {};

  // Phase label
  const s = SCENARIOS[scenario];
  ctx.fillStyle = '#94a3b8'; ctx.font = '12px system-ui'; ctx.textAlign = 'left';
  ctx.fillText(s.phaseLabel(progress), 16, H - 16);

  // Update status overlay
  const effDim = metrics.effDim || (s.getEigenvalues(progress).reduce((a, b) => a + b, 0) > 0 ?
    (() => { const ev = s.getEigenvalues(progress), t = ev.reduce((a, b) => a + b, 0), n = ev.map(e => e / t); return Math.exp(-n.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0)); })() : 0);
  const collapse = metrics.collapsePct || ((1 - effDim / 5) * 100);
  const dom = metrics.dominant || s.signals[s.getDominant(progress)];

  const edEl = el('flow-effdim'); if (edEl) edEl.textContent = effDim.toFixed(2);
  const cEl = el('flow-collapse'); if (cEl) {
    cEl.textContent = `${collapse.toFixed(0)}%`;
    cEl.style.color = collapse > 50 ? '#f85149' : collapse > 25 ? '#d29922' : '#3fb950';
  }
  const dEl = el('flow-driver'); if (dEl) dEl.textContent = dom;
  const ftEl = el('flow-ftle'); if (ftEl) ftEl.textContent = (metrics.ftle || 0).toFixed(4);
  const phEl = el('flow-phase'); if (phEl) {
    phEl.textContent = s.phaseLabel(progress);
    const phase = s.phaseLabel(progress);
    phEl.style.color = phase.includes('fail') || phase.includes('end') || phase.includes('locked') ? '#f85149' :
                        phase.includes('degrad') || phase.includes('separat') ? '#d29922' : '#3fb950';
  }
}

function tick() {
  const dt = 0.016 * speed;
  time += dt;
  if (playing) {
    progress = Math.min(1, progress + dt * 0.008);
    const timeline = el('flow-timeline');
    if (timeline) timeline.value = Math.round(progress * 1000);
    const idx = el('flow-index');
    if (idx) idx.textContent = `I = ${Math.floor(progress * 960)}`;
    if (progress >= 1) { progress = 0; time = 0; }
  }
  render();
  animId = requestAnimationFrame(tick);
}

// ═══════════════════════════════════════════════════════
// REAL DATA MODE (DuckDB)
// ═══════════════════════════════════════════════════════

let realDataMode = false;
let realFlowField = null;
let realRenderer = null;

const URGENCY_COLORS = { nominal: { r: 78, g: 204, b: 163 }, warning: { r: 255, g: 211, b: 105 }, elevated: { r: 255, g: 107, b: 53 }, critical: { r: 233, g: 69, b: 96 } };

class RealFlowField {
  constructor(conn, cohort) { this.conn = conn; this.cohort = cohort; this.projected = []; this.projectedByI = {}; this.velocityByI = {}; this.ftleByI = {}; this.geometryByI = {}; this.ridgeByI = {}; this.minI = 0; this.maxI = 0; }
  async query(sql) { const r = await this.conn.query(sql); return r.toArray().map(row => row.toJSON()); }
  async preload() {
    const geom = await this.query(`SELECT * FROM geometry_full WHERE cohort='${this.cohort}' ORDER BY I`);
    if (!geom.length) throw new Error(`No geometry_full for ${this.cohort}`);
    this.geometryByI = Object.fromEntries(geom.map(r => [r.I, r]));
    const iv = geom.map(r => r.I).sort((a, b) => a - b);
    this.minI = iv[0]; this.maxI = iv[iv.length - 1];
    try { const v = await this.query(`SELECT * FROM velocity_field WHERE cohort='${this.cohort}' ORDER BY I`); this.velocityByI = Object.fromEntries(v.map(r => [r.I, r])); } catch (e) {}
    try { const f = await this.query(`SELECT I,AVG(ftle) as avg_ftle,MAX(ftle) as max_ftle FROM ftle_rolling WHERE cohort='${this.cohort}' GROUP BY I ORDER BY I`); this.ftleByI = Object.fromEntries(f.map(r => [r.I, r])); } catch (e) {}
    try { const r = await this.query(`SELECT I,MAX(urgency) as max_urgency,(SELECT urgency_class FROM ridge_proximity r2 WHERE r2.cohort='${this.cohort}' AND r2.I=ridge_proximity.I ORDER BY urgency DESC LIMIT 1) as urgency_class FROM ridge_proximity WHERE cohort='${this.cohort}' GROUP BY I ORDER BY I`); this.ridgeByI = Object.fromEntries(r.map(r2 => [r2.I, r2])); } catch (e) {}
    this.projected = geom.map(r => ({ I: r.I, x: r.effective_dim || 0, y: r.eigenvalue_entropy || 0 }));
    const xs = this.projected.map(p => p.x), ys = this.projected.map(p => p.y);
    const xR = Math.max(...xs) - Math.min(...xs) || 1, yR = Math.max(...ys) - Math.min(...ys) || 1;
    const xM = Math.min(...xs), yM = Math.min(...ys);
    for (const p of this.projected) { p.x = 0.1 + (p.x - xM) / xR * 0.8; p.y = 0.1 + (p.y - yM) / yR * 0.8; }
    this.projectedByI = Object.fromEntries(this.projected.map(p => [p.I, p]));
  }
  getField(I) {
    const v = this.velocityByI[I] || {}, f = this.ftleByI[I] || {}, g = this.geometryByI[I] || {}, r = this.ridgeByI[I] || {};
    return { speed: v.speed || 0, dominant_motion_signal: v.dominant_motion_signal || '', ftle: f.avg_ftle || 0, effective_dim: g.effective_dim || 0, urgency_class: r.urgency_class || 'nominal' };
  }
}

function renderRealData(ctx, W, H) {
  if (!realFlowField) return;
  const ff = realFlowField;
  ctx.fillStyle = '#0d1117'; ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = 'rgba(48,54,61,0.4)'; ctx.lineWidth = 1;
  for (let x = 0; x < W; x += 60) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
  for (let y = 0; y < H; y += 60) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

  const currentIdx = Math.round(progress * (ff.projected.length - 1));
  const trailLen = 100;
  const startIdx = Math.max(0, currentIdx - trailLen);

  // FTLE glow
  for (let i = startIdx; i <= currentIdx; i++) {
    const p = ff.projected[i]; if (!p) continue;
    const field = ff.getField(p.I), ftle = field.ftle;
    if (ftle < 0.01) continue;
    const int = Math.min(1, ftle * 20), r = 15 + int * 50, x = p.x * W, y = p.y * H;
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, `rgba(255,60,30,${int * 0.25})`); g.addColorStop(1, 'rgba(255,30,15,0)');
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fillStyle = g; ctx.fill();
  }

  // Trail
  let prev = null;
  for (let i = startIdx; i <= currentIdx; i++) {
    const p = ff.projected[i]; if (!p) continue;
    if (prev) {
      const field = ff.getField(p.I), pr = (i - startIdx) / (currentIdx - startIdx + 1), alpha = 0.2 + pr * 0.8;
      const uc = URGENCY_COLORS[field.urgency_class] || URGENCY_COLORS.nominal;
      ctx.beginPath(); ctx.moveTo(prev.x * W, prev.y * H); ctx.lineTo(p.x * W, p.y * H);
      ctx.strokeStyle = `rgba(${uc.r},${uc.g},${uc.b},${alpha})`; ctx.lineWidth = 1 + Math.min(4, (field.speed || 0) * 2); ctx.lineCap = 'round'; ctx.stroke();
    }
    prev = p;
  }

  // Current position
  const cp = ff.projected[currentIdx]; if (cp) {
    const field = ff.getField(cp.I), x = cp.x * W, y = cp.y * H;
    const uc = URGENCY_COLORS[field.urgency_class] || URGENCY_COLORS.nominal;
    const gr = 18 + Math.min(40, (field.speed || 0) * 25);
    const g = ctx.createRadialGradient(x, y, 0, x, y, gr);
    g.addColorStop(0, `rgba(${uc.r},${uc.g},${uc.b},0.45)`); g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.beginPath(); ctx.arc(x, y, gr, 0, Math.PI * 2); ctx.fillStyle = g; ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fillStyle = `rgb(${uc.r},${uc.g},${uc.b})`; ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.fill();

    // Update status
    const edEl = el('flow-effdim'); if (edEl) edEl.textContent = (field.effective_dim || 0).toFixed(2);
    const dEl = el('flow-driver'); if (dEl) dEl.textContent = field.dominant_motion_signal || '--';
    const ftEl = el('flow-ftle'); if (ftEl) ftEl.textContent = (field.ftle || 0).toFixed(4);
    const phEl = el('flow-phase'); if (phEl) { phEl.textContent = field.urgency_class; phEl.style.color = field.urgency_class === 'critical' ? '#f85149' : field.urgency_class === 'warning' ? '#d29922' : '#3fb950'; }
  }

  // Axes
  ctx.fillStyle = 'rgba(139,148,158,0.7)'; ctx.font = '10px monospace';
  ctx.fillText('effective_dim \u2192', W - 110, H - 8);
  ctx.save(); ctx.translate(12, 90); ctx.rotate(-Math.PI / 2); ctx.fillText('\u2190 eigenvalue_entropy', 0, 0); ctx.restore();
}

async function tryLoadRealData() {
  const conn = window._orthonConn?.();
  if (!conn) return false;
  const tables = ['geometry_full', 'velocity_field', 'ftle_rolling'];
  for (const t of tables) { try { await conn.query(`SELECT 1 FROM ${t} LIMIT 1`); } catch (e) { return false; } }

  // Populate cohort selector
  try {
    const rows = await conn.query('SELECT DISTINCT cohort FROM geometry_full ORDER BY cohort');
    const cohorts = rows.toArray().map(r => r.toJSON().cohort);
    const sel = el('flow-cohort');
    if (sel && cohorts.length > 0) {
      sel.innerHTML = cohorts.map(c => `<option value="${c}">${c} (real)</option>`).join('');
      sel.onchange = async () => { await switchToRealCohort(conn, sel.value); };
      await switchToRealCohort(conn, cohorts[0]);
    }
  } catch (e) { console.warn('Failed to load real cohorts:', e); return false; }
  return true;
}

async function switchToRealCohort(conn, cohort) {
  try {
    realFlowField = new RealFlowField(conn, cohort);
    await realFlowField.preload();
    realDataMode = true;
  } catch (e) { console.error('Real data load failed:', e); realDataMode = false; }
}

// ═══════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════

window.initFlowViz = function () {
  if (window._flowInitialized) return;
  window._flowInitialized = true;

  initEllipsoidPoints();
  initFlowParticles();

  // Act tabs
  document.querySelectorAll('.flow-act-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      act = parseInt(btn.dataset.act);
      document.querySelectorAll('.flow-act-tab').forEach(b => {
        b.style.background = 'transparent'; b.style.color = 'var(--dim)'; b.classList.remove('active');
      });
      btn.style.background = 'var(--input)'; btn.style.color = 'var(--accent)'; btn.classList.add('active');
      const sub = el('flow-subtitle');
      if (sub) sub.textContent = ACT_INFO[act].sub;
      if (act === 3) initFlowParticles();
    });
  });

  // Scenario selector
  const scenSel = el('flow-scenario');
  if (scenSel) {
    scenSel.addEventListener('change', () => {
      scenario = scenSel.value; progress = 0; time = 0;
      initEllipsoidPoints(); initFlowParticles();
    });
  }

  // Playback
  const playBtn = el('flow-play-btn');
  if (playBtn) playBtn.addEventListener('click', () => {
    playing = !playing;
    playBtn.textContent = playing ? '\u23F8' : '\u25B6';
  });

  const timeline = el('flow-timeline');
  if (timeline) timeline.addEventListener('input', () => {
    progress = parseInt(timeline.value) / 1000;
    playing = false;
    const pb = el('flow-play-btn'); if (pb) pb.textContent = '\u25B6';
    const idx = el('flow-index'); if (idx) idx.textContent = `I = ${Math.floor(progress * 960)}`;
  });

  const resetBtn = el('flow-reset-btn');
  if (resetBtn) resetBtn.addEventListener('click', () => {
    progress = 0; time = 0; playing = true;
    const pb = el('flow-play-btn'); if (pb) pb.textContent = '\u23F8';
    initFlowParticles();
  });

  const speedSel = el('flow-speed');
  if (speedSel) speedSel.addEventListener('change', () => { speed = parseFloat(speedSel.value); });

  // Override render for real data when available
  const origRender = render;
  const patchedRender = function () {
    if (realDataMode) {
      const canvas = el('flow-canvas');
      if (!canvas || !canvas.parentElement) return;
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width; canvas.height = rect.height - 99;
      if (canvas.width < 10 || canvas.height < 10) return;
      renderRealData(canvas.getContext('2d'), canvas.width, canvas.height);
    } else {
      origRender();
    }
  };

  // Start animation
  const animLoop = () => {
    const dt = 0.016 * speed;
    time += dt;
    if (playing && !realDataMode) {
      progress = Math.min(1, progress + dt * 0.008);
      if (progress >= 1) { progress = 0; time = 0; }
    } else if (playing && realDataMode && realFlowField) {
      progress = Math.min(1, progress + dt * 0.003);
      if (progress >= 1) progress = 0;
    }
    const tl = el('flow-timeline'); if (tl) tl.value = Math.round(progress * 1000);
    const idx = el('flow-index'); if (idx) idx.textContent = `I = ${Math.floor(progress * 960)}`;
    patchedRender();
    animId = requestAnimationFrame(animLoop);
  };
  animId = requestAnimationFrame(animLoop);

  // Try loading real data
  tryLoadRealData();

  // Periodic check for newly loaded data
  const check = setInterval(async () => {
    if (realDataMode) { clearInterval(check); return; }
    const ok = await tryLoadRealData();
    if (ok) clearInterval(check);
  }, 5000);
};

// Auto-init when the Flow tab is shown
// The showView() hook in index.html calls initFlowViz() on first view

})();
