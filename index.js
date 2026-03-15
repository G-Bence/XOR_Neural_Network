document.addEventListener('DOMContentLoaded', () => {
  const dataset = [
    { input: [0, 0], target: 0, label: '(0, 0) → 0' },
    { input: [0, 1], target: 1, label: '(0, 1) → 1' },
    { input: [1, 0], target: 1, label: '(1, 0) → 1' },
    { input: [1, 1], target: 0, label: '(1, 1) → 0' }
  ];

  const dom = {
    epochsInput: document.getElementById('epochsInput'),
    learningRateInput: document.getElementById('learningRateInput'),
    speedInput: document.getElementById('speedInput'),
    sampleSelect: document.getElementById('sampleSelect'),
    thresholdInput: document.getElementById('thresholdInput'),
    startBtn: document.getElementById('startBtn'),
    pauseBtn: document.getElementById('pauseBtn'),
    stepBtn: document.getElementById('stepBtn'),
    resetBtn: document.getElementById('resetBtn'),
    randomizeBtn: document.getElementById('randomizeBtn'),
    runTestBtn: document.getElementById('runTestBtn'),
    statusText: document.getElementById('statusText'),
    runDot: document.getElementById('runDot'),
    epochText: document.getElementById('epochText'),
    epochMaxText: document.getElementById('epochMaxText'),
    sampleText: document.getElementById('sampleText'),
    lrText: document.getElementById('lrText'),
    targetText: document.getElementById('targetText'),
    outputText: document.getElementById('outputText'),
    overlayHidden: document.getElementById('overlayHidden'),
    overlayOutput: document.getElementById('overlayOutput'),
    overlayAvgError: document.getElementById('overlayAvgError'),
    absErrorMetric: document.getElementById('absErrorMetric'),
    avgErrorMetric: document.getElementById('avgErrorMetric'),
    classMetric: document.getElementById('classMetric'),
    progressMetric: document.getElementById('progressMetric'),
    forwardDetails: document.getElementById('forwardDetails'),
    weightsTableBody: document.getElementById('weightsTableBody'),
    truthTableBody: document.getElementById('truthTableBody'),
    manualOutputText: document.getElementById('manualOutputText'),
    manualClassText: document.getElementById('manualClassText'),
    networkSvg: document.getElementById('networkSvg')
  };

  const paramInputs = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'b1', 'b2', 'b3']
    .reduce((acc, id) => ({ ...acc, [id]: document.getElementById(id) }), {});

  const manualBiasInputs = {
    b1: document.getElementById('manualBias1'),
    b2: document.getElementById('manualBias2'),
    b3: document.getElementById('manualBias3')
  };

  const testInputEls = {
    x1: document.getElementById('testX1'),
    x2: document.getElementById('testX2')
  };

  const state = {
    running: false,
    timerId: null,
    epoch: 0,
    maxEpochs: 2500,
    learningRate: 0.7,
    speed: 5,
    threshold: 0.5,
    sampleIndex: 0,
    cycleIndex: 0,
    params: {
      w1: 0.25, w2: -0.45, w3: -0.35, w4: 0.40, w5: 0.30, w6: -0.20,
      b1: 0.10, b2: -0.10, b3: 0.05
    },
    history: {
      epochs: [],
      absError: [],
      avgLoss: [],
      h1: [],
      h2: [],
      y: [],
      w1: [],
      w2: [],
      w5: [],
      w6: []
    }
  };

  function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  function sigmoidDerivativeFromOutput(y) {
    return y * (1 - y);
  }

  function round(value, digits = 4) {
    return Number(value).toFixed(digits);
  }

  function clamp01(value) {
    return Math.min(1, Math.max(0, value));
  }

  function randomWeight() {
    return Number(((Math.random() * 2) - 1).toFixed(2));
  }

  function readControls() {
    state.maxEpochs = Math.max(1, parseInt(dom.epochsInput.value, 10) || 1);
    state.learningRate = Math.max(0.001, parseFloat(dom.learningRateInput.value) || 0.7);
    state.speed = Math.max(1, parseInt(dom.speedInput.value, 10) || 1);
    state.threshold = clamp01(parseFloat(dom.thresholdInput.value) || 0.5);
    dom.epochMaxText.textContent = state.maxEpochs;
    dom.lrText.textContent = round(state.learningRate, 3);
  }

  function readParamsFromInputs() {
    Object.keys(state.params).forEach((key) => {
      const parsed = parseFloat(paramInputs[key].value);
      state.params[key] = Number.isFinite(parsed) ? parsed : state.params[key];
    });
  }

  function writeParamsToInputs() {
    Object.keys(state.params).forEach((key) => {
      paramInputs[key].value = Number(state.params[key]).toFixed(2);
    });
  }

  function resetHistory() {
    Object.keys(state.history).forEach((key) => {
      state.history[key] = [];
    });
  }

  function getVisibleSample() {
    const mode = dom.sampleSelect.value;
    if (mode === 'auto') {
      return dataset[state.cycleIndex % dataset.length];
    }
    const index = parseInt(mode, 10);
    return dataset[index] || dataset[0];
  }

  function forwardPass(x1, x2, params = state.params) {
    const z1 = params.w1 * x1 + params.w2 * x2 + params.b1;
    const h1 = sigmoid(z1);
    const z2 = params.w3 * x1 + params.w4 * x2 + params.b2;
    const h2 = sigmoid(z2);
    const z3 = params.w5 * h1 + params.w6 * h2 + params.b3;
    const y = sigmoid(z3);
    return { x1, x2, z1, h1, z2, h2, z3, y };
  }

  function computeAverageLoss() {
    let total = 0;
    for (const sample of dataset) {
      const result = forwardPass(sample.input[0], sample.input[1]);
      total += 0.5 * Math.pow(sample.target - result.y, 2);
    }
    return total / dataset.length;
  }

  function predictClass(y) {
    return y >= state.threshold ? 1 : 0;
  }

  function trainOneSample(sample) {
    const [x1, x2] = sample.input;
    const target = sample.target;
    const p = state.params;
    const fp = forwardPass(x1, x2, p);

    const error = target - fp.y;
    const deltaOutput = error * sigmoidDerivativeFromOutput(fp.y);
    const deltaHidden1 = sigmoidDerivativeFromOutput(fp.h1) * p.w5 * deltaOutput;
    const deltaHidden2 = sigmoidDerivativeFromOutput(fp.h2) * p.w6 * deltaOutput;

    p.w5 += state.learningRate * deltaOutput * fp.h1;
    p.w6 += state.learningRate * deltaOutput * fp.h2;
    p.b3 += state.learningRate * deltaOutput;

    p.w1 += state.learningRate * deltaHidden1 * x1;
    p.w2 += state.learningRate * deltaHidden1 * x2;
    p.b1 += state.learningRate * deltaHidden1;

    p.w3 += state.learningRate * deltaHidden2 * x1;
    p.w4 += state.learningRate * deltaHidden2 * x2;
    p.b2 += state.learningRate * deltaHidden2;
  }

  function pushHistory() {
    const sample = getVisibleSample();
    const visible = forwardPass(sample.input[0], sample.input[1]);
    const avgLoss = computeAverageLoss();
    const absError = Math.abs(sample.target - visible.y);

    state.history.epochs.push(state.epoch);
    state.history.absError.push(absError);
    state.history.avgLoss.push(avgLoss);
    state.history.h1.push(visible.h1);
    state.history.h2.push(visible.h2);
    state.history.y.push(visible.y);
    state.history.w1.push(state.params.w1);
    state.history.w2.push(state.params.w2);
    state.history.w5.push(state.params.w5);
    state.history.w6.push(state.params.w6);

    if (state.history.epochs.length > 320) {
      Object.keys(state.history).forEach((key) => {
        state.history[key].shift();
      });
    }
  }

  function trainTick(stepCount = state.speed) {
    if (state.epoch >= state.maxEpochs) {
      stopTraining('Training complete');
      updateAllViews();
      return;
    }

    for (let i = 0; i < stepCount && state.epoch < state.maxEpochs; i += 1) {
      const sample = dataset[state.sampleIndex % dataset.length];
      trainOneSample(sample);
      state.sampleIndex += 1;
      state.epoch += 1;
      state.cycleIndex = state.sampleIndex;
      pushHistory();
    }

    updateAllViews();

    if (state.epoch >= state.maxEpochs) {
      stopTraining('Training complete');
    }
  }

  function startTraining() {
    readControls();
    readParamsFromInputs();
    if (state.running) return;
    state.running = true;
    dom.statusText.textContent = 'Running';
    dom.runDot.style.background = 'var(--green)';
    dom.runDot.style.boxShadow = '0 0 14px rgba(54, 211, 153, 0.85)';
    state.timerId = setInterval(() => trainTick(), 90);
    toggleButtons();
  }

  function stopTraining(label = 'Paused') {
    state.running = false;
    dom.statusText.textContent = label;
    dom.runDot.style.background = label === 'Training complete' ? 'var(--cyan)' : 'var(--orange)';
    dom.runDot.style.boxShadow = label === 'Training complete'
      ? '0 0 14px rgba(100, 228, 255, 0.85)'
      : '0 0 14px rgba(255, 179, 87, 0.85)';
    clearInterval(state.timerId);
    state.timerId = null;
    toggleButtons();
  }

  function resetSimulation(keepParams = false) {
    stopTraining('Idle');
    readControls();
    if (!keepParams) {
      readParamsFromInputs();
    }
    state.epoch = 0;
    state.sampleIndex = 0;
    state.cycleIndex = 0;
    resetHistory();
    pushHistory();
    updateAllViews();
  }

  function randomizeParameters() {
    Object.keys(state.params).forEach((key) => {
      state.params[key] = randomWeight();
    });
    writeParamsToInputs();
    resetSimulation(true);
  }

  function toggleButtons() {
    dom.startBtn.disabled = state.running || state.epoch >= state.maxEpochs;
    dom.pauseBtn.disabled = !state.running;
    dom.stepBtn.disabled = state.running || state.epoch >= state.maxEpochs;
  }

  function createLineChart(ctx, labelConfigs, yMin = 0, yMax = 1.05) {
    return new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: labelConfigs.map(config => ({
          label: config.label,
          data: [],
          borderColor: config.color,
          backgroundColor: config.color,
          pointRadius: 0,
          borderWidth: 2.4,
          tension: 0.24
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: {
            labels: {
              color: '#dcecff'
            }
          }
        },
        scales: {
          x: {
            ticks: { color: '#aac7ef' },
            grid: { color: 'rgba(255,255,255,0.06)' }
          },
          y: {
            min: yMin,
            max: yMax,
            ticks: { color: '#aac7ef' },
            grid: { color: 'rgba(255,255,255,0.06)' }
          }
        }
      }
    });
  }

  const errorChart = createLineChart(
    document.getElementById('errorChart'),
    [
      { label: 'Absolute error', color: '#64e4ff' },
      { label: 'Average loss', color: '#f04f9b' }
    ],
    0,
    1
  );

  const activationChart = createLineChart(
    document.getElementById('activationChart'),
    [
      { label: 'Hidden neuron h1', color: '#4ea0ff' },
      { label: 'Hidden neuron h2', color: '#36d399' },
      { label: 'Output neuron y', color: '#ffb357' }
    ],
    0,
    1
  );

  const weightChart = createLineChart(
    document.getElementById('weightChart'),
    [
      { label: 'w1', color: '#7cc1ff' },
      { label: 'w2', color: '#f04f9b' },
      { label: 'w5', color: '#36d399' },
      { label: 'w6', color: '#ffb357' }
    ],
    -4,
    4
  );

  function updateCharts() {
    const labels = state.history.epochs;

    errorChart.data.labels = labels;
    errorChart.data.datasets[0].data = state.history.absError;
    errorChart.data.datasets[1].data = state.history.avgLoss;
    errorChart.update();

    activationChart.data.labels = labels;
    activationChart.data.datasets[0].data = state.history.h1;
    activationChart.data.datasets[1].data = state.history.h2;
    activationChart.data.datasets[2].data = state.history.y;
    activationChart.update();

    weightChart.data.labels = labels;
    weightChart.data.datasets[0].data = state.history.w1;
    weightChart.data.datasets[1].data = state.history.w2;
    weightChart.data.datasets[2].data = state.history.w5;
    weightChart.data.datasets[3].data = state.history.w6;
    weightChart.update();
  }

  function weightColor(value) {
    if (value >= 0) {
      const alpha = Math.min(0.95, 0.18 + Math.abs(value) * 0.22);
      return `rgba(100, 228, 255, ${alpha})`;
    }
    const alpha = Math.min(0.95, 0.18 + Math.abs(value) * 0.22);
    return `rgba(240, 79, 155, ${alpha})`;
  }

  function connectionThickness(value) {
    return Math.max(2, Math.min(10, 2 + Math.abs(value) * 3.2));
  }

  function neuronFill(value) {
    const v = clamp01(value);
    const r = Math.round(35 + v * 65);
    const g = Math.round(90 + v * 125);
    const b = Math.round(180 + v * 55);
    return `rgb(${r}, ${g}, ${b})`;
  }

  function labelBox(x, y, text, anchor = 'middle') {
    return `
      <g>
        <rect x="${x - 48}" y="${y - 18}" rx="10" width="96" height="28" fill="rgba(6,17,31,0.88)" stroke="rgba(255,255,255,0.08)"></rect>
        <text x="${x}" y="${y + 1}" fill="#ebf5ff" font-size="18" text-anchor="${anchor}">${text}</text>
      </g>
    `;
  }

  function renderNetworkDiagram() {
    const sample = getVisibleSample();
    const [x1, x2] = sample.input;
    const fp = forwardPass(x1, x2);
    const p = state.params;
    const svg = dom.networkSvg;
    const predictedClass = predictClass(fp.y);

    const nodes = {
      x1: { x: 140, y: 160, r: 46, value: x1, label: 'x1' },
      x2: { x: 140, y: 390, r: 46, value: x2, label: 'x2' },
      h1: { x: 560, y: 160, r: 58, value: fp.h1, label: 'h1' },
      h2: { x: 560, y: 390, r: 58, value: fp.h2, label: 'h2' },
      y: { x: 1030, y: 275, r: 64, value: fp.y, label: 'y' }
    };

    const lines = [
      { x1: 186, y1: 160, x2: 502, y2: 160, w: p.w1 },
      { x1: 186, y1: 390, x2: 502, y2: 160, w: p.w2 },
      { x1: 186, y1: 160, x2: 502, y2: 390, w: p.w3 },
      { x1: 186, y1: 390, x2: 502, y2: 390, w: p.w4 },
      { x1: 618, y1: 160, x2: 966, y2: 255, w: p.w5 },
      { x1: 618, y1: 390, x2: 966, y2: 295, w: p.w6 }
    ];

    const lineMarkup = lines.map(line => `
      <line x1="${line.x1}" y1="${line.y1}" x2="${line.x2}" y2="${line.y2}"
      stroke="${weightColor(line.w)}" stroke-width="${connectionThickness(line.w)}" stroke-linecap="round" />
    `).join('');

    const nodeMarkup = Object.values(nodes).map((n) => `
      <g>
        <circle cx="${n.x}" cy="${n.y}" r="${n.r}" fill="${neuronFill(n.value)}" stroke="rgba(255,255,255,0.18)" stroke-width="3"></circle>
        <text x="${n.x}" y="${n.y - 8}" fill="#07111f" font-size="30" font-weight="800" text-anchor="middle">${n.label}</text>
        <text x="${n.x}" y="${n.y + 24}" fill="#07111f" font-size="24" font-weight="700" text-anchor="middle">${round(n.value, 3)}</text>
      </g>
    `).join('');

    const labels = [
      labelBox(360, 130, `w1 = ${round(p.w1, 3)}`),
      labelBox(350, 220, `w2 = ${round(p.w2, 3)}`),
      labelBox(350, 338, `w3 = ${round(p.w3, 3)}`),
      labelBox(360, 428, `w4 = ${round(p.w4, 3)}`),
      labelBox(800, 172, `w5 = ${round(p.w5, 3)}`),
      labelBox(800, 388, `w6 = ${round(p.w6, 3)}`),
      labelBox(560, 80, `b1 = ${round(p.b1, 3)}`),
      labelBox(560, 478, `b2 = ${round(p.b2, 3)}`),
      labelBox(1030, 190, `b3 = ${round(p.b3, 3)}`)
    ].join('');

    svg.innerHTML = `
      <defs>
        <linearGradient id="bgGlow" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="rgba(78,160,255,0.10)"/>
          <stop offset="100%" stop-color="rgba(100,228,255,0.02)"/>
        </linearGradient>
        <marker id="arrowHead" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
          <polygon points="0 0, 10 4, 0 8" fill="rgba(255,255,255,0.55)"></polygon>
        </marker>
      </defs>
      <rect x="0" y="0" width="1200" height="560" fill="url(#bgGlow)"></rect>
      <text x="86" y="52" fill="#d9edff" font-size="30">Input layer</text>
      <text x="470" y="52" fill="#d9edff" font-size="30">Hidden layer</text>
      <text x="950" y="52" fill="#d9edff" font-size="30">Output layer</text>
      ${lineMarkup}
      ${labels}
      ${nodeMarkup}
      <text x="760" y="110" fill="#eaf4ff" font-size="24">z₃ = w₅h₁ + w₆h₂ + b₃</text>
      <text x="760" y="138" fill="#eaf4ff" font-size="24">y = σ(z₃)</text>
      <text x="770" y="438" fill="#aac7ef" font-size="22">target = ${sample.target}   •   class = ${predictedClass}</text>
      <path d="M 72 510 C 250 510, 880 510, 1116 510" stroke="rgba(255,255,255,0.14)" stroke-width="2" fill="none" marker-end="url(#arrowHead)" />
      <text x="78" y="500" fill="#aac7ef" font-size="18">Forward pass direction</text>
    `;

    dom.outputText.textContent = round(fp.y, 4);
    dom.targetText.textContent = sample.target;
    dom.overlayHidden.textContent = `${round(fp.h1, 4)}, ${round(fp.h2, 4)}`;
    dom.overlayOutput.textContent = round(fp.y, 5);
    dom.sampleText.textContent = sample.label;
  }

  function updateWeightsTable() {
    const rows = Object.entries(state.params).map(([name, value]) => `
      <tr>
        <td>${name}</td>
        <td>${round(value, 6)}</td>
      </tr>
    `).join('');
    dom.weightsTableBody.innerHTML = rows;
  }

  function updateTruthTable() {
    const rows = dataset.map((sample) => {
      const result = forwardPass(sample.input[0], sample.input[1]);
      const predictedClass = predictClass(result.y);
      const isCorrect = predictedClass === sample.target;
      return `
        <tr>
          <td>(${sample.input[0]}, ${sample.input[1]})</td>
          <td>${sample.target}</td>
          <td>${round(result.y, 6)}</td>
          <td style="color:${isCorrect ? '#7ff5c4' : '#ffc298'};">${predictedClass} ${isCorrect ? '✓' : '✗'}</td>
        </tr>
      `;
    }).join('');
    dom.truthTableBody.innerHTML = rows;
  }

  function updateForwardDetails() {
    const sample = getVisibleSample();
    const [x1, x2] = sample.input;
    const fp = forwardPass(x1, x2);
    const avgLoss = computeAverageLoss();
    const absError = Math.abs(sample.target - fp.y);
    const predictedClass = predictClass(fp.y);

    dom.forwardDetails.innerHTML = `
      <strong>Visible sample:</strong> ${sample.label}<br>
      <strong>Hidden neuron 1:</strong> z₁ = ${round(fp.z1, 6)}, h₁ = ${round(fp.h1, 6)}<br>
      <strong>Hidden neuron 2:</strong> z₂ = ${round(fp.z2, 6)}, h₂ = ${round(fp.h2, 6)}<br>
      <strong>Output neuron:</strong> z₃ = ${round(fp.z3, 6)}, y = ${round(fp.y, 6)}<br>
      <strong>Absolute error:</strong> ${round(absError, 6)}<br>
      <strong>Average loss:</strong> ${round(avgLoss, 6)}<br>
      <strong>Class with threshold ${round(state.threshold, 2)}:</strong> ${predictedClass}
    `;

    dom.absErrorMetric.textContent = round(absError, 6);
    dom.avgErrorMetric.textContent = round(avgLoss, 6);
    dom.overlayAvgError.textContent = round(avgLoss, 6);
    dom.classMetric.textContent = predictedClass;
    dom.epochText.textContent = state.epoch;
    dom.progressMetric.textContent = `${Math.min(100, Math.round((state.epoch / state.maxEpochs) * 100))}%`;
  }

  function drawBoundary() {
    const canvas = document.getElementById('boundaryCanvas');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(320, Math.floor(rect.width * dpr));
    const height = Math.max(260, Math.floor(rect.height * dpr));
    canvas.width = width;
    canvas.height = height;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const logicalW = rect.width;
    const logicalH = rect.height;
    ctx.scale(dpr, dpr);

    const resolution = 70;
    const cellW = logicalW / resolution;
    const cellH = logicalH / resolution;

    for (let gy = 0; gy < resolution; gy += 1) {
      for (let gx = 0; gx < resolution; gx += 1) {
        const x = gx / (resolution - 1);
        const y = 1 - gy / (resolution - 1);
        const out = forwardPass(x, y).y;
        const magenta = { r: 103, g: 23, b: 78 };
        const green = { r: 48, g: 98, b: 34 };
        const mixR = Math.round(magenta.r * (1 - out) + green.r * out);
        const mixG = Math.round(magenta.g * (1 - out) + green.g * out);
        const mixB = Math.round(magenta.b * (1 - out) + green.b * out);
        ctx.fillStyle = `rgba(${mixR}, ${mixG}, ${mixB}, 0.92)`;
        ctx.fillRect(gx * cellW, gy * cellH, cellW + 1, cellH + 1);
      }
    }

    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 16; i += 1) {
      const x = (logicalW / 16) * i;
      const y = (logicalH / 16) * i;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, logicalH);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(logicalW, y);
      ctx.stroke();
    }

    ctx.strokeStyle = 'rgba(240, 247, 255, 0.9)';
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.moveTo(60, logicalH - 54);
    ctx.lineTo(logicalW - 38, logicalH - 54);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(60, logicalH - 54);
    ctx.lineTo(60, 30);
    ctx.stroke();

    ctx.fillStyle = '#f1f8ff';
    ctx.font = '16px Segoe UI';
    ctx.fillText('x1', logicalW - 28, logicalH - 64);
    ctx.fillText('x2', 68, 24);

    dataset.forEach(sample => {
      const [x1, x2] = sample.input;
      const px = 60 + x1 * (logicalW - 120);
      const py = logicalH - 54 - x2 * (logicalH - 108);
      ctx.beginPath();
      ctx.arc(px, py, 16, 0, Math.PI * 2);
      ctx.fillStyle = sample.target === 1 ? '#42d26f' : '#ff006f';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgba(255,255,255,0.55)';
      ctx.stroke();
      ctx.fillStyle = '#f1f8ff';
      ctx.font = '15px Segoe UI';
      const labelOffsetX = x1 === 1 ? 16 : -48;
      const labelOffsetY = x2 === 1 ? -18 : 26;
      ctx.fillText(`(${x1}, ${x2})`, px + labelOffsetX, py + labelOffsetY);
    });

    ctx.fillStyle = '#f1f8ff';
    ctx.font = '15px Segoe UI';
    ctx.fillText('⊕ = 0', logicalW - 132, 34);
    ctx.fillText('⊕ = 1', logicalW - 132, 62);

    ctx.beginPath();
    ctx.arc(logicalW - 170, 28, 10, 0, Math.PI * 2);
    ctx.fillStyle = '#ff006f';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(logicalW - 170, 56, 10, 0, Math.PI * 2);
    ctx.fillStyle = '#42d26f';
    ctx.fill();
  }

  function runManualTest() {
    const x1 = clamp01(parseFloat(testInputEls.x1.value) || 0);
    const x2 = clamp01(parseFloat(testInputEls.x2.value) || 0);
    const params = { ...state.params };

    ['b1', 'b2', 'b3'].forEach((key) => {
      const value = parseFloat(manualBiasInputs[key].value);
      if (Number.isFinite(value)) params[key] = value;
    });

    const result = forwardPass(x1, x2, params);
    const predictedClass = predictClass(result.y);
    dom.manualOutputText.textContent = `${round(result.y, 6)}`;
    dom.manualClassText.textContent = `${predictedClass} (threshold ${round(state.threshold, 2)})`;
  }

  function updateAllViews() {
    readControls();
    renderNetworkDiagram();
    updateWeightsTable();
    updateTruthTable();
    updateForwardDetails();
    updateCharts();
    drawBoundary();
    toggleButtons();
    runManualTest();
  }

  dom.startBtn.addEventListener('click', startTraining);
  dom.pauseBtn.addEventListener('click', () => stopTraining('Paused'));
  dom.stepBtn.addEventListener('click', () => {
    readControls();
    readParamsFromInputs();
    trainTick(1);
  });
  dom.resetBtn.addEventListener('click', () => resetSimulation());
  dom.randomizeBtn.addEventListener('click', randomizeParameters);
  dom.runTestBtn.addEventListener('click', runManualTest);

  [dom.epochsInput, dom.learningRateInput, dom.speedInput, dom.sampleSelect, dom.thresholdInput].forEach((el) => {
    el.addEventListener('input', updateAllViews);
    el.addEventListener('change', updateAllViews);
  });

  Object.values(paramInputs).forEach((el) => {
    el.addEventListener('input', () => {
      if (!state.running) {
        readParamsFromInputs();
        updateAllViews();
      }
    });
  });

  [...Object.values(manualBiasInputs), ...Object.values(testInputEls)].forEach((el) => {
    el.addEventListener('input', runManualTest);
  });

  window.addEventListener('resize', drawBoundary);

  writeParamsToInputs();
  resetSimulation(true);
});
