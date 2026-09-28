// SSoG kernel for one query token, mirroring SSOGAttention in models/SSOGViT.py:
//   log a(d) = -0.5 log(2 pi) - log sigma - (d - mu)^2 / (2 sigma^2),  d = query - source
//   a = softmax over source positions of (log a / temperature), per axis and atom
//   A[j, k] = sum_r lambda_r * a_y,r[j] * a_x,r[k]
// atoms: {mu: [H][R][2], sigma: [H][R][2], lam: [H][R]}; head = -1 averages all heads.
function ssogKernel(atoms, qi, qj, grid, temperature, head) {
  const heads = head < 0 ? atoms.lam.map((_, h) => h) : [head];
  const out = new Float64Array(grid * grid);
  const LOG2PI = Math.log(2 * Math.PI);
  const axis = (q, mu, sigma) => {
    const z = new Float64Array(grid);
    let max = -Infinity;
    for (let s = 0; s < grid; s++) {
      const d = q - s - mu;
      z[s] = (-0.5 * LOG2PI - Math.log(sigma) - (d * d) / (2 * sigma * sigma)) / temperature;
      if (z[s] > max) max = z[s];
    }
    let sum = 0;
    for (let s = 0; s < grid; s++) { z[s] = Math.exp(z[s] - max); sum += z[s]; }
    for (let s = 0; s < grid; s++) z[s] /= sum;
    return z;
  };
  for (const h of heads) {
    const lam = atoms.lam[h];
    for (let r = 0; r < lam.length; r++) {
      const ay = axis(qi, atoms.mu[h][r][0], atoms.sigma[h][r][0]);
      const ax = axis(qj, atoms.mu[h][r][1], atoms.sigma[h][r][1]);
      const w = lam[r] / heads.length;
      for (let j = 0; j < grid; j++) {
        const a = w * ay[j];
        if (a === 0) continue;
        for (let k = 0; k < grid; k++) out[j * grid + k] += a * ax[k];
      }
    }
  }
  return out;
}

// Decode an 8-bit quantized tensor {lo, hi, shape, b64} into a Float32Array.
function dequantize(q) {
  const bytes = atob(q.b64);
  const out = new Float32Array(bytes.length);
  const scale = (q.hi - q.lo) / 255;
  for (let i = 0; i < bytes.length; i++) out[i] = q.lo + bytes.charCodeAt(i) * scale;
  return out;
}

// Steered atoms of one query at one layer from decoded [L, G, G, H, R, (2)] tensors.
function steeredAtoms(dec, layer, qi, qj) {
  const [, G, , H, R] = dec.mu.shape;
  const base = ((layer * G + qi) * G + qj) * H * R;
  const mu = [], sigma = [], lam = [];
  for (let h = 0; h < H; h++) {
    mu.push([]); sigma.push([]); lam.push([]);
    for (let r = 0; r < R; r++) {
      const i = base + h * R + r;
      mu[h].push([dec.mu.data[2 * i], dec.mu.data[2 * i + 1]]);
      sigma[h].push([Math.exp(dec.logsigma.data[2 * i]), Math.exp(dec.logsigma.data[2 * i + 1])]);
      lam[h].push(dec.lam.data[i]);
    }
    const total = lam[h].reduce((a, b) => a + b, 0) || 1;  // undo 8-bit rounding drift
    lam[h] = lam[h].map((v) => v / total);
  }
  return { mu, sigma, lam };
}

// Steered kernel for one query. SSoG filters rows first and columns second, so the
// row profile reaching query (qi, qj) from source (j, k) comes from the atoms at the
// intermediate position (qi, k):  A[j, k] = sum_r lam(qi,qj) a_x(qi,qj)[k] a_y(qi,k)[j]
// (SSOGAttention._dense_steered_attention). dec holds the decoded per-query atoms.
function ssogSteeredKernel(dec, layer, qi, qj, grid, temperature, head) {
  const query = steeredAtoms(dec, layer, qi, qj);
  const rowAtoms = [];
  for (let k = 0; k < grid; k++) rowAtoms.push(steeredAtoms(dec, layer, qi, k));
  const heads = head < 0 ? query.lam.map((_, h) => h) : [head];
  const out = new Float64Array(grid * grid);
  const LOG2PI = Math.log(2 * Math.PI);
  const axis = (q, mu, sigma) => {
    const z = new Float64Array(grid);
    let max = -Infinity;
    for (let s = 0; s < grid; s++) {
      const d = q - s - mu;
      z[s] = (-0.5 * LOG2PI - Math.log(sigma) - (d * d) / (2 * sigma * sigma)) / temperature;
      if (z[s] > max) max = z[s];
    }
    let sum = 0;
    for (let s = 0; s < grid; s++) { z[s] = Math.exp(z[s] - max); sum += z[s]; }
    for (let s = 0; s < grid; s++) z[s] /= sum;
    return z;
  };
  for (const h of heads) {
    for (let r = 0; r < query.lam[h].length; r++) {
      const ax = axis(qj, query.mu[h][r][1], query.sigma[h][r][1]);
      const w = query.lam[h][r] / heads.length;
      for (let k = 0; k < grid; k++) {
        const ay = axis(qi, rowAtoms[k].mu[h][r][0], rowAtoms[k].sigma[h][r][0]);
        const a = w * ax[k];
        for (let j = 0; j < grid; j++) out[j * grid + k] += a * ay[j];
      }
    }
  }
  return out;
}
