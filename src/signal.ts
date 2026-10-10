import { FS, NEEDS, WINDOW_SAMPLES, WINDOW_SEC } from "./needs";

export function mulberry32(seed: number) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function generateTrial(hz: number, noise = 0.85, seed = 1, samples = WINDOW_SAMPLES) {
  const rand = mulberry32(seed);
  const x = new Float64Array(samples);
  for (let n = 0; n < samples; n++) {
    const t = n / FS;
    const ssvep =
      Math.sin(2 * Math.PI * hz * t) +
      0.45 * Math.sin(2 * Math.PI * 2 * hz * t) +
      0.18 * Math.sin(2 * Math.PI * 3 * hz * t);
    const drift = 0.15 * Math.sin(2 * Math.PI * 0.4 * t);
    x[n] = ssvep + drift + noise * (rand() * 2 - 1);
  }
  return x;
}

function mean(x: ArrayLike<number>) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i];
  return s / x.length;
}

function corr(a: ArrayLike<number>, b: ArrayLike<number>) {
  const n = a.length;
  const ma = mean(a);
  const mb = mean(b);
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i++) {
    const xa = a[i] - ma;
    const xb = b[i] - mb;
    num += xa * xb;
    da += xa * xa;
    db += xb * xb;
  }
  const den = Math.sqrt(da * db);
  return den < 1e-12 ? 0 : num / den;
}

function refPair(hz: number, samples: number, harmonic = 1) {
  const sin = new Float64Array(samples);
  const cos = new Float64Array(samples);
  const f = hz * harmonic;
  for (let n = 0; n < samples; n++) {
    const t = n / FS;
    sin[n] = Math.sin(2 * Math.PI * f * t);
    cos[n] = Math.cos(2 * Math.PI * f * t);
  }
  return { sin, cos };
}

export function ccaScore(x: ArrayLike<number>, hz: number) {
  const { sin, cos } = refPair(hz, x.length, 1);
  const cs = corr(x, sin);
  const cc = corr(x, cos);
  return Math.sqrt(cs * cs + cc * cc);
}

export function fbccaScore(x: ArrayLike<number>, hz: number) {
  const weights = [1, 0.8, 0.4];
  let acc = 0;
  let wsum = 0;
  for (let h = 1; h <= 3; h++) {
    const { sin, cos } = refPair(hz, x.length, h);
    const s = Math.sqrt(corr(x, sin) ** 2 + corr(x, cos) ** 2);
    acc += weights[h - 1] * s;
    wsum += weights[h - 1];
  }
  return acc / wsum;
}

export type ModelName = "CCA" | "FBCCA" | "FFT-ML";

export function classifyCca(x: ArrayLike<number>) {
  return argmaxScores(NEEDS.map((need) => ccaScore(x, need.hz)));
}

export function classifyFbcca(x: ArrayLike<number>) {
  return argmaxScores(NEEDS.map((need) => fbccaScore(x, need.hz)));
}

function argmaxScores(scores: number[]) {
  let best = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
  const max = scores[best];
  const sum = scores.reduce((a, b) => a + Math.exp(8 * b), 0);
  const conf = Math.exp(8 * max) / sum;
  return { index: best, scores, confidence: conf };
}

export function fftPowerFeatures(x: ArrayLike<number>) {
  const freqs = NEEDS.flatMap((need) => [need.hz, 2 * need.hz]);
  return freqs.map((f) => goertzelPower(x, f));
}

function goertzelPower(x: ArrayLike<number>, freq: number) {
  const k = Math.round((x.length * freq) / FS);
  const w = (2 * Math.PI * k) / x.length;
  const coeff = 2 * Math.cos(w);
  let s0 = 0;
  let s1 = 0;
  let s2 = 0;
  for (let i = 0; i < x.length; i++) {
    s0 = x[i] + coeff * s1 - s2;
    s2 = s1;
    s1 = s0;
  }
  const real = s1 - s2 * Math.cos(w);
  const imag = s2 * Math.sin(w);
  return (real * real + imag * imag) / x.length;
}

export type SoftmaxModel = {
  weights: number[][];
  bias: number[];
};

export function trainSoftmax(
  features: number[][],
  labels: number[],
  classes: number,
  epochs = 80,
  lr = 0.25,
) {
  const dim = features[0].length;
  const weights = Array.from({ length: classes }, () =>
    Array.from({ length: dim }, () => (Math.random() - 0.5) * 0.05),
  );
  const bias = Array.from({ length: classes }, () => 0);
  const losses: number[] = [];

  for (let epoch = 0; epoch < epochs; epoch++) {
    let loss = 0;
    const gW = Array.from({ length: classes }, () => Array(dim).fill(0));
    const gB = Array(classes).fill(0);

    for (let i = 0; i < features.length; i++) {
      const logits = weights.map((w, c) => dot(w, features[i]) + bias[c]);
      const probs = softmax(logits);
      loss -= Math.log(Math.max(probs[labels[i]], 1e-9));
      for (let c = 0; c < classes; c++) {
        const err = probs[c] - (c === labels[i] ? 1 : 0);
        gB[c] += err;
        for (let d = 0; d < dim; d++) gW[c][d] += err * features[i][d];
      }
    }

    const n = features.length;
    for (let c = 0; c < classes; c++) {
      bias[c] -= (lr * gB[c]) / n;
      for (let d = 0; d < dim; d++) weights[c][d] -= (lr * gW[c][d]) / n;
    }
    losses.push(loss / n);
  }

  return { weights, bias, losses };
}

export function predictSoftmax(model: SoftmaxModel, feat: number[]) {
  const logits = model.weights.map((w, c) => dot(w, feat) + model.bias[c]);
  const probs = softmax(logits);
  let index = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[index]) index = i;
  return { index, confidence: probs[index], scores: probs };
}

function dot(a: number[], b: number[]) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function softmax(logits: number[]) {
  const m = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - m));
  const z = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / z);
}

export function informationTransferRate(n: number, p: number, trialSec: number) {
  const clamped = Math.min(Math.max(p, 1 / n + 1e-6), 1 - 1e-6);
  const bits =
    Math.log2(n) +
    clamped * Math.log2(clamped) +
    (1 - clamped) * Math.log2((1 - clamped) / (n - 1));
  return (60 / trialSec) * bits;
}

export type Benchmark = {
  name: ModelName;
  accuracy: number;
  itr: number;
  latencyMs: number;
  losses?: number[];
  confusion: number[][];
};

export function runBenchmark(seed = 7): Benchmark[] {
  const trialsPerClass = 18;
  const trainX: number[][] = [];
  const trainY: number[] = [];
  const testSignals: { x: Float64Array; y: number }[] = [];
  let k = seed;

  NEEDS.forEach((need, y) => {
    for (let i = 0; i < trialsPerClass; i++) {
      const noise = 0.7 + (i % 5) * 0.08;
      const x = generateTrial(need.hz, noise, k++);
      if (i < 12) {
        trainX.push(fftPowerFeatures(x));
        trainY.push(y);
      } else {
        testSignals.push({ x, y });
      }
    }
  });

  const ml = trainSoftmax(trainX, trainY, NEEDS.length);

  const evalModel = (
    name: ModelName,
    classify: (x: Float64Array) => { index: number },
    latencyMs: number,
    losses?: number[],
  ): Benchmark => {
    const confusion = NEEDS.map(() => NEEDS.map(() => 0));
    let correct = 0;
    for (const trial of testSignals) {
      const pred = classify(trial.x).index;
      confusion[trial.y][pred] += 1;
      if (pred === trial.y) correct += 1;
    }
    const accuracy = correct / testSignals.length;
    return {
      name,
      accuracy,
      itr: informationTransferRate(NEEDS.length, accuracy, WINDOW_SEC + latencyMs / 1000),
      latencyMs,
      losses,
      confusion,
    };
  };

  return [
    evalModel("CCA", (x) => classifyCca(x), 12),
    evalModel("FBCCA", (x) => classifyFbcca(x), 18),
    evalModel("FFT-ML", (x) => predictSoftmax(ml, fftPowerFeatures(Array.from(x))), 42, ml.losses),
  ];
}
