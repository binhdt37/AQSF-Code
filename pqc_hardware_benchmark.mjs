import os from 'node:os';
import fs from 'node:fs';
import { performance } from 'node:perf_hooks';
import { ml_kem512, ml_kem768, ml_kem1024 } from '@noble/post-quantum/ml-kem.js';
import {
  slh_dsa_sha2_128s,
  slh_dsa_sha2_128f,
  slh_dsa_sha2_192s,
  slh_dsa_sha2_192f,
  slh_dsa_sha2_256s,
  slh_dsa_sha2_256f,
} from '@noble/post-quantum/slh-dsa.js';

const encoder = new TextEncoder();
const message = encoder.encode('A-QSF reproducible PQC hardware benchmark v5');

function stats(samples) {
  const x = [...samples].sort((a, b) => a - b);
  const mean = x.reduce((a, b) => a + b, 0) / x.length;
  const variance = x.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, x.length - 1);
  const quantile = (p) => {
    const pos = (x.length - 1) * p;
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    return lo === hi ? x[lo] : x[lo] + (pos - lo) * (x[hi] - x[lo]);
  };
  return { mean, median: quantile(0.5), p95: quantile(0.95), std: Math.sqrt(variance), min: x[0], max: x.at(-1) };
}

function measure(operation, iterations, warmups = 2) {
  for (let i = 0; i < warmups; i++) operation();
  const samples = [];
  for (let i = 0; i < iterations; i++) {
    const t0 = performance.now();
    operation();
    samples.push(performance.now() - t0);
  }
  return stats(samples);
}

const rows = [];
function addRow(algorithm, operation, iterations, sizes, result) {
  rows.push({
    algorithm, operation, iterations,
    mean_ms: result.mean, median_ms: result.median, p95_ms: result.p95,
    std_ms: result.std, min_ms: result.min, max_ms: result.max,
    public_key_bytes: sizes.publicKey ?? '', secret_key_bytes: sizes.secretKey ?? '',
    ciphertext_bytes: sizes.cipherText ?? '', signature_bytes: sizes.signature ?? '',
  });
}

for (const [name, kem] of [
  ['ML-KEM-512', ml_kem512],
  ['ML-KEM-768', ml_kem768],
  ['ML-KEM-1024', ml_kem1024],
]) {
  const keys = kem.keygen();
  const encapsulated = kem.encapsulate(keys.publicKey);
  const recovered = kem.decapsulate(encapsulated.cipherText, keys.secretKey);
  if (!Buffer.from(recovered).equals(Buffer.from(encapsulated.sharedSecret))) {
    throw new Error(`${name}: shared-secret validation failed`);
  }
  const sizes = { publicKey: keys.publicKey.length, secretKey: keys.secretKey.length, cipherText: encapsulated.cipherText.length };
  addRow(name, 'keygen', 80, sizes, measure(() => kem.keygen(), 80, 5));
  addRow(name, 'encapsulate', 200, sizes, measure(() => kem.encapsulate(keys.publicKey), 200, 10));
  addRow(name, 'decapsulate', 200, sizes, measure(() => kem.decapsulate(encapsulated.cipherText, keys.secretKey), 200, 10));
}

for (const [name, sig, signIterations] of [
  ['SLH-DSA-SHA2-128s', slh_dsa_sha2_128s, 12],
  ['SLH-DSA-SHA2-128f', slh_dsa_sha2_128f, 20],
  ['SLH-DSA-SHA2-192s', slh_dsa_sha2_192s, 8],
  ['SLH-DSA-SHA2-192f', slh_dsa_sha2_192f, 14],
  ['SLH-DSA-SHA2-256s', slh_dsa_sha2_256s, 6],
  ['SLH-DSA-SHA2-256f', slh_dsa_sha2_256f, 10],
]) {
  const keys = sig.keygen();
  const signature = sig.sign(message, keys.secretKey);
  if (!sig.verify(signature, message, keys.publicKey)) {
    throw new Error(`${name}: signature validation failed`);
  }
  const sizes = { publicKey: keys.publicKey.length, secretKey: keys.secretKey.length, signature: signature.length };
  addRow(name, 'keygen', Math.max(6, signIterations), sizes, measure(() => sig.keygen(), Math.max(6, signIterations), 1));
  addRow(name, 'sign', signIterations, sizes, measure(() => sig.sign(message, keys.secretKey), signIterations, 1));
  addRow(name, 'verify', Math.max(20, signIterations), sizes, measure(() => sig.verify(signature, message, keys.publicKey), Math.max(20, signIterations), 2));
}

const metadata = {
  timestamp_utc: new Date().toISOString(),
  hostname: os.hostname(),
  cpu_model: os.cpus()[0]?.model ?? 'unknown',
  logical_cpus_visible: os.cpus().length,
  architecture: os.arch(),
  operating_system: `${os.type()} ${os.release()}`,
  node_version: process.version,
  implementation: '@noble/post-quantum 0.6.1 (pure JavaScript, FIPS 203/205 API)',
  benchmark_scope: 'single-thread operation latency; warm cache; no energy instrumentation',
  message_bytes: message.length,
};

const header = Object.keys(rows[0]);
const csv = [header.join(','), ...rows.map((row) => header.map((key) => row[key]).join(','))].join('\n') + '\n';
fs.writeFileSync(new URL('./pqc_hardware_benchmark.csv', import.meta.url), csv);
fs.writeFileSync(new URL('./pqc_hardware_benchmark_metadata.json', import.meta.url), JSON.stringify({ metadata, rows }, null, 2) + '\n');
console.log(JSON.stringify({ metadata, rows }, null, 2));
