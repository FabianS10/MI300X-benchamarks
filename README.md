# Prime ChainSentinel Benchmarks

Credit-safe benchmark suite for the Prime ChainSentinel graduation/project prototype.

## What this measures

- Real CPU SHA-256 / SHA3-256 / BLAKE3 throughput.
- CPU Merkle construction time with naive and corrected prediction errors.
- Offline toy proof-of-work timing with no networking and no real mining.
- Optional MI300X/PyTorch synthetic byte-workload baseline for simulation acceleration.

## Scientific boundary

The GPU mode currently measures a PyTorch synthetic byte-parallel workload. It does **not** measure native GPU SHA-256, SHA3, or BLAKE3 kernel throughput. Those claims would require HIP/CUDA/OpenCL hash kernels.

## Commands

```bash
# Local smoke test
python main.py --quick

# Full local CPU benchmark
python main.py --mode full --out exports_cpu_full

# MI300X smoke test
python main.py --gpu --quick --out exports_mi300x_smoke

# MI300X full benchmark
python main.py --gpu --mode full --out exports_mi300x_full
```

## Output artifacts

- `hardware_profile.json`
- `benchmark_full.json`
- `summary.json`
- `hash_benchmark.csv`
- `raw_hash_benchmark.csv`
- `gpu_synthetic_benchmark.csv` when `--gpu` is used
- `merkle_benchmark.csv`
- `pow_benchmark.csv`

## Paper finding to cite

Naive Merkle predictions based only on internal nodes `(N-1)` underestimate full construction time because the measured implementation hashes both transaction leaves and internal nodes. The corrected model uses approximately `2N-1` hash operations. Residual error after correction reflects implementation overhead, data movement, list allocation, cache behavior, and Python dispatch.
