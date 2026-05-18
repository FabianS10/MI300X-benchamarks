# MI300X Credit-Safe Run Checklist

## Local first

```bash
python main.py --quick
```

## On MI300X: smoke test first

```bash
python main.py --gpu --quick --out exports_mi300x_smoke --max-seconds 180
```

Confirm the run creates:

- `hardware_profile.json`
- `benchmark_full.json`
- `summary.json`
- `hash_benchmark.csv`
- `gpu_synthetic_benchmark.csv`
- `merkle_benchmark.csv`
- `pow_benchmark.csv`

## Full run

```bash
python main.py --gpu --mode full --out exports_mi300x_full --max-seconds 600
```

## Save evidence

Run:

```bash
rocm-smi > exports_mi300x_full/rocm_smi.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')" > exports_mi300x_full/torch_check.txt
```

Then download `exports_mi300x_full/` and destroy the instance.

## Scientific boundary

Use MI300X results as:

- GPU synthetic byte-workload baseline.
- Large-batch simulation acceleration profile.
- Hardware evidence for the project.

Do not describe GPU synthetic results as native SHA-256/SHA3/BLAKE3 hash throughput unless native GPU hash kernels are implemented.
