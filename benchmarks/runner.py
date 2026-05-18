"""
Prime ChainSentinel — Benchmark Runner
Reaper Eagle Technologies | Fabian Sabogal

Orchestrates the benchmark pipeline:
    1. Collect hardware profile
    2. Run real CPU hash benchmarks
    3. Optionally run GPU synthetic byte-workload benchmark
    4. Run CPU Merkle tree benchmarks using CPU real hash baselines
    5. Run Toy PoW benchmarks using CPU real SHA-256 baseline
    6. Export JSON + CSV artifacts

Scientific boundary:
    GPU synthetic benchmarks are NOT native SHA-256/SHA3/BLAKE3 throughput.
    They are MI300X PyTorch byte-workload baselines for simulation acceleration.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from . import hardware_info
    from .config import EXPORT_DIR, HASH_ALGORITHMS, MAX_RUNTIME_SECONDS, get_mode_config
    from .exporter import export_all
    from .hash_bench import run_gpu_synthetic_benchmarks, run_hash_benchmarks
    from .merkle_bench import run_merkle_benchmarks
    from .pow_bench import run_pow_benchmarks
except ImportError:  # pragma: no cover - direct script execution fallback
    import hardware_info
    from config import EXPORT_DIR, HASH_ALGORITHMS, MAX_RUNTIME_SECONDS, get_mode_config
    from exporter import export_all
    from hash_bench import run_gpu_synthetic_benchmarks, run_hash_benchmarks
    from merkle_bench import run_merkle_benchmarks
    from pow_bench import run_pow_benchmarks


def _banner() -> None:
    print("\n" + "=" * 68)
    print("  Prime ChainSentinel — Benchmark Suite")
    print("  Reaper Eagle Technologies | Fabian Sabogal")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 68)


def _section(title: str) -> None:
    print(f"\n{'─' * 68}")
    print(f"  {title}")
    print(f"{'─' * 68}")


def _time_guard(start: float, max_seconds: int = MAX_RUNTIME_SECONDS) -> None:
    elapsed = time.perf_counter() - start
    if elapsed > max_seconds:
        raise TimeoutError(
            f"Runtime guard triggered after {elapsed:.1f}s > {max_seconds}s. "
            "Stopping to protect cloud credits."
        )


def _build_cpu_hash_baselines(hash_results: list) -> dict[str, float]:
    """Use only real CPU hash benchmarks for Merkle/PoW predictions."""
    baselines: dict[str, float] = {}
    for r in hash_results:
        if getattr(r, "is_surrogate", False):
            continue
        if r.device != "cpu":
            continue
        alg = r.algorithm
        if alg not in baselines or r.peak_throughput_ops_sec > baselines[alg]:
            baselines[alg] = r.peak_throughput_ops_sec
    return baselines


def run(
    use_gpu: bool = False,
    quick: bool = False,
    mode: str | None = None,
    export_dir: str | Path = EXPORT_DIR,
    verbose: bool = True,
    require_blake3: bool = False,
    max_runtime_seconds: int = MAX_RUNTIME_SECONDS,
) -> dict:
    """Run the benchmark suite and export artifacts."""
    wall_start = time.perf_counter()
    _banner()

    selected_mode = mode or ("smoke" if quick else "full")
    cfg = get_mode_config(selected_mode)
    export_dir = Path(export_dir)

    mode_label = cfg["mode"].upper()
    device_label = "CPU + GPU synthetic" if use_gpu else "CPU"
    print(f"\n  Mode   : {mode_label}")
    print(f"  Device : {device_label}")
    print(f"  Export : {export_dir}/")
    print(f"  Guard  : {max_runtime_seconds}s")

    # 1. Hardware profile
    _section("Step 1 / 5 — Hardware Profile")
    hw = hardware_info.collect()
    hardware_info.print_summary(hw)

    if use_gpu and hw.get("gpu_backend") == "cpu_only":
        print("  ⚠  --gpu flag set but no GPU detected. GPU synthetic benchmark will be skipped.")
        use_gpu = False

    # 2. Real CPU hash benchmarks
    _time_guard(wall_start, max_runtime_seconds)
    _section("Step 2 / 5 — Real CPU Hash Benchmarks")
    print(f"  Algorithms : {HASH_ALGORITHMS}")
    print(f"  Batch sizes: {cfg['batch_sizes']}")
    print(f"  Runs/config: {cfg['runs_per_config']}  |  Warm-up: {cfg['warmup_runs']}\n")

    hash_results = run_hash_benchmarks(
        algorithms=HASH_ALGORITHMS,
        batch_sizes=cfg["batch_sizes"],
        runs=cfg["runs_per_config"],
        warmup=cfg["warmup_runs"],
        use_gpu=False,
        verbose=verbose,
        require_blake3=require_blake3,
    )

    cpu_hash_baselines = _build_cpu_hash_baselines(hash_results)
    print("\n  CPU real hash baselines (peak ops/s):")
    for alg, thr in cpu_hash_baselines.items():
        print(f"    {alg:12s}: {thr:>18,.0f}")

    # 3. Optional GPU synthetic workload
    _time_guard(wall_start, max_runtime_seconds)
    gpu_synthetic_results = []
    if use_gpu:
        _section("Step 3 / 5 — MI300X GPU Synthetic Byte-Workload Benchmark")
        print("  This is a simulation acceleration baseline, not native cryptographic hashing.\n")
        gpu_synthetic_results = run_gpu_synthetic_benchmarks(
            batch_sizes=cfg["batch_sizes"],
            runs=cfg["runs_per_config"],
            warmup=cfg["warmup_runs"],
            verbose=verbose,
        )
    else:
        _section("Step 3 / 5 — GPU Synthetic Benchmark Skipped")
        print("  Run with --gpu on MI300X to capture PyTorch byte-workload baselines.")

    # 4. Merkle benchmarks
    _time_guard(wall_start, max_runtime_seconds)
    _section("Step 4 / 5 — Merkle Tree Benchmarks")
    print(f"  Leaf counts: {cfg['merkle_sizes']}")
    print("  Algorithms : sha256, sha3_256, blake3")
    print("  Prediction : naive model + corrected 2N-1 model using CPU real hash baselines\n")

    merkle_results = run_merkle_benchmarks(
        leaf_counts=cfg["merkle_sizes"],
        algorithms=["sha256", "sha3_256", "blake3"],
        runs=cfg["runs_per_config"],
        warmup=cfg["warmup_runs"],
        hash_baselines=cpu_hash_baselines,
        verbose=verbose,
        require_blake3=require_blake3,
    )

    # 5. Toy PoW benchmarks
    _time_guard(wall_start, max_runtime_seconds)
    _section("Step 5 / 5 — Toy Proof-of-Work Benchmarks")
    print(f"  Difficulty levels (bits): {cfg['pow_difficulty_bits']}")
    print(f"  Runs/config: {cfg['runs_per_config']}\n")

    pow_results = run_pow_benchmarks(
        difficulty_bits=cfg["pow_difficulty_bits"],
        runs=cfg["runs_per_config"],
        warmup=cfg["warmup_runs"],
        hash_baselines=cpu_hash_baselines,
        verbose=verbose,
    )

    # Export
    export_paths = export_all(
        hardware_profile=hw,
        hash_results=hash_results,
        merkle_results=merkle_results,
        pow_results=pow_results,
        gpu_synthetic_results=gpu_synthetic_results,
        export_dir=export_dir,
    )

    wall_elapsed = time.perf_counter() - wall_start
    print("=" * 68)
    print("  BENCHMARK COMPLETE")
    print(f"  Total time       : {wall_elapsed:.1f}s")
    print(f"  Real hash configs: {len(hash_results)}")
    print(f"  GPU synthetic    : {len(gpu_synthetic_results)}")
    print(f"  Merkle configs   : {len(merkle_results)}")
    print(f"  PoW configs      : {len(pow_results)}")
    print(f"  Artifacts        : {export_dir}/")
    print("=" * 68 + "\n")

    return {
        "hardware": hw,
        "hash_results": hash_results,
        "gpu_synthetic_results": gpu_synthetic_results,
        "merkle_results": merkle_results,
        "pow_results": pow_results,
        "export_paths": export_paths,
        "elapsed_sec": round(wall_elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prime ChainSentinel — Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --quick                         # smoke test, CPU
  python main.py --mode full                     # full CPU run
  python main.py --gpu --quick                   # MI300X smoke test
  python main.py --gpu --mode full --out exports_mi300x
  python main.py --quick --require-blake3        # fail if BLAKE3 missing
        """,
    )
    parser.add_argument("--gpu", action="store_true", help="Enable GPU synthetic workload benchmark via PyTorch/ROCm/CUDA")
    parser.add_argument("--quick", action="store_true", help="Alias for --mode smoke")
    parser.add_argument("--mode", choices=["smoke", "full"], default=None, help="Benchmark mode")
    parser.add_argument("--out", type=str, default=str(EXPORT_DIR), help=f"Output directory (default: {EXPORT_DIR})")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-result verbose output")
    parser.add_argument("--require-blake3", action="store_true", help="Fail instead of falling back to BLAKE2b if blake3 is missing")
    parser.add_argument("--max-seconds", type=int, default=MAX_RUNTIME_SECONDS, help="Runtime guard in seconds")
    args = parser.parse_args()

    run(
        use_gpu=args.gpu,
        quick=args.quick,
        mode=args.mode,
        export_dir=args.out,
        verbose=not args.quiet,
        require_blake3=args.require_blake3,
        max_runtime_seconds=args.max_seconds,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
