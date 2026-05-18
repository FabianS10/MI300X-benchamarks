"""
Prime ChainSentinel — Hash and GPU Synthetic Workload Benchmark
Reaper Eagle Technologies | Fabian Sabogal

CPU mode benchmarks real SHA-256, SHA3-256, and BLAKE3/BLAKE2b fallback.
GPU mode benchmarks a PyTorch/ROCm synthetic byte-parallel workload. The GPU
workload is intentionally labeled as a surrogate, not native cryptographic
hash throughput. Native GPU hash throughput would require HIP/CUDA/OpenCL
kernels and should not be claimed from this Python tensor benchmark.
"""

from __future__ import annotations

import hashlib
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

try:
    from .config import (
        BATCH_SIZES,
        GPU_SYNTHETIC_ALGORITHM,
        GPU_SYNTHETIC_BACKEND_NAME,
        HASH_ALGORITHMS,
        RANDOM_SEED,
        RUNS_PER_CONFIG,
        TRANSACTION_BYTE_SIZE,
        WARMUP_RUNS,
    )
except ImportError:  # pragma: no cover - standalone execution fallback
    from config import (
        BATCH_SIZES,
        GPU_SYNTHETIC_ALGORITHM,
        GPU_SYNTHETIC_BACKEND_NAME,
        HASH_ALGORITHMS,
        RANDOM_SEED,
        RUNS_PER_CONFIG,
        TRANSACTION_BYTE_SIZE,
        WARMUP_RUNS,
    )


@dataclass
class HashBenchResult:
    algorithm: str
    batch_size: int
    transaction_byte_size: int
    device: str                       # cpu or gpu
    backend: str                      # hashlib, blake3, blake2b_fallback, torch_rocm_synthetic...
    warmup_runs: int
    measurement_runs: int

    # Scientific honesty flags
    effective_algorithm: Optional[str] = None
    is_surrogate: bool = False
    notes: str = ""

    # Timing (milliseconds per batch)
    times_ms: list[float] = field(default_factory=list)
    mean_ms: float = 0.0
    median_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    # Throughput
    mean_throughput_ops_sec: float = 0.0
    median_throughput_ops_sec: float = 0.0
    peak_throughput_ops_sec: float = 0.0

    # Memory
    memory_mb: Optional[float] = None

    def finalize(self) -> "HashBenchResult":
        if not self.times_ms:
            return self
        self.mean_ms = statistics.mean(self.times_ms)
        self.median_ms = statistics.median(self.times_ms)
        self.std_ms = statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0
        self.min_ms = min(self.times_ms)
        self.max_ms = max(self.times_ms)
        self.mean_throughput_ops_sec = self.batch_size / (self.mean_ms / 1000.0) if self.mean_ms > 0 else 0.0
        self.median_throughput_ops_sec = self.batch_size / (self.median_ms / 1000.0) if self.median_ms > 0 else 0.0
        self.peak_throughput_ops_sec = self.batch_size / (self.min_ms / 1000.0) if self.min_ms > 0 else 0.0
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def _generate_payloads(batch_size: int, byte_size: int, seed: int = RANDOM_SEED) -> list[bytes]:
    """
    Generate deterministic pseudo-random byte strings simulating transaction
    payloads. Generated outside timing so benchmark measures hashing only.
    """
    rng = random.Random(seed + batch_size * 1_000_003 + byte_size)
    return [rng.randbytes(byte_size) for _ in range(batch_size)]


def _run_sha256_batch(payloads: list[bytes]) -> float:
    t0 = time.perf_counter()
    for p in payloads:
        hashlib.sha256(p).digest()
    return (time.perf_counter() - t0) * 1000.0


def _run_sha3_256_batch(payloads: list[bytes]) -> float:
    t0 = time.perf_counter()
    for p in payloads:
        hashlib.sha3_256(p).digest()
    return (time.perf_counter() - t0) * 1000.0


def _detect_blake3_backend(require_blake3: bool = False) -> tuple[str, str]:
    try:
        import blake3 as _b3  # noqa: F401
        return "blake3", "blake3"
    except ImportError as e:
        if require_blake3:
            raise ImportError("BLAKE3 package is required. Install with: pip install blake3") from e
        return "blake2b_fallback", "blake2b"


def _run_blake3_batch(payloads: list[bytes], require_blake3: bool = False) -> float:
    backend, _effective_algorithm = _detect_blake3_backend(require_blake3=require_blake3)
    if backend == "blake3":
        import blake3 as b3
        t0 = time.perf_counter()
        for p in payloads:
            b3.blake3(p).digest()
        return (time.perf_counter() - t0) * 1000.0

    # Fallback is deliberately allowed for robustness, but exported metadata
    # marks it as BLAKE2b fallback. Do not cite as BLAKE3 in the paper.
    t0 = time.perf_counter()
    for p in payloads:
        hashlib.blake2b(p, digest_size=32).digest()
    return (time.perf_counter() - t0) * 1000.0


CPU_RUNNERS = {
    "sha256": _run_sha256_batch,
    "sha3_256": _run_sha3_256_batch,
    "blake3": _run_blake3_batch,
}


def run_hash_benchmarks(
    algorithms: list[str] = HASH_ALGORITHMS,
    batch_sizes: list[int] = BATCH_SIZES,
    runs: int = RUNS_PER_CONFIG,
    warmup: int = WARMUP_RUNS,
    use_gpu: bool = False,
    verbose: bool = True,
    require_blake3: bool = False,
) -> list[HashBenchResult]:
    """
    Run real CPU hash benchmarks for all (algorithm, batch_size) combinations.

    Important: This function intentionally ignores use_gpu for cryptographic
    hashes because SHA-256/SHA3/BLAKE3 are not implemented as native GPU kernels
    here. Use run_gpu_synthetic_benchmarks() for MI300X PyTorch workload baseline.
    """
    if use_gpu and verbose:
        print(
            "  ⚠  use_gpu=True was passed to real hash benchmark, but native GPU "
            "hash kernels are not implemented. Running real CPU hash benchmarks. "
            "Use run_gpu_synthetic_benchmarks for MI300X simulation baseline.\n"
        )

    results: list[HashBenchResult] = []
    total = len(algorithms) * len(batch_sizes)
    done = 0

    for algorithm in algorithms:
        if algorithm not in CPU_RUNNERS:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        for batch_size in batch_sizes:
            done += 1
            if verbose:
                print(f"  [{done}/{total}] {algorithm.upper():12s} | batch={batch_size:>9,} ", end="", flush=True)

            backend = "hashlib"
            effective_algorithm = algorithm
            notes = "Real CPU cryptographic hash throughput."
            if algorithm == "blake3":
                backend, effective_algorithm = _detect_blake3_backend(require_blake3=require_blake3)
                if backend == "blake2b_fallback":
                    notes = "BLAKE3 package missing; measured BLAKE2b fallback. Do not cite as BLAKE3."

            result = HashBenchResult(
                algorithm=algorithm,
                effective_algorithm=effective_algorithm,
                batch_size=batch_size,
                transaction_byte_size=TRANSACTION_BYTE_SIZE,
                device="cpu",
                backend=backend,
                warmup_runs=warmup,
                measurement_runs=runs,
                is_surrogate=False,
                notes=notes,
            )

            payloads = _generate_payloads(batch_size, TRANSACTION_BYTE_SIZE)

            for _ in range(warmup):
                if algorithm == "blake3":
                    _run_blake3_batch(payloads, require_blake3=require_blake3)
                else:
                    CPU_RUNNERS[algorithm](payloads)

            for _ in range(runs):
                if algorithm == "blake3":
                    elapsed_ms = _run_blake3_batch(payloads, require_blake3=require_blake3)
                else:
                    elapsed_ms = CPU_RUNNERS[algorithm](payloads)
                result.times_ms.append(elapsed_ms)

            result.finalize()

            if verbose:
                print(
                    f"mean={result.mean_ms:8.2f}ms  "
                    f"median={result.median_ms:8.2f}ms  "
                    f"throughput={result.mean_throughput_ops_sec:>15,.0f} ops/s  "
                    f"backend={result.backend}"
                )

            results.append(result)

    return results


def _make_gpu_tensor(batch_size: int, byte_size: int, seed: int = RANDOM_SEED):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate on CPU deterministically, upload once per configuration.
    payloads = _generate_payloads(batch_size, byte_size, seed=seed)
    flat = bytearray().join(payloads)
    data_cpu = torch.frombuffer(bytes(flat), dtype=torch.uint8).reshape(batch_size, byte_size)
    return data_cpu.to(device, non_blocking=False), device


def _run_gpu_synthetic_once(data, device) -> tuple[float, float]:
    import torch

    if device.type != "cpu":
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        # Synthetic byte-parallel workload: memory bandwidth + arithmetic.
        _ = (data.float() * 0.003921568).pow(2).sum(dim=1)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        mem_after = torch.cuda.memory_allocated(device)
        memory_mb = max(0.0, (mem_after - mem_before) / (1024 ** 2))
        return float(elapsed_ms), float(memory_mb)

    t0 = time.perf_counter()
    _ = (data.float() * 0.003921568).pow(2).sum(dim=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms, 0.0


def run_gpu_synthetic_benchmarks(
    batch_sizes: list[int] = BATCH_SIZES,
    runs: int = RUNS_PER_CONFIG,
    warmup: int = WARMUP_RUNS,
    verbose: bool = True,
) -> list[HashBenchResult]:
    """
    Run a PyTorch GPU synthetic byte workload benchmark.

    This produces MI300X simulation acceleration baselines. It does NOT measure
    SHA-256/SHA3/BLAKE3 native GPU hash throughput.
    """
    import torch

    if not torch.cuda.is_available():
        if verbose:
            print("  ⚠  No GPU detected by PyTorch; skipping GPU synthetic benchmark.")
        return []

    results: list[HashBenchResult] = []
    if verbose:
        print("  ⚠  GPU synthetic mode: measures byte-parallel PyTorch workload, not native hashes.")

    for idx, batch_size in enumerate(batch_sizes, start=1):
        if verbose:
            print(f"  [{idx}/{len(batch_sizes)}] GPU/{GPU_SYNTHETIC_ALGORITHM:12s} | batch={batch_size:>9,} ", end="", flush=True)

        data, device = _make_gpu_tensor(batch_size, TRANSACTION_BYTE_SIZE)
        result = HashBenchResult(
            algorithm=GPU_SYNTHETIC_ALGORITHM,
            effective_algorithm="synthetic_byte_reduction",
            batch_size=batch_size,
            transaction_byte_size=TRANSACTION_BYTE_SIZE,
            device="gpu",
            backend=GPU_SYNTHETIC_BACKEND_NAME,
            warmup_runs=warmup,
            measurement_runs=runs,
            is_surrogate=True,
            notes="Synthetic PyTorch byte workload for MI300X simulation baseline; not native cryptographic hashing.",
        )

        for _ in range(warmup):
            _run_gpu_synthetic_once(data, device)

        memory_mb = None
        for _ in range(runs):
            elapsed_ms, mem = _run_gpu_synthetic_once(data, device)
            result.times_ms.append(elapsed_ms)
            if memory_mb is None:
                memory_mb = mem
        result.memory_mb = memory_mb
        result.finalize()

        if verbose:
            print(
                f"mean={result.mean_ms:8.2f}ms  "
                f"median={result.median_ms:8.2f}ms  "
                f"throughput={result.mean_throughput_ops_sec:>15,.0f} ops/s"
            )
        results.append(result)

    return results


if __name__ == "__main__":
    print("\nPrime ChainSentinel — Hash Benchmark (standalone smoke test)\n")
    out = run_hash_benchmarks(
        algorithms=["sha256", "sha3_256", "blake3"],
        batch_sizes=[1_000, 10_000],
        runs=2,
        warmup=1,
        use_gpu=False,
        verbose=True,
    )
    print(f"\n✓ {len(out)} real CPU hash benchmark configurations completed.")
