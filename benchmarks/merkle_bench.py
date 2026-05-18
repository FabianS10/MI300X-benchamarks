"""
Prime ChainSentinel — Merkle Tree Benchmark
Reaper Eagle Technologies | Fabian Sabogal

Benchmarks binary Merkle tree construction across transaction counts.
The module exports both naive and corrected prediction models:

Naive model:
    t_naive = (N - 1) / T_h

Corrected operation-count model:
    t_corrected = (2N - 1) / T_h

Why both? The naive model counts only internal Merkle nodes. The measured
implementation hashes leaves and internal nodes. The gap between naive,
corrected, and measured timing is a paper finding: raw hash throughput alone
is not sufficient to predict end-to-end Merkle construction time.
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

try:
    from .config import MERKLE_SIZES, RANDOM_SEED, RUNS_PER_CONFIG, TRANSACTION_BYTE_SIZE, WARMUP_RUNS
except ImportError:  # pragma: no cover
    from config import MERKLE_SIZES, RANDOM_SEED, RUNS_PER_CONFIG, TRANSACTION_BYTE_SIZE, WARMUP_RUNS


@dataclass
class MerkleBenchResult:
    leaf_count: int
    algorithm: str
    device: str
    transaction_byte_size: int
    warmup_runs: int
    measurement_runs: int

    # Tree structure
    internal_nodes: int = 0        # N - 1 for power-of-two N
    leaf_hashes: int = 0           # N
    total_hash_ops: int = 0        # N + (N - 1) = 2N - 1
    tree_depth: int = 0            # log2(N)

    # Scientific honesty flags
    backend: str = ""
    effective_algorithm: Optional[str] = None
    notes: str = ""

    # Timing
    times_ms: list[float] = field(default_factory=list)
    mean_ms: float = 0.0
    median_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    # Measured throughput
    internal_hash_ops_per_sec: float = 0.0
    total_hash_ops_per_sec: float = 0.0

    # Prediction models
    naive_predicted_ms: Optional[float] = None
    corrected_predicted_ms: Optional[float] = None
    naive_error_pct: Optional[float] = None
    corrected_error_pct: Optional[float] = None
    residual_overhead_factor: Optional[float] = None

    # Backward-compatible aliases used by older exporter/UI
    predicted_ms: Optional[float] = None
    prediction_error_pct: Optional[float] = None
    hash_ops_per_sec: float = 0.0

    memory_mb: Optional[float] = None

    def finalize(self) -> "MerkleBenchResult":
        if not self.times_ms:
            return self
        self.mean_ms = statistics.mean(self.times_ms)
        self.median_ms = statistics.median(self.times_ms)
        self.std_ms = statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0
        self.min_ms = min(self.times_ms)
        self.max_ms = max(self.times_ms)
        if self.mean_ms > 0:
            elapsed_sec = self.mean_ms / 1000.0
            self.internal_hash_ops_per_sec = self.internal_nodes / elapsed_sec
            self.total_hash_ops_per_sec = self.total_hash_ops / elapsed_sec
            self.hash_ops_per_sec = self.total_hash_ops_per_sec
        return self

    def fill_prediction(self, baseline_throughput_ops_sec: float) -> None:
        """
        Fill naive and corrected predictions from a real CPU hash baseline.
        This should not be fed with GPU synthetic baselines.
        """
        if baseline_throughput_ops_sec <= 0:
            return
        self.naive_predicted_ms = (self.internal_nodes / baseline_throughput_ops_sec) * 1000.0
        self.corrected_predicted_ms = (self.total_hash_ops / baseline_throughput_ops_sec) * 1000.0
        # Backward-compatible fields now point to corrected model.
        self.predicted_ms = self.corrected_predicted_ms
        if self.mean_ms > 0:
            self.naive_error_pct = abs(self.naive_predicted_ms - self.mean_ms) / self.mean_ms * 100.0
            self.corrected_error_pct = abs(self.corrected_predicted_ms - self.mean_ms) / self.mean_ms * 100.0
            self.prediction_error_pct = self.corrected_error_pct
            if self.corrected_predicted_ms and self.corrected_predicted_ms > 0:
                self.residual_overhead_factor = self.mean_ms / self.corrected_predicted_ms

    def to_dict(self) -> dict:
        return asdict(self)


def _generate_leaves(leaf_count: int, byte_size: int, seed: int = RANDOM_SEED) -> list[bytes]:
    rng = random.Random(seed + leaf_count * 2_000_003 + byte_size)
    return [rng.randbytes(byte_size) for _ in range(leaf_count)]


def _pad_to_power_of_two(leaves: list[bytes]) -> list[bytes]:
    """Pad leaves to next power of two by duplicating the last leaf."""
    if not leaves:
        raise ValueError("Merkle tree requires at least one leaf")
    n = len(leaves)
    target = 1 << math.ceil(math.log2(n)) if n > 1 else 1
    padded = list(leaves)
    while len(padded) < target:
        padded.append(padded[-1])
    return padded


def _blake3_hasher(require_blake3: bool = False) -> tuple[Callable[[bytes], bytes], str, str, str]:
    try:
        import blake3 as b3
        return lambda x: b3.blake3(x).digest(), "blake3", "blake3", "Real BLAKE3 Merkle construction."
    except ImportError as e:
        if require_blake3:
            raise ImportError("BLAKE3 package is required. Install with: pip install blake3") from e
        return (
            lambda x: hashlib.blake2b(x, digest_size=32).digest(),
            "blake2b_fallback",
            "blake2b",
            "BLAKE3 package missing; measured BLAKE2b fallback. Do not cite as BLAKE3.",
        )


def _get_hasher(algorithm: str, require_blake3: bool = False) -> tuple[Callable[[bytes], bytes], str, str, str]:
    if algorithm == "sha256":
        return lambda x: hashlib.sha256(x).digest(), "hashlib", "sha256", "Real SHA-256 Merkle construction."
    if algorithm == "sha3_256":
        return lambda x: hashlib.sha3_256(x).digest(), "hashlib", "sha3_256", "Real SHA3-256 Merkle construction."
    if algorithm == "blake3":
        return _blake3_hasher(require_blake3=require_blake3)
    raise ValueError(f"Unsupported Merkle hash algorithm: {algorithm}")


def _build_merkle(leaves: list[bytes], hasher: Callable[[bytes], bytes]) -> tuple[bytes, float]:
    """Build a binary Merkle tree and return (root_hash, elapsed_ms)."""
    t0 = time.perf_counter()
    current_level = [hasher(leaf) for leaf in leaves]  # leaf hashing included
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            right = current_level[i + 1] if i + 1 < len(current_level) else left
            next_level.append(hasher(left + right))
        current_level = next_level
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return current_level[0], elapsed_ms


def run_merkle_benchmarks(
    leaf_counts: list[int] = MERKLE_SIZES,
    algorithms: list[str] = ("sha256", "sha3_256", "blake3"),
    runs: int = RUNS_PER_CONFIG,
    warmup: int = WARMUP_RUNS,
    hash_baselines: Optional[dict] = None,
    verbose: bool = True,
    require_blake3: bool = False,
) -> list[MerkleBenchResult]:
    """Benchmark Merkle tree construction for all (leaf_count, algorithm) combos."""
    results: list[MerkleBenchResult] = []
    total = len(algorithms) * len(leaf_counts)
    done = 0

    for algorithm in algorithms:
        hasher, backend, effective_algorithm, notes = _get_hasher(algorithm, require_blake3=require_blake3)
        for leaf_count in leaf_counts:
            done += 1
            leaves = _pad_to_power_of_two(_generate_leaves(leaf_count, TRANSACTION_BYTE_SIZE))
            actual_count = len(leaves)
            internal_nodes = actual_count - 1
            leaf_hashes = actual_count
            total_hash_ops = leaf_hashes + internal_nodes
            tree_depth = int(math.log2(actual_count))

            if verbose:
                print(
                    f"  [{done}/{total}] Merkle/{algorithm.upper():12s} | "
                    f"leaves={actual_count:>7,}  depth={tree_depth:>2} ",
                    end="", flush=True,
                )

            result = MerkleBenchResult(
                leaf_count=actual_count,
                algorithm=algorithm,
                device="cpu",
                transaction_byte_size=TRANSACTION_BYTE_SIZE,
                warmup_runs=warmup,
                measurement_runs=runs,
                internal_nodes=internal_nodes,
                leaf_hashes=leaf_hashes,
                total_hash_ops=total_hash_ops,
                tree_depth=tree_depth,
                backend=backend,
                effective_algorithm=effective_algorithm,
                notes=notes,
            )

            for _ in range(warmup):
                _build_merkle(leaves, hasher)
            for _ in range(runs):
                _, elapsed_ms = _build_merkle(leaves, hasher)
                result.times_ms.append(elapsed_ms)
            result.finalize()

            if hash_baselines and algorithm in hash_baselines:
                result.fill_prediction(hash_baselines[algorithm])

            if verbose:
                pred = ""
                if result.corrected_predicted_ms is not None:
                    pred = (
                        f"  naive_err={result.naive_error_pct:.1f}%"
                        f"  corr_err={result.corrected_error_pct:.1f}%"
                        f"  overhead={result.residual_overhead_factor:.2f}x"
                    )
                print(
                    f"mean={result.mean_ms:8.2f}ms  "
                    f"ops/s(total)={result.total_hash_ops_per_sec:>12,.0f}"
                    f"{pred}"
                )
            results.append(result)

    return results


if __name__ == "__main__":
    print("\nPrime ChainSentinel — Merkle Benchmark (standalone smoke test)\n")
    results = run_merkle_benchmarks(
        leaf_counts=[128, 512, 1_024],
        algorithms=["sha256"],
        runs=2,
        warmup=1,
        verbose=True,
    )
    print(f"\n✓ {len(results)} Merkle configurations completed.")
