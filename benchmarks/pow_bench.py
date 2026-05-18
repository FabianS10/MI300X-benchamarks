"""
Prime ChainSentinel — Toy Proof-of-Work Benchmark
Reaper Eagle Technologies | Fabian Sabogal

SAFETY BOUNDARY:
    Purely offline synthetic educational simulation. No blockchain network,
    no mining pool, no peer-to-peer system, no payout-generating workload.

Purpose:
    Measure how many Bitcoin-style double-SHA-256 operations per second a CPU
    Python nonce-search workload can sustain, and compare expected-trial and
    actual-trial prediction models.
"""

from __future__ import annotations

import hashlib
import random
import statistics
import struct
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

try:
    from .config import (
        POW_DIFFICULTY_BITS,
        POW_HEADER_SIZE,
        POW_MAX_DIFFICULTY_BITS,
        POW_SAFETY_MULTIPLIER,
        RANDOM_SEED,
        RUNS_PER_CONFIG,
        WARMUP_RUNS,
    )
except ImportError:  # pragma: no cover
    from config import (
        POW_DIFFICULTY_BITS,
        POW_HEADER_SIZE,
        POW_MAX_DIFFICULTY_BITS,
        POW_SAFETY_MULTIPLIER,
        RANDOM_SEED,
        RUNS_PER_CONFIG,
        WARMUP_RUNS,
    )


@dataclass
class PoWBenchResult:
    difficulty_bits: int
    device: str
    header_size_bytes: int
    warmup_runs: int
    measurement_runs: int

    # Theoretical model
    expected_trials: int = 0
    predicted_ms_expected_trials: Optional[float] = None
    predicted_ms_actual_trials: Optional[float] = None
    predicted_ms: Optional[float] = None  # backward-compatible alias = expected-trial model

    # Measurements
    actual_trials_list: list[int] = field(default_factory=list)
    solved_list: list[bool] = field(default_factory=list)
    times_ms: list[float] = field(default_factory=list)

    mean_trials: float = 0.0
    median_trials: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    solved_rate: float = 0.0

    # Derived throughput
    mean_hash_rate_ops_sec: float = 0.0
    prediction_error_expected_pct: Optional[float] = None
    prediction_error_actual_pct: Optional[float] = None
    prediction_error_pct: Optional[float] = None  # backward-compatible alias = expected error

    notes: str = "Offline synthetic toy PoW; not real mining."

    def finalize(self, baseline_throughput_ops_sec: Optional[float] = None) -> "PoWBenchResult":
        if not self.times_ms:
            return self
        self.mean_ms = statistics.mean(self.times_ms)
        self.median_ms = statistics.median(self.times_ms)
        self.std_ms = statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0
        self.min_ms = min(self.times_ms)
        self.max_ms = max(self.times_ms)
        self.mean_trials = statistics.mean(self.actual_trials_list) if self.actual_trials_list else 0.0
        self.median_trials = statistics.median(self.actual_trials_list) if self.actual_trials_list else 0.0
        self.solved_rate = sum(1 for x in self.solved_list if x) / len(self.solved_list) if self.solved_list else 0.0
        if self.mean_ms > 0:
            self.mean_hash_rate_ops_sec = self.mean_trials / (self.mean_ms / 1000.0)

        if baseline_throughput_ops_sec and baseline_throughput_ops_sec > 0:
            self.predicted_ms_expected_trials = (self.expected_trials / baseline_throughput_ops_sec) * 1000.0
            self.predicted_ms_actual_trials = (self.mean_trials / baseline_throughput_ops_sec) * 1000.0
            self.predicted_ms = self.predicted_ms_expected_trials
            if self.mean_ms > 0:
                self.prediction_error_expected_pct = abs(self.predicted_ms_expected_trials - self.mean_ms) / self.mean_ms * 100.0
                self.prediction_error_actual_pct = abs(self.predicted_ms_actual_trials - self.mean_ms) / self.mean_ms * 100.0
                self.prediction_error_pct = self.prediction_error_expected_pct
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def _make_synthetic_header(seed: Optional[int] = None) -> bytearray:
    rng = random.Random((seed or 0) + RANDOM_SEED)
    header = bytearray(POW_HEADER_SIZE)
    struct.pack_into(">I", header, 0, 1)
    header[4:36] = bytes(rng.getrandbits(8) for _ in range(32))
    header[36:68] = bytes(rng.getrandbits(8) for _ in range(32))
    struct.pack_into(">I", header, 68, 1_700_000_000 + rng.randint(0, 1_000_000))
    struct.pack_into(">I", header, 72, 0x1D00FFFF)
    struct.pack_into(">I", header, 76, 0)
    return header


def _set_nonce(header: bytearray, nonce: int) -> None:
    struct.pack_into(">I", header, 76, nonce & 0xFFFFFFFF)


def _double_sha256(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def _satisfies_difficulty(hash_bytes: bytes, difficulty_bits: int) -> bool:
    hash_int = int.from_bytes(hash_bytes, "big")
    target = 1 << (256 - difficulty_bits)
    return hash_int < target


def _solve_pow(difficulty_bits: int, seed: Optional[int] = None) -> tuple[int, float, bool]:
    if difficulty_bits > POW_MAX_DIFFICULTY_BITS:
        raise ValueError(f"difficulty_bits={difficulty_bits} too high for Python toy PoW. Max={POW_MAX_DIFFICULTY_BITS}")

    rng = random.Random((seed or 0) + RANDOM_SEED * 17)
    header = _make_synthetic_header(seed)
    expected_trials = 1 << difficulty_bits
    max_trials = expected_trials * POW_SAFETY_MULTIPLIER
    nonce = rng.randint(0, 0xFFFF)

    t0 = time.perf_counter()
    trials = 0
    solved = False
    while trials < max_trials:
        _set_nonce(header, nonce)
        digest = _double_sha256(bytes(header))
        trials += 1
        if _satisfies_difficulty(digest, difficulty_bits):
            solved = True
            break
        nonce = (nonce + 1) & 0xFFFFFFFF

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return trials, elapsed_ms, solved


def run_pow_benchmarks(
    difficulty_bits: list[int] = POW_DIFFICULTY_BITS,
    runs: int = RUNS_PER_CONFIG,
    warmup: int = WARMUP_RUNS,
    hash_baselines: Optional[dict] = None,
    verbose: bool = True,
) -> list[PoWBenchResult]:
    """Benchmark toy proof-of-work across difficulty levels."""
    results: list[PoWBenchResult] = []
    baseline = hash_baselines.get("sha256") if hash_baselines else None

    print("  ⚠  SAFETY NOTE: Toy PoW — fully offline synthetic simulation.")
    print("     No network connections. No real mining.\n")

    for i, d_bits in enumerate(difficulty_bits):
        expected_trials = 1 << d_bits
        predicted_ms = (expected_trials / baseline) * 1000.0 if baseline and baseline > 0 else None

        if verbose:
            pred_str = f"  predicted(E)={predicted_ms:.1f}ms" if predicted_ms else ""
            print(
                f"  [{i+1}/{len(difficulty_bits)}] PoW difficulty={d_bits:>2} bits  "
                f"E[trials]={expected_trials:>8,}{pred_str} ",
                end="", flush=True,
            )

        result = PoWBenchResult(
            difficulty_bits=d_bits,
            device="cpu",
            header_size_bytes=POW_HEADER_SIZE,
            warmup_runs=warmup,
            measurement_runs=runs,
            expected_trials=expected_trials,
            predicted_ms_expected_trials=predicted_ms,
            predicted_ms=predicted_ms,
        )

        for w in range(warmup):
            _solve_pow(d_bits, seed=w)
        for r in range(runs):
            trials, elapsed_ms, solved = _solve_pow(d_bits, seed=r + 100)
            result.actual_trials_list.append(trials)
            result.solved_list.append(solved)
            result.times_ms.append(elapsed_ms)

        result.finalize(baseline_throughput_ops_sec=baseline)

        if verbose:
            err = f"  err(E)={result.prediction_error_expected_pct:.1f}%" if result.prediction_error_expected_pct is not None else ""
            print(
                f"mean={result.mean_ms:8.2f}ms  "
                f"actual_trials={result.mean_trials:>10,.0f}  "
                f"hash_rate={result.mean_hash_rate_ops_sec:>12,.0f} h/s  "
                f"solved={result.solved_rate:.0%}"
                f"{err}"
            )
        results.append(result)

    return results


if __name__ == "__main__":
    print("\nPrime ChainSentinel — Toy PoW Benchmark (standalone smoke test)\n")
    results = run_pow_benchmarks(difficulty_bits=[8, 12], runs=2, warmup=1, verbose=True)
    print(f"\n✓ {len(results)} PoW configurations completed.")
