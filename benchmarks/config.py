"""
Prime ChainSentinel — Benchmark Configuration
Reaper Eagle Technologies | Fabian Sabogal

All tunable parameters live here. Nothing else imports magic numbers.
Designed for safe local debugging and credit-efficient MI300X execution.
"""

from pathlib import Path

# ── Project metadata ──────────────────────────────────────────────────────────
PROJECT_NAME = "Prime ChainSentinel"
VERSION = "0.2.0-benchmark-safe"
AUTHOR = "Fabian Sabogal — Reaper Eagle Technologies"

# ── Run modes ─────────────────────────────────────────────────────────────────
DEFAULT_MODE = "smoke"  # smoke | full
VALID_MODES = ["smoke", "full"]

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 1337

# ── Runtime guard ─────────────────────────────────────────────────────────────
# Prevents runaway scripts from burning AMD credit.
MAX_RUNTIME_SECONDS = 600

# ── Batch sizes for throughput curves ─────────────────────────────────────────
SMOKE_BATCH_SIZES = [1_000, 10_000]
FULL_BATCH_SIZES = [1_000, 10_000, 100_000, 500_000, 1_000_000]
BATCH_SIZES = FULL_BATCH_SIZES  # legacy alias

# ── Algorithms to benchmark ───────────────────────────────────────────────────
HASH_ALGORITHMS = ["sha256", "sha3_256", "blake3"]

# ── Statistical rigor ─────────────────────────────────────────────────────────
SMOKE_RUNS_PER_CONFIG = 2
FULL_RUNS_PER_CONFIG = 5
RUNS_PER_CONFIG = FULL_RUNS_PER_CONFIG  # legacy alias

SMOKE_WARMUP_RUNS = 1
FULL_WARMUP_RUNS = 2
WARMUP_RUNS = FULL_WARMUP_RUNS  # legacy alias

# ── Input data ────────────────────────────────────────────────────────────────
# 256 bytes is a reasonable proxy for a serialized transaction payload.
TRANSACTION_BYTE_SIZE = 256

# ── GPU synthetic workload labeling ───────────────────────────────────────────
# Important: current PyTorch GPU mode is a synthetic byte-parallel workload,
# not native SHA-256/SHA3/BLAKE3 hashing.
GPU_SYNTHETIC_BACKEND_NAME = "torch_rocm_synthetic_byte_workload"
GPU_SYNTHETIC_ALGORITHM = "byte_workload"

# ── Merkle tree ───────────────────────────────────────────────────────────────
SMOKE_MERKLE_SIZES = [128, 512, 1_024]
FULL_MERKLE_SIZES = [128, 512, 1_024, 4_096, 16_384, 65_536]
MERKLE_SIZES = FULL_MERKLE_SIZES  # legacy alias

# ── Toy Proof-of-Work ─────────────────────────────────────────────────────────
# 2^d expected trials. Keep low. This is NOT real mining.
SMOKE_POW_DIFFICULTY_BITS = [8, 12]
FULL_POW_DIFFICULTY_BITS = [8, 12, 16, 20]
POW_DIFFICULTY_BITS = FULL_POW_DIFFICULTY_BITS  # legacy alias
POW_HEADER_SIZE = 80  # bytes, mirrors Bitcoin block header size
POW_MAX_DIFFICULTY_BITS = 24
POW_SAFETY_MULTIPLIER = 16

# ── Monte Carlo simulation placeholders ───────────────────────────────────────
SMOKE_MC_TRIALS = 1_000
FULL_MC_TRIALS = 100_000
SMOKE_MC_HORIZON_YEARS = 10
FULL_MC_HORIZON_YEARS = 30

# ── Export paths ─────────────────────────────────────────────────────────────
EXPORT_DIR = Path("exports")
HASH_CSV = "hash_benchmark.csv"
GPU_SYNTHETIC_CSV = "gpu_synthetic_benchmark.csv"
MERKLE_CSV = "merkle_benchmark.csv"
POW_CSV = "pow_benchmark.csv"
MONTE_CARLO_CSV = "monte_carlo_benchmark.csv"
HARDWARE_JSON = "hardware_profile.json"
FULL_JSON = "benchmark_full.json"
SUMMARY_JSON = "summary.json"


def get_mode_config(mode: str = DEFAULT_MODE) -> dict:
    """Return safe benchmark settings for a given mode."""
    mode = (mode or DEFAULT_MODE).lower().strip()

    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Valid modes: {VALID_MODES}")

    if mode == "smoke":
        return {
            "mode": "smoke",
            "batch_sizes": SMOKE_BATCH_SIZES,
            "runs_per_config": SMOKE_RUNS_PER_CONFIG,
            "warmup_runs": SMOKE_WARMUP_RUNS,
            "merkle_sizes": SMOKE_MERKLE_SIZES,
            "pow_difficulty_bits": SMOKE_POW_DIFFICULTY_BITS,
            "mc_trials": SMOKE_MC_TRIALS,
            "mc_horizon_years": SMOKE_MC_HORIZON_YEARS,
        }

    return {
        "mode": "full",
        "batch_sizes": FULL_BATCH_SIZES,
        "runs_per_config": FULL_RUNS_PER_CONFIG,
        "warmup_runs": FULL_WARMUP_RUNS,
        "merkle_sizes": FULL_MERKLE_SIZES,
        "pow_difficulty_bits": FULL_POW_DIFFICULTY_BITS,
        "mc_trials": FULL_MC_TRIALS,
        "mc_horizon_years": FULL_MC_HORIZON_YEARS,
    }
