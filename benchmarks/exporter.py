"""
Prime ChainSentinel — Benchmark Exporter
Reaper Eagle Technologies | Fabian Sabogal

Writes benchmark results to full JSON plus CSV artifacts. The export format
separates measured CPU hash results, optional GPU synthetic workload baselines,
Merkle predictions, and Toy PoW predictions.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .config import (
        AUTHOR,
        EXPORT_DIR,
        FULL_JSON,
        GPU_SYNTHETIC_CSV,
        HASH_CSV,
        HARDWARE_JSON,
        MERKLE_CSV,
        POW_CSV,
        PROJECT_NAME,
        SUMMARY_JSON,
        VERSION,
    )
except ImportError:  # pragma: no cover
    from config import (
        AUTHOR,
        EXPORT_DIR,
        FULL_JSON,
        GPU_SYNTHETIC_CSV,
        HASH_CSV,
        HARDWARE_JSON,
        MERKLE_CSV,
        POW_CSV,
        PROJECT_NAME,
        SUMMARY_JSON,
        VERSION,
    )


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_val(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    if isinstance(v, list):
        return [_safe_val(i) for i in v]
    if isinstance(v, dict):
        return {str(k): _safe_val(val) for k, val in v.items()}
    return str(v)


def _write_csv(rows: list[dict], filepath: Path) -> None:
    if not rows:
        return
    # union keys so older/newer dataclasses do not break exports
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_hardware_json(hardware_profile: dict, export_dir: str | Path = EXPORT_DIR, filename: str = HARDWARE_JSON) -> str:
    out_dir = _ensure_dir(export_dir)
    filepath = out_dir / filename
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(_safe_val(hardware_profile), f, indent=2, ensure_ascii=False)
    print(f"  ✓ Hardware JSON → {filepath}")
    return str(filepath)


def export_json(
    hardware_profile: dict,
    hash_results: list,
    merkle_results: list,
    pow_results: list,
    gpu_synthetic_results: list | None = None,
    export_dir: str | Path = EXPORT_DIR,
    filename: str = FULL_JSON,
) -> str:
    out_dir = _ensure_dir(export_dir)
    filepath = out_dir / filename
    gpu_synthetic_results = gpu_synthetic_results or []
    payload = {
        "meta": {
            "project": PROJECT_NAME,
            "version": VERSION,
            "author": AUTHOR,
            "exported_at": _ts(),
            "paper_note": (
                "CPU hash benchmarks measure real cryptographic hash throughput. "
                "GPU synthetic results, if present, are PyTorch byte-workload baselines, "
                "not native GPU SHA-256/SHA3/BLAKE3 hash kernels."
            ),
        },
        "hardware": _safe_val(hardware_profile),
        "hash_benchmarks": [_safe_val(r.to_dict()) for r in hash_results],
        "gpu_synthetic_benchmarks": [_safe_val(r.to_dict()) for r in gpu_synthetic_results],
        "merkle_benchmarks": [_safe_val(r.to_dict()) for r in merkle_results],
        "pow_benchmarks": [_safe_val(r.to_dict()) for r in pow_results],
        "summary": _build_summary(hash_results, merkle_results, pow_results, gpu_synthetic_results),
    }
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON exported  → {filepath}")
    return str(filepath)


def export_hash_csv(results: list, export_dir: str | Path = EXPORT_DIR, filename: str = HASH_CSV) -> str:
    if not results:
        return ""
    out_dir = _ensure_dir(export_dir)
    filepath = out_dir / filename
    raw_path = out_dir / ("raw_" + filename)

    rows = []
    for r in results:
        for i, t in enumerate(r.times_ms):
            rows.append({
                "algorithm": r.algorithm,
                "effective_algorithm": getattr(r, "effective_algorithm", None),
                "batch_size": r.batch_size,
                "transaction_byte_size": r.transaction_byte_size,
                "device": r.device,
                "backend": r.backend,
                "is_surrogate": getattr(r, "is_surrogate", False),
                "run_index": i,
                "time_ms": round(t, 6),
                "throughput_ops_sec": round(r.batch_size / (t / 1000.0), 0) if t > 0 else 0,
            })

    summary_rows = []
    for r in results:
        summary_rows.append({
            "algorithm": r.algorithm,
            "effective_algorithm": getattr(r, "effective_algorithm", None),
            "batch_size": r.batch_size,
            "transaction_byte_size": r.transaction_byte_size,
            "device": r.device,
            "backend": r.backend,
            "is_surrogate": getattr(r, "is_surrogate", False),
            "mean_ms": round(r.mean_ms, 6),
            "median_ms": round(getattr(r, "median_ms", 0.0), 6),
            "std_ms": round(r.std_ms, 6),
            "min_ms": round(r.min_ms, 6),
            "max_ms": round(r.max_ms, 6),
            "mean_throughput_ops_sec": round(r.mean_throughput_ops_sec, 0),
            "median_throughput_ops_sec": round(getattr(r, "median_throughput_ops_sec", 0.0), 0),
            "peak_throughput_ops_sec": round(r.peak_throughput_ops_sec, 0),
            "memory_mb": r.memory_mb,
            "notes": getattr(r, "notes", ""),
        })

    _write_csv(rows, raw_path)
    _write_csv(summary_rows, filepath)
    print(f"  ✓ Hash CSV       → {filepath}")
    print(f"  ✓ Hash CSV (raw) → {raw_path}")
    return str(filepath)


def export_gpu_synthetic_csv(results: list, export_dir: str | Path = EXPORT_DIR, filename: str = GPU_SYNTHETIC_CSV) -> str:
    if not results:
        return ""
    return export_hash_csv(results, export_dir=export_dir, filename=filename)


def export_merkle_csv(results: list, export_dir: str | Path = EXPORT_DIR, filename: str = MERKLE_CSV) -> str:
    if not results:
        return ""
    out_dir = _ensure_dir(export_dir)
    filepath = out_dir / filename
    rows = []
    for r in results:
        rows.append({
            "algorithm": r.algorithm,
            "effective_algorithm": getattr(r, "effective_algorithm", None),
            "backend": getattr(r, "backend", ""),
            "leaf_count": r.leaf_count,
            "tree_depth": r.tree_depth,
            "leaf_hashes": getattr(r, "leaf_hashes", None),
            "internal_nodes": r.internal_nodes,
            "total_hash_ops": getattr(r, "total_hash_ops", None),
            "device": r.device,
            "mean_ms": round(r.mean_ms, 6),
            "median_ms": round(getattr(r, "median_ms", 0.0), 6),
            "std_ms": round(r.std_ms, 6),
            "min_ms": round(r.min_ms, 6),
            "max_ms": round(r.max_ms, 6),
            "internal_hash_ops_per_sec": round(getattr(r, "internal_hash_ops_per_sec", 0.0), 0),
            "total_hash_ops_per_sec": round(getattr(r, "total_hash_ops_per_sec", getattr(r, "hash_ops_per_sec", 0.0)), 0),
            "naive_predicted_ms": round(r.naive_predicted_ms, 6) if getattr(r, "naive_predicted_ms", None) is not None else None,
            "corrected_predicted_ms": round(r.corrected_predicted_ms, 6) if getattr(r, "corrected_predicted_ms", None) is not None else None,
            "naive_error_pct": round(r.naive_error_pct, 4) if getattr(r, "naive_error_pct", None) is not None else None,
            "corrected_error_pct": round(r.corrected_error_pct, 4) if getattr(r, "corrected_error_pct", None) is not None else None,
            "residual_overhead_factor": round(r.residual_overhead_factor, 6) if getattr(r, "residual_overhead_factor", None) is not None else None,
            "notes": getattr(r, "notes", ""),
        })
    _write_csv(rows, filepath)
    print(f"  ✓ Merkle CSV     → {filepath}")
    return str(filepath)


def export_pow_csv(results: list, export_dir: str | Path = EXPORT_DIR, filename: str = POW_CSV) -> str:
    if not results:
        return ""
    out_dir = _ensure_dir(export_dir)
    filepath = out_dir / filename
    rows = []
    for r in results:
        rows.append({
            "difficulty_bits": r.difficulty_bits,
            "expected_trials": r.expected_trials,
            "mean_actual_trials": round(r.mean_trials, 0),
            "median_actual_trials": round(getattr(r, "median_trials", 0), 0),
            "solved_rate": round(getattr(r, "solved_rate", 0.0), 4),
            "mean_ms": round(r.mean_ms, 6),
            "median_ms": round(getattr(r, "median_ms", 0.0), 6),
            "std_ms": round(r.std_ms, 6),
            "min_ms": round(r.min_ms, 6),
            "max_ms": round(r.max_ms, 6),
            "mean_hash_rate_ops_sec": round(r.mean_hash_rate_ops_sec, 0),
            "predicted_ms_expected_trials": round(r.predicted_ms_expected_trials, 6) if getattr(r, "predicted_ms_expected_trials", None) is not None else None,
            "predicted_ms_actual_trials": round(r.predicted_ms_actual_trials, 6) if getattr(r, "predicted_ms_actual_trials", None) is not None else None,
            "prediction_error_expected_pct": round(r.prediction_error_expected_pct, 4) if getattr(r, "prediction_error_expected_pct", None) is not None else None,
            "prediction_error_actual_pct": round(r.prediction_error_actual_pct, 4) if getattr(r, "prediction_error_actual_pct", None) is not None else None,
            "notes": getattr(r, "notes", ""),
        })
    _write_csv(rows, filepath)
    print(f"  ✓ PoW CSV        → {filepath}")
    return str(filepath)


def _build_summary(hash_results, merkle_results, pow_results, gpu_synthetic_results=None) -> dict:
    gpu_synthetic_results = gpu_synthetic_results or []
    summary = {"generated_at": _ts(), "hash": {}, "gpu_synthetic": {}, "merkle": {}, "pow": {}, "paper_findings": []}

    for alg in ["sha256", "sha3_256", "blake3"]:
        alg_results = [r for r in hash_results if r.algorithm == alg and not getattr(r, "is_surrogate", False)]
        if alg_results:
            best = max(alg_results, key=lambda r: r.peak_throughput_ops_sec)
            summary["hash"][alg] = {
                "peak_throughput_ops_sec": round(best.peak_throughput_ops_sec, 0),
                "mean_throughput_ops_sec": round(best.mean_throughput_ops_sec, 0),
                "best_batch_size": best.batch_size,
                "device": best.device,
                "backend": best.backend,
                "effective_algorithm": getattr(best, "effective_algorithm", alg),
            }

    if gpu_synthetic_results:
        best = max(gpu_synthetic_results, key=lambda r: r.peak_throughput_ops_sec)
        summary["gpu_synthetic"] = {
            "peak_throughput_ops_sec": round(best.peak_throughput_ops_sec, 0),
            "best_batch_size": best.batch_size,
            "backend": best.backend,
            "note": "Synthetic PyTorch byte workload; not native cryptographic hashing.",
        }

    for alg in ["sha256", "sha3_256", "blake3"]:
        alg_results = [r for r in merkle_results if r.algorithm == alg]
        if alg_results:
            best = max(alg_results, key=lambda r: getattr(r, "total_hash_ops_per_sec", getattr(r, "hash_ops_per_sec", 0.0)))
            valid_corr = [r.corrected_error_pct for r in alg_results if getattr(r, "corrected_error_pct", None) is not None]
            valid_naive = [r.naive_error_pct for r in alg_results if getattr(r, "naive_error_pct", None) is not None]
            summary["merkle"][alg] = {
                "peak_total_hash_ops_sec": round(getattr(best, "total_hash_ops_per_sec", getattr(best, "hash_ops_per_sec", 0.0)), 0),
                "best_leaf_count": best.leaf_count,
                "mean_naive_error_pct": round(sum(valid_naive) / len(valid_naive), 2) if valid_naive else None,
                "mean_corrected_error_pct": round(sum(valid_corr) / len(valid_corr), 2) if valid_corr else None,
            }

    for r in pow_results:
        summary["pow"][f"d{r.difficulty_bits}"] = {
            "mean_hash_rate_ops_sec": round(r.mean_hash_rate_ops_sec, 0),
            "mean_ms": round(r.mean_ms, 6),
            "expected_trials": r.expected_trials,
            "mean_actual_trials": round(r.mean_trials, 0),
            "prediction_error_expected_pct": round(r.prediction_error_expected_pct, 2) if getattr(r, "prediction_error_expected_pct", None) is not None else None,
        }

    summary["paper_findings"].append(
        "Naive Merkle predictions based only on internal nodes can underestimate full construction time because measured implementations hash both leaves and internal nodes."
    )
    summary["paper_findings"].append(
        "Corrected Merkle predictions use total hash operations (2N-1); residual gaps indicate implementation overhead and data movement costs."
    )
    return summary


def export_summary_json(summary: dict, export_dir: str | Path = EXPORT_DIR, filename: str = SUMMARY_JSON) -> str:
    out_dir = _ensure_dir(export_dir)
    filepath = out_dir / filename
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(_safe_val(summary), f, indent=2, ensure_ascii=False)
    print(f"  ✓ Summary JSON  → {filepath}")
    return str(filepath)


def export_all(
    hardware_profile: dict,
    hash_results: list,
    merkle_results: list,
    pow_results: list,
    export_dir: str | Path = EXPORT_DIR,
    gpu_synthetic_results: list | None = None,
) -> dict:
    print("\n" + "─" * 60)
    print("  Exporting benchmark artifacts...")
    print("─" * 60)
    gpu_synthetic_results = gpu_synthetic_results or []
    summary = _build_summary(hash_results, merkle_results, pow_results, gpu_synthetic_results)
    paths = {
        "hardware_json": export_hardware_json(hardware_profile, export_dir),
        "json": export_json(hardware_profile, hash_results, merkle_results, pow_results, gpu_synthetic_results, export_dir),
        "summary_json": export_summary_json(summary, export_dir),
        "hash_csv": export_hash_csv(hash_results, export_dir),
        "gpu_synthetic_csv": export_gpu_synthetic_csv(gpu_synthetic_results, export_dir) if gpu_synthetic_results else "",
        "merkle_csv": export_merkle_csv(merkle_results, export_dir),
        "pow_csv": export_pow_csv(pow_results, export_dir),
    }
    print("─" * 60)
    print(f"  ✓ All artifacts saved to: {export_dir}/")
    print("─" * 60 + "\n")
    return paths
