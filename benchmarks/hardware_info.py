"""
Prime ChainSentinel — Hardware Info Collector
Reaper Eagle Technologies | Fabian Sabogal

Detects CPU, RAM, and GPU (CUDA or ROCm).
Returns a structured dict that becomes the hardware_profile in the database.
The same script runs on a laptop and on the MI300X — no code changes needed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(cmd: list[str], timeout: int = 10) -> str:
    """Run a shell command and return stdout. Empty string on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return (result.stdout or "").strip()
    except Exception:
        return ""


def _cpu_name() -> str:
    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass

    elif system == "Darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name

    elif system == "Windows":
        raw = _run(["wmic", "cpu", "get", "name"])
        lines = [x.strip() for x in raw.splitlines() if x.strip() and x.strip().lower() != "name"]
        if lines:
            return lines[0]

    return platform.processor() or "Unknown CPU"


def _cpu_cores() -> int:
    try:
        return os.cpu_count() or 0
    except Exception:
        return 0


def _ram_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        try:
            with open("/proc/meminfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 1)
        except Exception:
            pass
    return 0.0


def _parse_rocm_vram_gb(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value) / (1024 ** 3), 1)

    text = str(value).strip().replace(",", "")
    try:
        return round(int(text) / (1024 ** 3), 1)
    except Exception:
        pass

    parts = text.split()
    if not parts:
        return None
    try:
        number = float(parts[0])
    except Exception:
        return None
    unit = parts[1].lower() if len(parts) > 1 else ""
    if unit.startswith("gb") or unit.startswith("gib"):
        return round(number, 1)
    if unit.startswith("mb") or unit.startswith("mib"):
        return round(number / 1024.0, 1)
    return None


def _detect_rocm_gpu() -> dict[str, Any] | None:
    raw = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if not raw:
        return None

    gpu_info = {
        "gpu_name": "AMD GPU",
        "vram_gb": None,
        "backend": "rocm",
        "device_count": 0,
        "raw_info": raw,
    }

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        gpu_info["gpu_name"] = "AMD Instinct (ROCm detected)"
        gpu_info["device_count"] = 1
        return gpu_info

    names: list[str] = []
    vrams: list[float] = []
    for _, val in data.items():
        if not isinstance(val, dict):
            continue
        name = (
            val.get("Card series")
            or val.get("Card Series")
            or val.get("GPU")
            or val.get("Product Name")
            or val.get("Unique ID")
            or "AMD GPU"
        )
        vram_value = (
            val.get("VRAM Total Memory (B)")
            or val.get("VRAM Total Memory")
            or val.get("GPU Memory Total")
            or val.get("Total VRAM")
        )
        names.append(str(name))
        parsed_vram = _parse_rocm_vram_gb(vram_value)
        if parsed_vram:
            vrams.append(parsed_vram)

    if names:
        gpu_info["gpu_name"] = names[0]
        gpu_info["device_count"] = len(names)
    if vrams:
        gpu_info["vram_gb"] = vrams[0]
    return gpu_info


def _detect_torch_gpu() -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        device_count = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        backend = "rocm_pytorch" if getattr(torch.version, "hip", None) else "cuda_pytorch"
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_gb": round(props.total_memory / (1024 ** 3), 1),
            "backend": backend,
            "device_count": device_count,
            "raw_info": None,
        }
    except Exception:
        return None


def _detect_nvidia_gpu() -> dict[str, Any] | None:
    raw = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if not raw:
        return None
    lines = [x.strip() for x in raw.splitlines() if x.strip()]
    if not lines:
        return None
    parts = lines[0].split(",")
    vram_gb = None
    if len(parts) > 1:
        try:
            vram_gb = round(int(parts[1].strip().split()[0]) / 1024, 1)
        except Exception:
            pass
    return {
        "gpu_name": parts[0].strip(),
        "vram_gb": vram_gb,
        "backend": "cuda",
        "device_count": len(lines),
        "raw_info": raw,
    }


def _detect_gpu() -> dict[str, Any]:
    default = {
        "gpu_name": None,
        "vram_gb": None,
        "backend": "cpu_only",
        "device_count": 0,
        "raw_info": None,
    }
    for detector in (_detect_rocm_gpu, _detect_torch_gpu, _detect_nvidia_gpu):
        info = detector()
        if info:
            return info
    return default


def _git_commit() -> str | None:
    commit = _run(["git", "rev-parse", "--short", "HEAD"], timeout=5)
    return commit or None


def collect() -> dict[str, Any]:
    """Collect full hardware profile ready for database insertion/export."""
    gpu = _detect_gpu()
    profile = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_name": _cpu_name(),
        "cpu_cores": _cpu_cores(),
        "ram_gb": _ram_gb(),
        "gpu_name": gpu["gpu_name"],
        "vram_gb": gpu["vram_gb"],
        "gpu_backend": gpu["backend"],
        "gpu_device_count": gpu["device_count"],
        "gpu_raw_info": gpu["raw_info"],
        "device_type": "GPU" if gpu["backend"] != "cpu_only" else "CPU",
        "display_name": gpu["gpu_name"] if gpu["gpu_name"] else _cpu_name(),
        "git_commit": _git_commit(),
        "env": {
            "HIP_VISIBLE_DEVICES": os.getenv("HIP_VISIBLE_DEVICES"),
            "ROCR_VISIBLE_DEVICES": os.getenv("ROCR_VISIBLE_DEVICES"),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
            "HSA_OVERRIDE_GFX_VERSION": os.getenv("HSA_OVERRIDE_GFX_VERSION"),
        },
    }

    rocm_ver = _run(["rocm-smi", "--version"])
    profile["rocm_version"] = rocm_ver if rocm_ver else None

    try:
        import torch
        profile["pytorch_version"] = torch.__version__
        profile["torch_cuda_available"] = bool(torch.cuda.is_available())
        profile["torch_hip_version"] = getattr(torch.version, "hip", None)
        profile["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    except ImportError:
        profile["pytorch_version"] = None
        profile["torch_cuda_available"] = False
        profile["torch_hip_version"] = None
        profile["torch_cuda_version"] = None

    return profile


def print_summary(profile: dict[str, Any]) -> None:
    print("\n" + "=" * 68)
    print("  Prime ChainSentinel — Hardware Profile")
    print("=" * 68)
    print(f"  Host      : {profile.get('hostname')}")
    print(f"  CPU       : {profile.get('cpu_name')}")
    print(f"  Cores     : {profile.get('cpu_cores')}")
    print(f"  RAM       : {profile.get('ram_gb')} GB")
    print(f"  GPU       : {profile.get('gpu_name') or 'None detected'}")
    print(f"  VRAM      : {profile.get('vram_gb')} GB" if profile.get("vram_gb") else "  VRAM      : N/A")
    print(f"  Backend   : {profile.get('gpu_backend')}")
    print(f"  Devices   : {profile.get('gpu_device_count')}")
    print(f"  ROCm      : {profile.get('rocm_version') or 'N/A'}")
    print(f"  PyTorch   : {profile.get('pytorch_version') or 'N/A'}")
    print(f"  Torch HIP : {profile.get('torch_hip_version') or 'N/A'}")
    print(f"  Platform  : {profile.get('platform')} {profile.get('platform_release')}")
    print(f"  Machine   : {profile.get('machine')}")
    print(f"  Git       : {profile.get('git_commit') or 'N/A'}")
    print("=" * 68 + "\n")


def save_profile(profile: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Prime ChainSentinel hardware profile.")
    parser.add_argument("--out", default=None, help="Optional output JSON path, e.g. exports/hardware_profile.json")
    args = parser.parse_args()
    hw = collect()
    print_summary(hw)
    if args.out:
        path = save_profile(hw, args.out)
        print(f"Saved hardware profile to: {path}")
