"""Print installation, acceleration, and free-disk diagnostics for local training."""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
from pathlib import Path
import shutil
import sys

from brain_mri.storage_paths import STORAGE_ROOT, require_ssd_if_targeted


def collect(data_root: Path) -> dict:
    require_ssd_if_targeted(data_root)
    data_root = data_root.expanduser().resolve()
    disk_path = next((p for p in (data_root, *data_root.parents) if p.exists()), None)
    disk = shutil.disk_usage(disk_path or Path.cwd())
    result = {
        "python": sys.version.split()[0],
        "python_compatible": sys.version_info >= (3, 10),
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "requested_data_root": str(data_root),
        "free_disk_GiB": round(disk.free / 2**30, 2),
        "nibabel_installed": importlib.util.find_spec("nibabel") is not None,
        "matplotlib_installed": importlib.util.find_spec("matplotlib") is not None,
        "streamlit_installed": importlib.util.find_spec("streamlit") is not None,
        "torch_installed": importlib.util.find_spec("torch") is not None,
    }
    if result["torch_installed"]:
        import torch
        result.update({
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(getattr(torch.backends, "mps", None)
                                  and torch.backends.mps.is_available()),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=STORAGE_ROOT)
    args = parser.parse_args()
    report = collect(args.data_root)
    print(json.dumps(report, indent=2))
    if not report["python_compatible"]:
        raise SystemExit("Python 3.10+ required (Python 3.11 recommended)")


if __name__ == "__main__":
    main()
