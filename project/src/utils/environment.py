import os
import platform
import shutil
import subprocess
from pathlib import Path


def describe_hardware():
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    hardware = {
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
    }
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            hardware["gpus"] = result.stdout.strip().splitlines()
    return hardware
