"""
System Specs Checker – Shows CPU, GPU, RAM, Disk, OS, Python info.
"""
import platform
import os
import sys
import subprocess
import shutil

def hr():
    print("=" * 60)

def get_cpu_info():
    hr()
    print("🖥️  CPU INFO")
    hr()
    print(f"  Processor      : {platform.processor()}")
    print(f"  Architecture   : {platform.machine()}")
    print(f"  Logical Cores  : {os.cpu_count()}")
    try:
        import psutil
        freq = psutil.cpu_freq()
        if freq:
            print(f"  Max Freq       : {freq.max:.0f} MHz")
            print(f"  Current Freq   : {freq.current:.0f} MHz")
        phys = psutil.cpu_count(logical=False)
        print(f"  Physical Cores : {phys}")
    except ImportError:
        print("  (Install psutil for more detail: pip install psutil)")

def get_gpu_info():
    hr()
    print("🎮  GPU INFO")
    hr()
    # Try NVIDIA
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,cuda_version",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True
        )
        for i, line in enumerate(out.strip().split("\n")):
            parts = [p.strip() for p in line.split(",")]
            print(f"  GPU {i}          : {parts[0]}")
            print(f"  VRAM           : {parts[1]} MiB")
            print(f"  Driver         : {parts[2]}")
            print(f"  CUDA Version   : {parts[3]}")
    except Exception:
        print("  No NVIDIA GPU detected (nvidia-smi not found).")
    
    # Try PyTorch CUDA
    try:
        import torch
        print(f"  PyTorch CUDA   : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Device    : {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  CUDA VRAM      : {mem:.1f} GB")
    except ImportError:
        print("  (PyTorch not installed yet)")

def get_ram_info():
    hr()
    print("🧠  RAM INFO")
    hr()
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"  Total RAM      : {vm.total / (1024**3):.1f} GB")
        print(f"  Available      : {vm.available / (1024**3):.1f} GB")
        print(f"  Used           : {vm.used / (1024**3):.1f} GB ({vm.percent}%)")
    except ImportError:
        # Fallback for Windows
        try:
            out = subprocess.check_output(
                ["wmic", "computersystem", "get", "totalphysicalmemory"],
                text=True, stderr=subprocess.DEVNULL
            )
            for line in out.strip().split("\n"):
                line = line.strip()
                if line.isdigit():
                    print(f"  Total RAM      : {int(line) / (1024**3):.1f} GB")
        except Exception:
            print("  (Install psutil for RAM info: pip install psutil)")

def get_disk_info():
    hr()
    print("💾  DISK INFO")
    hr()
    try:
        import psutil
        for part in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(part.mountpoint)
                print(f"  Drive {part.device}")
                print(f"    File System  : {part.fstype}")
                print(f"    Total        : {usage.total / (1024**3):.1f} GB")
                print(f"    Used         : {usage.used / (1024**3):.1f} GB ({usage.percent}%)")
                print(f"    Free         : {usage.free / (1024**3):.1f} GB")
            except PermissionError:
                print(f"  Drive {part.device} — access denied")
    except ImportError:
        # Fallback
        total, used, free = shutil.disk_usage("C:\\")
        print(f"  C:\\ Total      : {total / (1024**3):.1f} GB")
        print(f"  C:\\ Used       : {used / (1024**3):.1f} GB")
        print(f"  C:\\ Free       : {free / (1024**3):.1f} GB")

def get_os_python_info():
    hr()
    print("🐍  OS & PYTHON INFO")
    hr()
    print(f"  OS             : {platform.system()} {platform.release()}")
    print(f"  OS Version     : {platform.version()}")
    print(f"  Python         : {sys.version}")
    print(f"  Python Path    : {sys.executable}")

if __name__ == "__main__":
    print()
    print("🔍  SYSTEM SPECIFICATIONS REPORT")
    get_os_python_info()
    get_cpu_info()
    get_gpu_info()
    get_ram_info()
    get_disk_info()
    hr()
    print("✅  Done!")
    print()
