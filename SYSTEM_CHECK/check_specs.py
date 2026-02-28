import platform, os, sys, shutil, json
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

specs = {}

# OS & Python
specs["os"] = f"{platform.system()} {platform.release()} ({platform.version()})"
specs["python_version"] = sys.version.split()[0]
specs["python_path"] = sys.executable
specs["architecture"] = platform.machine()

# CPU
specs["cpu_processor"] = platform.processor()
specs["cpu_logical_cores"] = os.cpu_count()
if HAS_PSUTIL:
    specs["cpu_physical_cores"] = psutil.cpu_count(logical=False)
    freq = psutil.cpu_freq()
    if freq:
        specs["cpu_max_freq_mhz"] = round(freq.max)
        specs["cpu_current_freq_mhz"] = round(freq.current)

# RAM
if HAS_PSUTIL:
    vm = psutil.virtual_memory()
    specs["ram_total_gb"] = round(vm.total / (1024**3), 1)
    specs["ram_available_gb"] = round(vm.available / (1024**3), 1)
    specs["ram_used_percent"] = vm.percent

# Disks
if HAS_PSUTIL:
    disks = []
    for p in psutil.disk_partitions():
        if p.fstype:
            try:
                u = psutil.disk_usage(p.mountpoint)
                disks.append({
                    "drive": p.device,
                    "fs": p.fstype,
                    "total_gb": round(u.total / (1024**3), 1),
                    "free_gb": round(u.free / (1024**3), 1),
                    "used_pct": u.percent
                })
            except Exception:
                pass
    specs["disks"] = disks

# GPU (nvidia-smi)
import subprocess
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader,nounits"],
        stderr=subprocess.DEVNULL, text=True
    )
    gpus = []
    for line in out.strip().split("\n"):
        parts = [x.strip() for x in line.split(",")]
        gpus.append({"name": parts[0], "vram_mib": parts[1], "driver": parts[2]})
    specs["nvidia_gpus"] = gpus
except Exception:
    specs["nvidia_gpus"] = "Not detected"

# Write to file
with open("specs_result.json", "w") as f:
    json.dump(specs, f, indent=2)

print("DONE")
