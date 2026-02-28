"""
GPU & CUDA Verification Script
Checks if PyTorch can detect and use your NVIDIA GPU.
"""
import torch
import sys

def main():
    print("=" * 55)
    print("  GPU & CUDA VERIFICATION")
    print("=" * 55)

    # PyTorch version
    print(f"\n  PyTorch Version   : {torch.__version__}")
    print(f"  Python Version    : {sys.version.split()[0]}")

    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\n  CUDA Available    : {cuda_available}")

    if not cuda_available:
        print("\n  [FAIL] CUDA is NOT available!")
        print("  Possible reasons:")
        print("    - PyTorch installed without CUDA support")
        print("    - NVIDIA drivers not installed")
        print("    - No NVIDIA GPU detected")
        print("=" * 55)
        return

    # CUDA details
    print(f"  CUDA Version      : {torch.version.cuda}")
    print(f"  cuDNN Version     : {torch.backends.cudnn.version()}")
    print(f"  cuDNN Enabled     : {torch.backends.cudnn.enabled}")

    # GPU details
    gpu_count = torch.cuda.device_count()
    print(f"\n  GPU Count         : {gpu_count}")

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\n  --- GPU {i} ---")
        print(f"  Name              : {props.name}")
        print(f"  Total VRAM        : {props.total_memory / (1024**3):.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi Processors  : {props.multi_processor_count}")

    # Current GPU
    current = torch.cuda.current_device()
    print(f"\n  Active GPU        : {current} ({torch.cuda.get_device_name(current)})")

    # Quick tensor test on GPU
    print("\n  Running GPU tensor test...")
    try:
        a = torch.randn(1000, 1000, device="cuda")
        b = torch.randn(1000, 1000, device="cuda")
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        print("  [PASS] GPU tensor operations working!")
    except Exception as e:
        print(f"  [FAIL] GPU tensor test failed: {e}")

    # Memory info
    print(f"\n  VRAM Allocated    : {torch.cuda.memory_allocated() / (1024**2):.1f} MB")
    print(f"  VRAM Reserved     : {torch.cuda.memory_reserved() / (1024**2):.1f} MB")

    # Cleanup
    del a, b, c
    torch.cuda.empty_cache()

    print(f"\n  [PASS] Everything looks good!")
    print("=" * 55)


if __name__ == "__main__":
    main()
