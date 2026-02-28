#!/bin/bash
# ============================================================
#  Offroad Segmentation — Linux/macOS Environment Setup
#  Run this from the project root directory:
#    chmod +x ENV_SETUP/setup_env.sh && ./ENV_SETUP/setup_env.sh
# ============================================================

set -e

echo ""
echo "============================================================"
echo "  OFFROAD SEGMENTATION - Environment Setup (Linux/macOS)"
echo "============================================================"
echo ""

# --- Check Python ---
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed."
    echo "        Install with: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi
echo "[✓] Python found: $(python3 --version)"

# --- Create Virtual Environment ---
echo ""
echo "[1/4] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "       venv already exists, skipping creation."
else
    python3 -m venv venv
    echo "       venv created successfully."
fi

# --- Activate venv ---
echo ""
echo "[2/4] Activating virtual environment..."
source venv/bin/activate

# --- Upgrade pip ---
echo ""
echo "[3/4] Upgrading pip..."
pip install --upgrade pip --quiet

# --- Install Dependencies ---
echo ""
echo "[4/4] Installing dependencies (PyTorch + CUDA 12.6 + ML libs)..."
echo "       This may take 5-10 minutes on first install..."
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Failed to install some packages!"
    echo "        Check your internet connection and try again."
    exit 1
fi

# --- Verify GPU ---
echo ""
echo "============================================================"
echo "  Verifying GPU and CUDA..."
echo "============================================================"
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
else:
    print('  No GPU detected (CPU mode)')
"

echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "  To activate the environment later:"
echo "    source venv/bin/activate"
echo ""
echo "  To start training:"
echo "    python DATASET/Offroad_Segmentation_Scripts/train_phase3.py"
echo ""
