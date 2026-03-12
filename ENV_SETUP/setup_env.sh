#!/bin/bash
# ============================================================
#  Offroad Segmentation — Linux/macOS Full Setup Script
#  Run from the project root directory:
#    chmod +x ENV_SETUP/setup_env.sh && ./ENV_SETUP/setup_env.sh
#
#  This script will:
#    1. Create full project directory structure
#    2. Guide you to manually place each dataset (3 pause points)
#    3. Verify each dataset after you place it
#    4. Create virtual environment and install all packages
#    5. Verify GPU + CUDA
#
#  MARKER LEGEND (matches REPRODUCE.md):
#    [AUTO_DATA_1]  <- Training Dataset placement checkpoint
#    [AUTO_DATA_2]  <- Test Images placement checkpoint
#    [AUTO_DATA_3]  <- Official Scripts placement checkpoint
# ============================================================

set -e

echo ""
echo "============================================================"
echo "  OFFROAD SEGMENTATION - Full Setup (Linux/macOS)"
echo "  github.com/Daksh-M-Coder/offroad-seg-hackathon-duality-ai"
echo "============================================================"
echo ""

# ─── check project root ──────────────────────────────────────────
if [ ! -f "requirements.txt" ]; then
    echo "[ERROR] Run from the PROJECT ROOT directory."
    echo "        chmod +x ENV_SETUP/setup_env.sh && ./ENV_SETUP/setup_env.sh"
    exit 1
fi

# ─── check Python ────────────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found."
    echo "        sudo apt install python3 python3-venv python3-pip  (Ubuntu)"
    echo "        brew install python                                  (macOS)"
    exit 1
fi
echo "[OK] Python: $(python3 --version)"
echo ""

# ============================================================
#  STEP 1 — Create Directory Structure
# ============================================================
echo "[1/5] Creating project directory structure..."

mkdir -p MODELS
mkdir -p TRAINING_SCRIPTS
mkdir -p ENV_SETUP
mkdir -p SYSTEM_CHECK
mkdir -p DATA_INSIGHTS
mkdir -p "DATASET/Offroad_Segmentation_Training_Dataset/train/Color_Images"
mkdir -p "DATASET/Offroad_Segmentation_Training_Dataset/train/Segmentation"
mkdir -p "DATASET/Offroad_Segmentation_Training_Dataset/val/Color_Images"
mkdir -p "DATASET/Offroad_Segmentation_Training_Dataset/val/Segmentation"
mkdir -p "DATASET/Offroad_Segmentation_testImages"
mkdir -p "DATASET/Offroad_Segmentation_Scripts"
mkdir -p "TRAINING AND PROGRESS/PHASE_1_BASELINE"
mkdir -p "TRAINING AND PROGRESS/PHASE_2_IMPROVED"
mkdir -p "TRAINING AND PROGRESS/PHASE_3_ADVANCED"
mkdir -p "TRAINING AND PROGRESS/PHASE_4_MASTERY"
mkdir -p "TRAINING AND PROGRESS/PHASE_5_CONTROLLED"
mkdir -p "TRAINING AND PROGRESS/PHASE_6_BOUNDARY"
mkdir -p "TESTING_INTERFACE/RESULTS"
mkdir -p "TESTING_INTERFACE/IMGS"

echo "[OK] All directories created."
echo ""

# ============================================================
#  STEP 2 — Dataset Placement (Manual — 3 Checkpoints)
# ============================================================
echo "[2/5] Dataset setup — you will be guided through 3 steps."
echo "      See REPRODUCE.md for full instructions and download links."
echo ""

# ── AUTO_DATA_1 — Training Dataset ───────────────────────────────
echo "╔══════════════════════════════════════════════════════════╗"
echo "║               [AUTO_DATA_1] CHECKPOINT                  ║"
echo "║           Training Dataset Placement Required           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  ACTION REQUIRED:"
echo "  1. Download:  Offroad_Segmentation_Training_Dataset.zip"
echo "     From the Duality AI Hackathon portal / your access link"
echo ""
echo "  2. Unzip the archive"
echo ""
echo "  3. Place the contents so your folder looks like this:"
echo ""
echo "     DATASET/Offroad_Segmentation_Training_Dataset/"
echo "       train/Color_Images/    <- RGB images go here"
echo "       train/Segmentation/   <- Mask files go here"
echo "       val/Color_Images/"
echo "       val/Segmentation/"
echo ""
echo "  4. Press ENTER when done to verify and continue."
echo ""
read -r -p ""

# Verify AUTO_DATA_1
D1="DATASET/Offroad_Segmentation_Training_Dataset/train/Color_Images"
if [ ! -d "$D1" ]; then
    echo "[ERROR] Folder not found: $D1"
    echo "        Make sure you placed the dataset correctly, then re-run this script."
    exit 1
fi
python3 - <<'EOF'
import os, sys
t = len([f for f in os.listdir("DATASET/Offroad_Segmentation_Training_Dataset/train/Color_Images") if os.path.isfile(os.path.join("DATASET/Offroad_Segmentation_Training_Dataset/train/Color_Images", f))])
v = len([f for f in os.listdir("DATASET/Offroad_Segmentation_Training_Dataset/val/Color_Images") if os.path.isfile(os.path.join("DATASET/Offroad_Segmentation_Training_Dataset/val/Color_Images", f))])
print(f"  Train images : {t}  |  Val images : {v}")
print("[OK] Training Dataset verified." if t > 2000 and v > 200 else "[WARN] Low file count — double-check placement.")
EOF
echo ""

# ── AUTO_DATA_2 — Test Images ─────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════╗"
echo "║               [AUTO_DATA_2] CHECKPOINT                  ║"
echo "║             Test Images Placement Required              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  ACTION REQUIRED:"
echo "  1. Download:  Offroad_Segmentation_testImages.zip"
echo "     From the Duality AI Hackathon portal / your access link"
echo ""
echo "  2. Unzip the archive"
echo ""
echo "  3. Place the extracted folder at:"
echo "     DATASET/Offroad_Segmentation_testImages/"
echo ""
echo "  4. Press ENTER when done to verify and continue."
echo ""
read -r -p ""

# Verify AUTO_DATA_2
D2="DATASET/Offroad_Segmentation_testImages"
if [ ! -d "$D2" ] || [ -z "$(ls -A "$D2" 2>/dev/null)" ]; then
    echo "[ERROR] Folder is missing or empty: $D2"
    echo "        Place test images there, then re-run this script."
    exit 1
fi
echo "[OK] Test Images folder found and non-empty."
echo ""

# ── AUTO_DATA_3 — Official Segmentation Scripts ───────────────────
echo "╔══════════════════════════════════════════════════════════╗"
echo "║               [AUTO_DATA_3] CHECKPOINT                  ║"
echo "║       Official Scripts Placement Required               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  ACTION REQUIRED:"
echo "  1. Download:  Offroad_Segmentation_Scripts.zip"
echo "     From the Duality AI Hackathon portal / your access link"
echo ""
echo "  2. Unzip the archive"
echo ""
echo "  3. Place the extracted folder at:"
echo "     DATASET/Offroad_Segmentation_Scripts/"
echo ""
echo "  4. Press ENTER when done to verify and continue."
echo ""
read -r -p ""

# Verify AUTO_DATA_3
D3="DATASET/Offroad_Segmentation_Scripts"
if [ ! -d "$D3" ] || [ -z "$(ls -A "$D3" 2>/dev/null)" ]; then
    echo "[ERROR] Folder is missing or empty: $D3"
    echo "        Place the scripts folder there, then re-run this script."
    exit 1
fi
echo "[OK] Official Scripts folder found and non-empty."
echo ""

echo "============================================================"
echo "  All 3 datasets verified. Continuing with environment setup..."
echo "============================================================"
echo ""

# ============================================================
#  STEP 3 — Create Virtual Environment
# ============================================================
echo "[3/5] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "[SKIP] venv already exists."
else
    python3 -m venv venv
    echo "[OK] venv created."
fi
echo ""

# ============================================================
#  STEP 4 — Install Dependencies
# ============================================================
echo "[4/5] Installing dependencies (PyTorch + CUDA 12.6 + ML libs)..."
echo "      This may take 5-15 minutes on first install..."
echo ""
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Package installation failed!"
    echo "        CPU only: pip install -r requirements.txt"
    exit 1
fi
echo "[OK] All packages installed."
echo ""

# ============================================================
#  STEP 5 — Verify GPU + PyTorch
# ============================================================
echo "[5/5] Verifying GPU and CUDA..."
echo ""
python3 -c "
import torch
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA     : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU      : {torch.cuda.get_device_name(0)}')
else:
    print('  GPU      : Not detected — will use CPU (slower)')
"
echo ""

echo "============================================================"
echo "  FULL SETUP COMPLETE"
echo "============================================================"
echo ""
echo "  Activate environment anytime:"
echo "    source venv/bin/activate"
echo ""
echo "  Start training (Phase 1):"
echo "    python TRAINING_SCRIPTS/train_phase1_baseline.py"
echo ""
echo "  Full step-by-step guide: REPRODUCE.md"
echo ""
