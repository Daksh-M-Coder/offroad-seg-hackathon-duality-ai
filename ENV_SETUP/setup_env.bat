@echo off
REM ============================================================
REM  Offroad Segmentation — Windows Full Setup Script
REM  Run from the project root directory:
REM    ENV_SETUP\setup_env.bat
REM
REM  This script will:
REM    1. Create full project directory structure
REM    2. Guide you to manually place each dataset (3 pause points)
REM    3. Verify each dataset after you place it
REM    4. Create virtual environment and install all packages
REM    5. Verify GPU + CUDA
REM
REM  MARKER LEGEND (matches REPRODUCE.md):
REM    [AUTO_DATA_1]  ← Training Dataset placement checkpoint
REM    [AUTO_DATA_2]  ← Test Images placement checkpoint
REM    [AUTO_DATA_3]  ← Official Scripts placement checkpoint
REM ============================================================

@echo off
setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo   OFFROAD SEGMENTATION - Full Setup (Windows)
echo   github.com/Daksh-M-Coder/offroad-seg-hackathon-duality-ai
echo ============================================================
echo.

REM ─── check project root ──────────────────────────────────────
if not exist "requirements.txt" (
    echo [ERROR] Run this script from the PROJECT ROOT directory.
    echo         Example:  ENV_SETUP\setup_env.bat
    pause
    exit /b 1
)

REM ─── check Python ────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo         Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)
echo [OK] Python:
python --version
echo.

REM ============================================================
REM  STEP 1 — Create Directory Structure
REM ============================================================
echo [1/5] Creating project directory structure...

mkdir "MODELS"                                                              2>nul
mkdir "TRAINING_SCRIPTS"                                                    2>nul
mkdir "ENV_SETUP"                                                           2>nul
mkdir "SYSTEM_CHECK"                                                        2>nul
mkdir "DATA_INSIGHTS"                                                       2>nul
mkdir "DATASET"                                                             2>nul
mkdir "DATASET\Offroad_Segmentation_Training_Dataset"                       2>nul
mkdir "DATASET\Offroad_Segmentation_Training_Dataset\train"                 2>nul
mkdir "DATASET\Offroad_Segmentation_Training_Dataset\train\Color_Images"    2>nul
mkdir "DATASET\Offroad_Segmentation_Training_Dataset\train\Segmentation"    2>nul
mkdir "DATASET\Offroad_Segmentation_Training_Dataset\val"                   2>nul
mkdir "DATASET\Offroad_Segmentation_Training_Dataset\val\Color_Images"      2>nul
mkdir "DATASET\Offroad_Segmentation_Training_Dataset\val\Segmentation"      2>nul
mkdir "DATASET\Offroad_Segmentation_testImages"                             2>nul
mkdir "DATASET\Offroad_Segmentation_Scripts"                                2>nul
mkdir "TRAINING AND PROGRESS"                                               2>nul
mkdir "TRAINING AND PROGRESS\PHASE_1_BASELINE"                              2>nul
mkdir "TRAINING AND PROGRESS\PHASE_2_IMPROVED"                              2>nul
mkdir "TRAINING AND PROGRESS\PHASE_3_ADVANCED"                              2>nul
mkdir "TRAINING AND PROGRESS\PHASE_4_MASTERY"                               2>nul
mkdir "TRAINING AND PROGRESS\PHASE_5_CONTROLLED"                            2>nul
mkdir "TRAINING AND PROGRESS\PHASE_6_BOUNDARY"                              2>nul
mkdir "TESTING_INTERFACE"                                                   2>nul
mkdir "TESTING_INTERFACE\RESULTS"                                           2>nul
mkdir "TESTING_INTERFACE\IMGS"                                              2>nul

echo [OK] All directories created.
echo.

REM ============================================================
REM  STEP 2 — Dataset Placement (Manual — 3 Checkpoints)
REM ============================================================
echo [2/5] Dataset setup — you will be guided through 3 steps.
echo       See REPRODUCE.md for full instructions and download links.
echo.

REM ── AUTO_DATA_1 — Training Dataset ──────────────────────────
echo ╔══════════════════════════════════════════════════════════╗
echo ║               [AUTO_DATA_1] CHECKPOINT                  ║
echo ║           Training Dataset Placement Required           ║
echo ╚══════════════════════════════════════════════════════════╝
echo.
echo   ACTION REQUIRED:
echo   1. Download:  Offroad_Segmentation_Training_Dataset.zip
echo      From the Duality AI Hackathon portal / your access link
echo.
echo   2. Unzip the archive
echo.
echo   3. Place the contents so your folder looks like this:
echo.
echo      DATASET\Offroad_Segmentation_Training_Dataset\
echo        train\Color_Images\    ^<-- RGB images go here
echo        train\Segmentation\   ^<-- Mask files go here
echo        val\Color_Images\
echo        val\Segmentation\
echo.
echo   4. Press ENTER when done to verify and continue.
echo.
pause

REM — Verify AUTO_DATA_1 ——
set "D1=DATASET\Offroad_Segmentation_Training_Dataset\train\Color_Images"
if not exist "!D1!" (
    echo [ERROR] Folder not found: !D1!
    echo         Make sure you placed the dataset correctly, then re-run this script.
    pause
    exit /b 1
)
python -c "import os; t=len(os.listdir('DATASET/Offroad_Segmentation_Training_Dataset/train/Color_Images')); v=len(os.listdir('DATASET/Offroad_Segmentation_Training_Dataset/val/Color_Images')); print(f'  Train images : {t}  |  Val images : {v}'); ok=t>2000 and v>200; print('[OK] Training Dataset verified.' if ok else '[WARN] Low file count — double-check placement.')"
echo.

REM ── AUTO_DATA_2 — Test Images ────────────────────────────────
echo ╔══════════════════════════════════════════════════════════╗
echo ║               [AUTO_DATA_2] CHECKPOINT                  ║
echo ║             Test Images Placement Required              ║
echo ╚══════════════════════════════════════════════════════════╝
echo.
echo   ACTION REQUIRED:
echo   1. Download:  Offroad_Segmentation_testImages.zip
echo      From the Duality AI Hackathon portal / your access link
echo.
echo   2. Unzip the archive
echo.
echo   3. Place the extracted folder at:
echo      DATASET\Offroad_Segmentation_testImages\
echo.
echo   4. Press ENTER when done to verify and continue.
echo.
pause

REM — Verify AUTO_DATA_2 ——
set "D2=DATASET\Offroad_Segmentation_testImages"
if not exist "!D2!" (
    echo [ERROR] Folder not found: !D2!
    echo         Place the test images folder there, then re-run this script.
    pause
    exit /b 1
)
echo [OK] Test Images folder found.
echo.

REM ── AUTO_DATA_3 — Official Segmentation Scripts ──────────────
echo ╔══════════════════════════════════════════════════════════╗
echo ║               [AUTO_DATA_3] CHECKPOINT                  ║
echo ║       Official Scripts Placement Required               ║
echo ╚══════════════════════════════════════════════════════════╝
echo.
echo   ACTION REQUIRED:
echo   1. Download:  Offroad_Segmentation_Scripts.zip
echo      From the Duality AI Hackathon portal / your access link
echo.
echo   2. Unzip the archive
echo.
echo   3. Place the extracted folder at:
echo      DATASET\Offroad_Segmentation_Scripts\
echo.
echo   4. Press ENTER when done to verify and continue.
echo.
pause

REM — Verify AUTO_DATA_3 ——
set "D3=DATASET\Offroad_Segmentation_Scripts"
if not exist "!D3!" (
    echo [ERROR] Folder not found: !D3!
    echo         Place the scripts folder there, then re-run this script.
    pause
    exit /b 1
)
echo [OK] Official Scripts folder found.
echo.

echo ============================================================
echo   All 3 datasets verified. Continuing with environment setup...
echo ============================================================
echo.

REM ============================================================
REM  STEP 3 — Create Virtual Environment
REM ============================================================
echo [3/5] Creating virtual environment...
if exist "venv" (
    echo [SKIP] venv already exists.
) else (
    python -m venv venv
    echo [OK] venv created.
)
echo.

REM ============================================================
REM  STEP 4 — Install Dependencies
REM ============================================================
echo [4/5] Installing dependencies (PyTorch + CUDA 12.6 + ML libs)...
echo       This may take 5-15 minutes on first install...
echo.
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install some packages!
    echo         Check your internet connection and CUDA version.
    echo         CPU only: remove --extra-index-url from the command.
    pause
    exit /b 1
)
echo [OK] All packages installed.
echo.

REM ============================================================
REM  STEP 5 — Verify GPU + PyTorch
REM ============================================================
echo [5/5] Verifying GPU and CUDA...
echo.
python -c "import torch; print(f'  PyTorch  : {torch.__version__}'); print(f'  CUDA     : {torch.cuda.is_available()}'); print(f'  GPU      : {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '  GPU      : Not detected — will use CPU (slower)')"
echo.

echo ============================================================
echo   FULL SETUP COMPLETE
echo ============================================================
echo.
echo   Activate environment anytime:
echo     venv\Scripts\activate.bat
echo.
echo   Start training (Phase 1):
echo     python TRAINING_SCRIPTS\train_phase1_baseline.py
echo.
echo   Full step-by-step guide: REPRODUCE.md
echo.
pause
endlocal
