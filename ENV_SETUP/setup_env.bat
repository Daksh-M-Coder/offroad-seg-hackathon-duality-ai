@echo off
REM ============================================================
REM  Offroad Segmentation — Windows Environment Setup
REM  Run this from the project root directory
REM ============================================================

echo.
echo ============================================================
echo   OFFROAD SEGMENTATION - Environment Setup (Windows)
echo ============================================================
echo.

REM --- Check Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)
echo [✓] Python found:
python --version

REM --- Create Virtual Environment ---
echo.
echo [1/4] Creating virtual environment...
if exist "venv" (
    echo       venv already exists, skipping creation.
) else (
    python -m venv venv
    echo       venv created successfully.
)

REM --- Activate venv ---
echo.
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM --- Upgrade pip ---
echo.
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM --- Install Dependencies ---
echo.
echo [4/4] Installing dependencies (PyTorch + CUDA 12.6 + ML libs)...
echo       This may take 5-10 minutes on first install...
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install some packages!
    echo         Check your internet connection and try again.
    pause
    exit /b 1
)

REM --- Verify GPU ---
echo.
echo ============================================================
echo   Verifying GPU and CUDA...
echo ============================================================
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '  No GPU detected')"

echo.
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo   To activate the environment later:
echo     venv\Scripts\activate.bat
echo.
echo   To start training:
echo     python DATASET\Offroad_Segmentation_Scripts\train_phase3.py
echo.
pause
