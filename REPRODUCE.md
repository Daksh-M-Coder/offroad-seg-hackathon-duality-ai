# REPRODUCE.md — Complete Reproduction Guide

---

## 📌 What Is This File?

This document is the **single authoritative guide** to reproduce the complete **Offroad Semantic Scene Segmentation** pipeline — from an empty folder all the way to a trained model achieving **Multi-Scale TTA IoU of 0.5527** across 10 semantic classes of desert/offroad terrain.

**Who is this for?**

| Reader | Purpose |
|---|---|
| **Judges / Evaluators** | Verify that every result was produced reproducibly and honestly — follow Phase 1→6 in order and you'll get the same numbers |
| **Fellow GitHub Visitors** | Learn the exact ML engineering decisions (architecture, loss, LR strategy) that took us from IoU 0.30 to 0.55 across 6 progressive training phases |
| **Students / Researchers** | Reference implementation for DINOv2-based semantic segmentation with controlled backbone fine-tuning and boundary-aware losses |
| **Future Self** | Resume the project months later without losing context — every decision and result is recorded here |

**What this guide covers:**

1. **Setup script** — one script creates all directories, pauses for dataset placement at 3 clearly-marked checkpoints, installs venv and packages, and checks GPU
2. **Phase-by-phase training** — exact commands for all 6 phases with expected metrics, times, and notes on anomalies
3. **Troubleshooting** — the most common issues we encountered, with exact fixes
4. **File management rules** — naming conventions, checkpoint discipline, and folder governance

> Total training time: approximately **28 hours** (GPU required for Phase 3+). Each phase checkpoint is a valid standalone result.

---

## 🔖 Marker System

The setup scripts and this guide use **unique checkpoint markers** so you always know exactly where you are in the process.

| Marker | Meaning |
|---|---|
| `[AUTO_DATA_1]` | Training Dataset placement checkpoint — **script pauses here** |
| `[AUTO_DATA_2]` | Test Images placement checkpoint — **script pauses here** |
| `[AUTO_DATA_3]` | Official Scripts placement checkpoint — **script pauses here** |

When the script prints `[AUTO_DATA_1]`, find the matching section in this guide for the exact download instructions.

---

## 📋 Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the Repository](#2-clone-the-repository)
3. [Run the Setup Script](#3-run-the-setup-script)
4. [AUTO_DATA_1 — Training Dataset](#auto_data_1--training-dataset)
5. [AUTO_DATA_2 — Test Images](#auto_data_2--test-images)
6. [AUTO_DATA_3 — Official Scripts](#auto_data_3--official-scripts)
7. [Dataset Details and Verification](#7-dataset-details-and-verification)
8. [Phase 1 — Baseline](#8-phase-1--baseline-training)
9. [Phase 2 — Improved](#9-phase-2--improved-training)
10. [Phase 3 — Advanced](#10-phase-3--advanced-training)
11. [Phase 4 — Mastery](#11-phase-4--mastery-training)
12. [Phase 5 — Controlled Backbone Fine-Tuning](#12-phase-5--controlled-backbone-fine-tuning)
13. [Phase 6 — Boundary-Aware Fine-Tuning ⭐](#13-phase-6--boundary-aware-fine-tuning-)
14. [Visual Testing Interface](#14-visual-testing-interface)
15. [File Naming & Project Management Rules](#15-file-naming--project-management-rules)
16. [Troubleshooting](#16-troubleshooting)
17. [Expected Results](#17-expected-results-summary)

---

## 1. Prerequisites

| Requirement | Minimum | **What We Used (Recommended)** |
|---|---|---|
| **OS** | Windows 10 / Ubuntu 20.04 / macOS 12 | **Windows 11 23H2** |
| **Python** | 3.10 | **3.11.9** |
| **GPU** | 6 GB VRAM NVIDIA | **NVIDIA RTX 3050 6 GB Laptop GPU** |
| **CUDA** | 11.8 | **12.6** |
| **RAM** | 8 GB | **16 GB** |
| **Disk** | 10 GB free | **20 GB** (dataset + venv + models + plots) |
| **Internet** | Required for pip + DINOv2 | Required (DINOv2 ~330 MB on first Phase 3 run) |

### CUDA Version Note

The setup script installs **PyTorch with CUDA 12.6** by default. If your CUDA version differs:

```bash
# In requirements.txt or manual install, change cu126 to:
# CUDA 11.8 →  --extra-index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1 →  --extra-index-url https://download.pytorch.org/whl/cu121
# CPU only  →  remove --extra-index-url entirely
```

---

## 2. Clone the Repository

```bash
git clone https://github.com/Daksh-M-Coder/offroad-seg-hackathon-duality-ai.git
cd offroad-seg-hackathon-duality-ai
```

The cloned repo contains:
```
offroad-seg-hackathon-duality-ai/
├── TRAINING_SCRIPTS/        ← All 6 phase scripts + recover_phase6_outputs.py
├── ENV_SETUP/               ← setup_env.bat / setup_env.sh / project_management.md
├── TESTING_INTERFACE/       ← Gradio visual tester (app.py)
├── SYSTEM_CHECK/            ← GPU/CUDA verification
├── DATA_INSIGHTS/           ← Dataset analysis
├── requirements.txt
├── README.md
└── REPRODUCE.md             ← You are here
```

> `venv/`, `DATASET/`, `MODELS/`, `TRAINING AND PROGRESS/` are created by the setup script — **not in the repo**.

---

## 3. Run the Setup Script

### Windows

```cmd
ENV_SETUP\setup_env.bat
```

### Linux / macOS

```bash
chmod +x ENV_SETUP/setup_env.sh
./ENV_SETUP/setup_env.sh
```

The script runs 5 steps:

| Step | Action |
|---|---|
| **1** | Creates full directory structure (MODELS/, DATASET/, TRAINING AND PROGRESS/PHASE_1-6/, TESTING_INTERFACE/RESULTS/, TESTING_INTERFACE/IMGS/) |
| **2** | `[AUTO_DATA_1]` pause → dataset placement + verification |
| **2** | `[AUTO_DATA_2]` pause → test images placement + verification |
| **2** | `[AUTO_DATA_3]` pause → scripts placement + verification |
| **3** | Creates Python `venv/` |
| **4** | Installs all packages (`requirements.txt` + PyTorch CUDA 12.6) |
| **5** | Verifies GPU, CUDA, and PyTorch version |

**When you see a boxed `[AUTO_DATA_X]` message, the script is waiting for you.** Find the matching section below, follow the instructions, then press Enter to continue.

---

## AUTO_DATA_1 — Training Dataset

> **You will see this when the setup script pauses at Step 2a.**

### What to download

**Archive name**: `Offroad_Segmentation_Training_Dataset.zip`  
**Source**: Duality AI Hackathon portal (requires hackathon access)

### Where to place it

1. Download `Offroad_Segmentation_Training_Dataset.zip`
2. Unzip it — you will get a folder called `Offroad_Segmentation_Training_Dataset/`
3. Place its **contents** inside the already-created `DATASET/Offroad_Segmentation_Training_Dataset/`

**After placement, your structure must look exactly like this:**

```
DATASET/
└── Offroad_Segmentation_Training_Dataset/
    ├── train/
    │   ├── Color_Images/       ← 2857 RGB .jpg images
    │   └── Segmentation/       ← 2857 matching mask .jpg files
    └── val/
        ├── Color_Images/       ← 317 RGB .jpg images
        └── Segmentation/       ← 317 matching mask .jpg files
```

4. Press **Enter** in the setup script terminal to verify and continue.

**The script checks**:
- `DATASET/Offroad_Segmentation_Training_Dataset/train/Color_Images/` exists
- File count: train > 2000, val > 200

---

## AUTO_DATA_2 — Test Images

> **You will see this when the setup script pauses at Step 2b.**

### What to download

**Archive name**: `Offroad_Segmentation_testImages.zip`  
**Source**: Duality AI Hackathon portal (requires hackathon access)

### Where to place it

1. Download `Offroad_Segmentation_testImages.zip`
2. Unzip it — you will get a folder (typically `Offroad_Segmentation_testImages/`)
3. Place the extracted folder inside `DATASET/` so the result is:

```
DATASET/
└── Offroad_Segmentation_testImages/
    └── ...   ← test images (no ground-truth masks)
```

4. Press **Enter** in the setup script terminal to verify and continue.

**The script checks**: `DATASET/Offroad_Segmentation_testImages/` exists and is non-empty.

> These are the **held-out test images** used for final submission scoring. No ground-truth masks are provided for them — they are used with the `TESTING_INTERFACE/app.py` Upload tab.

---

## AUTO_DATA_3 — Official Scripts

> **You will see this when the setup script pauses at Step 2c.**

### What to download

**Archive name**: `Offroad_Segmentation_Scripts.zip`  
**Source**: Duality AI Hackathon portal (requires hackathon access)

### Where to place it

1. Download `Offroad_Segmentation_Scripts.zip`
2. Unzip it — gives you `Offroad_Segmentation_Scripts/` folder
3. Place it inside `DATASET/`:

```
DATASET/
└── Offroad_Segmentation_Scripts/
    └── ...   ← Duality AI's official baseline scripts
```

4. Press **Enter** in the setup script terminal to verify and continue.

**The script checks**: `DATASET/Offroad_Segmentation_Scripts/` exists and is non-empty.

> These are Duality AI's official baseline scripts. Useful for reference — our Phase 1 starts from scratch rather than these, but they provide context for the task definition and evaluation protocol.

---

## 7. Dataset Details and Verification

### Class Labels (Mask Pixel Value → Class ID)

| Pixel Value | Class ID | Class Name |
|---|---|---|
| 0 | 0 | Background |
| 100 | 1 | Trees |
| 200 | 2 | Lush Bushes |
| 300 | 3 | Dry Grass |
| 500 | 4 | Dry Bushes |
| 550 | 5 | Ground Clutter |
| 700 | 6 | Logs |
| 800 | 7 | Rocks |
| 7100 | 8 | Landscape |
| 10000 | 9 | Sky |

### Manually Verify Dataset After Setup

```bash
# Windows
venv\Scripts\python.exe -c "import os; t=len(os.listdir('DATASET/Offroad_Segmentation_Training_Dataset/train/Color_Images')); v=len(os.listdir('DATASET/Offroad_Segmentation_Training_Dataset/val/Color_Images')); print(f'Train: {t}, Val: {v}')"

# Linux/macOS
venv/bin/python -c "import os; t=len(os.listdir('DATASET/Offroad_Segmentation_Training_Dataset/train/Color_Images')); v=len(os.listdir('DATASET/Offroad_Segmentation_Training_Dataset/val/Color_Images')); print(f'Train: {t}, Val: {v}')"
```

Expected: `Train: 2857, Val: 317`

---

## 8. Phase 1 — Baseline Training

> **Goal**: Establish starting point. Simple CNN head, SGD, 10 epochs, fully frozen backbone.  
> **Expected Val IoU**: ~0.297 | **Expected time**: ~83 minutes

```bash
# Activate environment first (if not already active)
# Windows:  venv\Scripts\activate.bat
# Linux:    source venv/bin/activate

python TRAINING_SCRIPTS/train_phase1_baseline.py
```

### Archive the best model

```bash
# Windows (PowerShell)
Copy-Item "TRAINING AND PROGRESS\PHASE_1_BASELINE\best_model.pth" "MODELS\phase1_best_model_iou0.2971.pth"

# Linux/macOS
cp "TRAINING AND PROGRESS/PHASE_1_BASELINE/best_model.pth" "MODELS/phase1_best_model_iou0.2971.pth"
```

> Replace `0.2971` with your actual Val IoU from the run.

---

## 9. Phase 2 — Improved Training

> **Goal**: AdamW + augmentations + class weights + CosineAnnealing. 30 epochs.  
> **Expected Val IoU**: ~0.404 (+35.8%) | **Expected time**: ~247 minutes

```bash
python TRAINING_SCRIPTS/train_phase2_improved.py
```

```bash
cp "TRAINING AND PROGRESS/PHASE_2_IMPROVED/best_model.pth" "MODELS/phase2_best_model_iou0.4036.pth"
```

---

## 10. Phase 3 — Advanced Training

> **Goal**: DINOv2 ViT-Base backbone (frozen) + UPerNet head + Focal+Dice loss + 644×364. 40 epochs.  
> **Expected Val IoU**: ~0.516 (+27.9%) | **Expected time**: ~420 minutes (~7 hours, GPU required)

> ⚠️ **First run**: DINOv2 (~330 MB) downloads from `torch.hub` automatically. Requires internet.

```bash
python TRAINING_SCRIPTS/train_phase3_advanced.py
```

```bash
cp "TRAINING AND PROGRESS/PHASE_3_ADVANCED/best_model.pth" "MODELS/phase3_best_model_iou0.5161.pth"
```

---

## 11. Phase 4 — Mastery Training

> **Goal**: Multi-scale augmentation + loss rebalancing + HFlip TTA. Backbone still frozen.  
> **Expected Val IoU**: ~0.517 TTA | **Expected time**: ~121 minutes (early-stops at Ep 11 — normal)

```bash
python TRAINING_SCRIPTS/train_phase4_mastery.py
```

```bash
cp "TRAINING AND PROGRESS/PHASE_4_MASTERY/best_model.pth" "MODELS/phase4_best_model_iou0.5150.pth"
```

> **Phase 4 early-stops at Epoch 11** — this is correct. The frozen backbone has hit its ceiling. Proceed to Phase 5.

---

## 12. Phase 5 — Controlled Backbone Fine-Tuning

> **Goal**: Unfreeze DINOv2 blocks 10-11, differential LR (backbone 5e-6 / head 2e-4), gradient clipping.  
> **Expected Val IoU**: ~0.529 non-TTA, ~0.531 HFlip TTA | **Expected time**: ~377 minutes (~6.3 hours)

> ⚠️ **Requires** `TRAINING AND PROGRESS/PHASE_4_MASTERY/best_model.pth`

```bash
python TRAINING_SCRIPTS/train_phase5_controlled.py
```

```bash
cp "TRAINING AND PROGRESS/PHASE_5_CONTROLLED/best_model.pth" "MODELS/phase5_best_model_iou0.5294.pth"
```

### ⚠️ CRITICAL: Backup Phase 5 before starting Phase 6

```bash
# Windows (PowerShell)
Copy-Item "MODELS\phase5_best_model_iou0.5294.pth" "MODELS\phase5_best_model_iou0.5294 - Backup.pth"

# Linux/macOS
cp "MODELS/phase5_best_model_iou0.5294.pth" "MODELS/phase5_best_model_iou0.5294 - Backup.pth"
```

---

## 13. Phase 6 — Boundary-Aware Fine-Tuning ⭐

> **Goal**: Unfreeze blocks 9-11 + Boundary Loss + Multi-Scale TTA (8 passes: 0.9×–1.2× × HFlip).  
> **Expected Val IoU**: ~0.537 non-TTA, **~0.553 Multi-Scale TTA** | **Expected time**: ~439 minutes (~7.3 hours)

> ⚠️ **Requires** `TRAINING AND PROGRESS/PHASE_5_CONTROLLED/best_model.pth`

```bash
python TRAINING_SCRIPTS/train_phase6_boundary.py
```

### What to expect during Phase 6

```
Ep 1  | Val IoU: 0.527   ← strong warm-start from Phase 5 weights
Ep 2-4 | Val IoU dips to 0.521  ← block 9 adapting (NORMAL — do not stop)
Ep 5-9 | Recovery to 0.527+
Ep 13  | Val IoU 0.530   ← exceeds Phase 5's TTA score WITHOUT TTA
Ep 28  | Val IoU 0.537   ← best checkpoint saved
[Multi-Scale TTA runs automatically after Epoch 30]
TTA IoU: 0.553
Total: ~7.3 hours
```

```bash
cp "TRAINING AND PROGRESS/PHASE_6_BOUNDARY/best_model.pth" "MODELS/phase6_best_model_iou0.5368.pth"
```

### If training ends with UnicodeEncodeError

Model is already saved — only the output text report failed. Run:

```bash
python TRAINING_SCRIPTS/recover_phase6_outputs.py
```

---

## 14. Visual Testing Interface

```bash
# Windows
venv\Scripts\python.exe TESTING_INTERFACE\app.py

# Linux/macOS
venv/bin/python TESTING_INTERFACE/app.py
```

Open **`http://localhost:7860`**

| Tab | Purpose | GT Available |
|---|---|---|
| **Class Samples** | 10 fixed samples per class — reproducible | ✅ |
| **Random Pick** | Random from full class pool | ✅ |
| **Upload** | Your own unseen image | ❌ |

Results auto-saved to `TESTING_INTERFACE/RESULTS/` (Markdown + 3 images per prediction).

---

## 15. File Naming & Project Management Rules

### Model Archive Convention

```
MODELS/phaseN_best_model_iou{X.XXXX}.pth
```

The IoU in the filename means you can see every phase's performance at a glance.

### Phase Output Standard

Every phase produces exactly:

| File | Contents |
|---|---|
| `00_phaseN_log.md` | Full training report (13 sections) |
| `best_model.pth` | Best checkpoint by Val IoU |
| `final_model.pth` | Last epoch |
| `all_metrics_curves.png` | Loss / IoU / Dice / Accuracy |
| `lr_schedule.png` | LR curve |
| `overfit_gap.png` | Train-Val gap (Phase 5+) |
| `evaluation_metrics.txt` | Per-epoch table (UTF-8) |
| `history.json` | Machine-readable metrics |

### One Rule Above All

> **Never delete. Never overwrite. Never skip documentation.**  
> Every phase folder is a permanent record. `MODELS/` backups are your restore points.

---

## 16. Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

Not in the venv:
```bash
venv\Scripts\activate.bat    # Windows
source venv/bin/activate     # Linux/macOS
```

### `CUDA out of memory`

Reduce batch size in the script:
```python
batch_size  = 1   # default: 2
accum_steps = 4   # keep effective batch = 4
```

### `DINOv2 download fails` (Phase 3+)

Clear hub cache and retry:
```bash
# Linux/macOS
rm -rf ~/.cache/torch/hub/
# Windows PowerShell
Remove-Item -Recurse "$env:USERPROFILE\.cache\torch\hub"
```

### `FileNotFoundError: Phase N checkpoint not found`

Each phase requires the previous phase's `best_model.pth`. Run phases in order 1→2→3→4→5→6.

### `UnicodeEncodeError` at end of Phase 6

Model is already saved. Run:
```bash
python TRAINING_SCRIPTS/recover_phase6_outputs.py
```

### `Phase 4 early-stopped at Epoch 11`

**Correct and expected** — frozen backbone ceiling. Proceed to Phase 5.

### `⛔ 3 consecutive val drops` warning (Phase 5/6)

Printed warning only — not a training halt (unless gap also > 0.05). The Epoch 4 dip during Phase 6 triggered this; training recovered by Epoch 9.

### Setup script verification fails after placing dataset

Check that you placed the **contents** of the unzipped folder at the right path. The script checks for `Color_Images/` inside `train/` specifically.

---

## 17. Expected Results Summary

| Phase | Script | Val IoU | TTA IoU | Time |
|---|---|---|---|---|
| 1 | `train_phase1_baseline.py` | 0.2971 | — | ~83 min |
| 2 | `train_phase2_improved.py` | 0.4036 | — | ~247 min |
| 3 | `train_phase3_advanced.py` | 0.5161 | — | ~420 min |
| 4 | `train_phase4_mastery.py` | 0.5150 | 0.5169 | ~121 min |
| 5 | `train_phase5_controlled.py` | 0.5294 | 0.5310 | ~377 min |
| **6 ⭐** | `train_phase6_boundary.py` | **0.5368** | **0.5527** | **~439 min** |
| **Total** | | | | **~28 hours** |

> Measured on **NVIDIA RTX 3050 6 GB Laptop GPU**. Seeds set in Phases 5-6 (torch=42, numpy=42, random=42) for reproducibility.

---

## 🔗 Repository Links

**GitHub**: [https://github.com/Daksh-M-Coder/offroad-seg-hackathon-duality-ai](https://github.com/Daksh-M-Coder/offroad-seg-hackathon-duality-ai)

| Resource | Link |
|---|---|
| Training Scripts | [TRAINING_SCRIPTS/](https://github.com/Daksh-M-Coder/offroad-seg-hackathon-duality-ai/tree/main/TRAINING_SCRIPTS) |
| Phase 6 Script | [train_phase6_boundary.py](https://github.com/Daksh-M-Coder/offroad-seg-hackathon-duality-ai/blob/main/TRAINING_SCRIPTS/train_phase6_boundary.py) |
| Windows Setup | [ENV_SETUP/setup_env.bat](https://github.com/Daksh-M-Coder/offroad-seg-hackathon-duality-ai/blob/main/ENV_SETUP/setup_env.bat) |
| Linux/macOS Setup | [ENV_SETUP/setup_env.sh](https://github.com/Daksh-M-Coder/offroad-seg-hackathon-duality-ai/blob/main/ENV_SETUP/setup_env.sh) |
| Testing Interface | [TESTING_INTERFACE/app.py](https://github.com/Daksh-M-Coder/offroad-seg-hackathon-duality-ai/blob/main/TESTING_INTERFACE/app.py) |
| Phase 6 Report | [TRAINING%20AND%20PROGRESS/PHASE_6_BOUNDARY/00_phase6_log.md](https://github.com/Daksh-M-Coder/offroad-seg-hackathon-duality-ai/blob/main/TRAINING%20AND%20PROGRESS/PHASE_6_BOUNDARY/00_phase6_log.md) |
| Requirements | [requirements.txt](https://github.com/Daksh-M-Coder/offroad-seg-hackathon-duality-ai/blob/main/requirements.txt) |
| Full README | [README.md](https://github.com/Daksh-M-Coder/offroad-seg-hackathon-duality-ai/blob/main/README.md) |

---

*Complete reproduction guide for the Offroad Semantic Scene Segmentation project — Phase 1 to Phase 6, IoU 0.30 → 0.5527.*
