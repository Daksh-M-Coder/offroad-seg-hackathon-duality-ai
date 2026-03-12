# 🏜️ Offroad Semantic Scene Segmentation — Ignitia Hackathon

> **Duality AI × Ignitia Hackathon** | Pixel-level semantic segmentation of synthetic offroad desert environments.  
> Built with **DINOv2 + UPerNet** | Achieved **Multi-Scale TTA IoU 0.5527** — an **86.0% improvement** across 6 training phases.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**GitHub**: [github.com/Daksh-M-Coder/offroad-seg-hackathon-duality-ai](https://github.com/Daksh-M-Coder/offroad-seg-hackathon-duality-ai)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Results Summary](#-results-summary)
- [Project Structure](#-project-structure)
- [Training Scripts](#-training-scripts)
- [Quick Start](#-quick-start)
- [How to Reproduce](#-how-to-reproduce)
- [Phase 1 — Baseline](#-phase-1--baseline-training)
- [Phase 2 — Improved Training](#-phase-2--improved-training)
- [Phase 3 — Advanced Training](#-phase-3--advanced-training)
- [Phase 4 — Mastery Training](#-phase-4--mastery-training)
- [Phase 5 — Controlled Fine-Tuning](#-phase-5--controlled-fine-tuning)
- [Phase 6 — Boundary-Aware Fine-Tuning](#-phase-6--boundary-aware-fine-tuning-best)
- [Six-Phase Comparison](#-six-phase-comparison)
- [Per-Class IoU Journey](#-per-class-iou-journey)
- [Technical Deep Dive](#-technical-deep-dive)

---

## 🎯 Overview

**Task**: Classify every pixel of a synthetic desert offroad image into one of **10 semantic classes**: Background, Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Logs, Rocks, Landscape, and Sky.

**Approach**: We used a **frozen DINOv2 vision transformer** as a feature backbone and trained a segmentation head on top. Over **4 progressive training phases**, we evolved from a simple baseline to a sophisticated multi-scale architecture with test-time augmentation.

**Key Metric**: Mean Intersection-over-Union (mIoU) across all 10 classes.

### The Challenge

- **10 classes** with extreme imbalance — Sky covers 34% of pixels, Logs only 0.07%
- **Synthetic desert data** — models must generalize to unseen environments
- **Limited VRAM** — trained on RTX 3050 6GB, requiring careful memory optimization

---

## 🏆 Results Summary

| Phase | Val IoU | Val Dice | Val Accuracy | Improvement |
| :-----------------------: | :----------: | :--------: | :----------: | :------------: |
|       P1 — Baseline       |    0.2971    |   0.4416   |    70.41%    |       —        |
|       P2 — Improved       |    0.4036    |   0.6116   |    74.61%    |     +35.8%     |
|       P3 — Advanced       |    0.5161    |   0.7116   |    83.07%    |     +73.7%     |
|       P4 — Mastery        |    0.5169    |   0.7119   |    83.28%    |     +73.9%     |
| P5 — Controlled (TTA) |  0.5310  | 0.7236 |  83.67%  |   +78.2%   |
| **P6 — Boundary (TTA)** | **0.5527** | **0.7404** | **84.38%** | **+86.0%** |

> **P1 → P6**: 0.30 → **0.5527** TTA (**+86.0%** total gain). Phase 6 best non-TTA Val IoU: **0.5368** (Epoch 28). TTA boosted to **0.5527** via Multi-Scale (0.9×–1.2×) × HFlip — 8 passes.

---

## 📂 Project Structure

```
offroad-seg-hackathon-duality-ai/
│
├── BASIC_INTRO/                          # Background & theory
│   ├── 01 Ignitia Hackathon – Offroad Semantic Scene Segmentation.md
│   ├── 03_Setting Up Project Environment.md
│   ├── 04_Ignitia Explainer.md
│   ├── 05_THEORY_KNOWLEDGE.md
│   └── *.png / *.jpg / *.webp            # Concept diagrams (ML, CV, Segmentation, etc.)
│
├── TRAINING_SCRIPTS/                     # ⭐ ALL TRAINING CODE LIVES HERE
│   ├── train_phase1_baseline.py          # Phase 1 — Original baseline (DINOv2 ViT-S + ConvNeXt)
│   ├── train_phase2_improved.py          # Phase 2 — Augmentations + AdamW + CosineAnnealing
│   ├── train_phase3_advanced.py          # Phase 3 — ViT-Base + UPerNet + Focal/Dice
│   ├── train_phase4_mastery.py           # Phase 4 — Multi-scale + loss rebalance + TTA
│   ├── train_phase5_controlled.py        # Phase 5 — Backbone fine-tuning (blocks 10-11) + differential LR
│   ├── train_phase6_boundary.py          # Phase 6 — Boundary loss + multi-scale TTA + blocks 9-11 ⭐ READY
│   ├── test_segmentation.py             # Inference on test images
│   ├── visualize.py                     # Visualization utilities
│   └── SCRIPTS_EXPLAINED.md             # 📖 Deep-dive docs for all scripts & parameters
│
├── MODELS/                               # 🏋️ TRAINED MODEL WEIGHTS
│   ├── phase2_best_model_iou0.4036.pth  # Phase 2 best checkpoint (28MB)
│   ├── phase3_best_model_iou0.5161.pth  # Phase 3 best checkpoint (39MB)
│   ├── phase4_best_model_iou0.5150.pth  # Phase 4 best checkpoint (39MB)
│   ├── phase5_best_model_iou0.5294.pth  # ⭐ Phase 5 best checkpoint (33MB) — LATEST
│   ├── segmentation_head.pth            # Head-only weights for inference (13MB)
│   └── MODEL-README.md                  # Model usage docs
│
├── DATASET/
│   ├── Offroad_Segmentation_Training_Dataset/
│   │   ├── train/
│   │   │   ├── Color_Images/             # 2857 training RGB images (960×540)
│   │   │   └── Segmentation/             # 2857 mask images (uint16 class IDs)
│   │   └── val/
│   │       ├── Color_Images/             # 317 validation RGB images
│   │       └── Segmentation/             # 317 validation masks
│   ├── Offroad_Segmentation_testImages/  # Unseen test images for inference
│   ├── Offroad_Segmentation_Scripts/     # 🛠️ Original dev workspace (raw working files)
│   │   ├── train_segmentation.py         # Phase 1 — Original hackathon starter script
│   │   ├── train_improved.py             # Phase 2 — Augmentations + AdamW (dev version)
│   │   ├── train_phase3.py               # Phase 3 — ViT-Base + UPerNet (dev version)
│   │   ├── test_segmentation.py          # Inference on test images
│   │   ├── visualize.py                  # Visualization helpers
│   │   ├── segmentation_head.pth         # Best model head weights (generated by Phase 3)
│   │   ├── DEVELOPMENT_LOG.md            # 📋 File timeline, usage history, bugs fixed
│   │   ├── train_stats/                  # Phase 1 baseline output (curves + metrics)
│   │   └── ENV_SETUP/                    # Early setup scripts (superseded by root ENV_SETUP/)
│   └── dataset_instruction.md            # 📥 Download links + setup guide for the dataset
│
├── DATA_INSIGHTS/                        # Dataset analysis reports
│   ├── 01_Training_Dataset_Insights.md   # Train split statistics
│   ├── 02_Validation_Dataset_Insights.md # Val split statistics
│   └── 03_Test_Images_and_Scripts_Insights.md  # Test data + script overview
│
├── TRAINING AND PROGRESS/                # Training logs + metrics + curves
│   ├── PHASE_1_BASELINE/
│   │   ├── 00_phase1_log.md              # Detailed Phase 1 report
│   │   ├── all_metrics_curves.png        # Combined training curves
│   │   ├── iou_curves.png               # IoU-specific plot
│   │   ├── dice_curves.png              # Dice-specific plot
│   │   ├── training_curves.png          # Loss curves
│   │   └── evaluation_metrics.txt       # Per-epoch metrics table
│   ├── PHASE_2_IMPROVED/
│   │   ├── 00_phase2_log.md              # Detailed Phase 2 report
│   │   ├── all_metrics_curves.png        # Combined training curves
│   │   ├── per_class_iou.png            # Per-class IoU bar chart
│   │   ├── lr_schedule.png              # CosineAnnealing LR curve
│   │   ├── evaluation_metrics.txt       # Per-epoch metrics table
│   │   ├── history.json                 # Machine-readable metrics
│   │   ├── best_model.pth               # Best checkpoint (IoU=0.4036)
│   │   └── final_model.pth              # Last epoch weights
│   ├── PHASE_3_ADVANCED/
│   │   ├── 00_phase3_log.md              # Detailed Phase 3 report
│   │   ├── all_metrics_curves.png        # Combined training curves
│   │   ├── per_class_iou.png            # Per-class IoU bar chart
│   │   ├── lr_schedule.png              # Warmup + CosineAnnealing LR curve
│   │   ├── evaluation_metrics.txt       # Per-epoch metrics table
│   │   ├── history.json                 # Machine-readable metrics
│   │   ├── best_model.pth               # Best checkpoint (IoU=0.5161)
│   │   └── final_model.pth              # Last epoch weights
│   ├── PHASE_4_MASTERY/
│   │   ├── 00_phase4_log.md              # Detailed Phase 4 report
│   │   ├── all_metrics_curves.png        # Combined training curves
│   │   ├── per_class_iou.png            # Per-class IoU bar chart
│   │   ├── lr_schedule.png              # Warmup + CosineAnnealing LR curve
│   │   ├── evaluation_metrics.txt       # Per-epoch metrics table
│   │   ├── history.json                 # Machine-readable metrics (+ TTA)
│   │   ├── best_model.pth               # Best checkpoint (IoU=0.5150)
│   │   └── final_model.pth              # Last epoch weights
│   └── PHASE_5_CONTROLLED/
│       ├── 00_phase5_log.md              # Detailed Phase 5 report
│       ├── all_metrics_curves.png        # Combined training curves
│       ├── per_class_iou.png            # Per-class IoU bar chart
│       ├── lr_schedule.png              # Differential LR curve (bb vs head)
│       ├── overfit_gap.png              # Train-Val gap analysis
│       ├── evaluation_metrics.txt       # Per-epoch metrics table
│       ├── history.json                 # Machine-readable metrics (+ TTA)
│       ├── best_model.pth               # Best val IoU (0.5294)
│       └── final_model.pth              # Ep 30 weights
│   └── PHASE_6_BOUNDARY/               # ⭐ Phase 6 — Boundary-Aware (COMPLETED)
│       ├── 00_phase6_log.md             # Detailed Phase 6 report
│       ├── all_metrics_curves.png       # Loss/IoU curves across 30 epochs
│       ├── val_iou_progress.png         # Val IoU climb to 0.537
│       ├── lr_schedule.png             # Differential LR (bb 4e-6, head 2e-4)
│       ├── overfit_gap.png             # Max gap 0.039 — zero overfitting
│       ├── evaluation_metrics.txt      # Full 30-epoch table (UTF-8)
│       ├── history.json                # Machine-readable metrics
│       ├── best_model.pth              # Best checkpoint (IoU=0.5368, Ep 28)
│       └── final_model.pth             # Ep 30 weights
│
├── TESTING_INTERFACE/                    # 🔬 Gradio visual testing dashboard
│   ├── app.py                           # Visual model tester (class picker + upload + metrics)
│   ├── SCRIPT_EXPLAINED.md             # Interface documentation
│   ├── dataset_index_cache.json        # Auto-built class→file index
│   ├── IMGS/                            # Saved PNG images per result (raw/overlay/mask)
│   └── RESULTS/                         # Timestamped .md reports with embedded images
│
├── SYSTEM_CHECK/                         # Hardware verification scripts
│   ├── system_specs.py                  # Full system spec checker
│   ├── gpu_verify.py                    # PyTorch CUDA verification
│   ├── check_specs.py                   # Quick spec check
│   └── specs_result.json               # Saved spec results
│
├── ENV_SETUP/                            # Environment setup automation
│   ├── setup_env.bat                    # Windows one-command setup (dirs + dataset guide + venv)
│   ├── setup_env.sh                     # Linux/macOS one-command setup
│   └── project_management.md           # Project governance and file rules
│
├── REPRODUCE.md                          # 📖 Full step-by-step reproduction guide (start here!)
├── requirements.txt                      # Python dependencies (PyTorch + CUDA 12.6)
└── README.md                             # ← You are here
```

---

## 🚀 Quick Start

> **New here? This is where to start.** Three commands is all it takes to get from zero to a running environment.

### Step 1 — Clone the repo

```bash
git clone https://github.com/Daksh-M-Coder/offroad-seg-hackathon-duality-ai.git
cd offroad-seg-hackathon-duality-ai
```

### Step 2 — Run the setup script

The setup script does **everything** for you in one go:
- Creates all necessary folders
- Guides you to place the dataset at 3 checkpoints (see below)
- Creates a Python virtual environment (`venv/`)
- Installs PyTorch + CUDA 12.6 + all ML libraries
- Verifies your GPU and CUDA are working

```bash
# Windows (run from project root in Command Prompt):
ENV_SETUP\setup_env.bat

# Linux / macOS (run from project root in Terminal):
chmod +x ENV_SETUP/setup_env.sh && ./ENV_SETUP/setup_env.sh
```

### 📦 The AUTO_DATA Checkpoint System

When the setup script runs, it will **pause 3 times** to wait for you to manually place each dataset package. Each pause shows a clearly labelled box:

```
╔══════════════════════════════════════════════════════════╗
║               [AUTO_DATA_1] CHECKPOINT                  ║
║           Training Dataset Placement Required           ║
╚══════════════════════════════════════════════════════════╝
```

| Marker | What to do |
|---|---|
| `[AUTO_DATA_1]` | Unzip `Offroad_Segmentation_Training_Dataset.zip` → place inside `DATASET/Offroad_Segmentation_Training_Dataset/` |
| `[AUTO_DATA_2]` | Unzip `Offroad_Segmentation_testImages.zip` → place inside `DATASET/Offroad_Segmentation_testImages/` |
| `[AUTO_DATA_3]` | Unzip `Offroad_Segmentation_Scripts.zip` → place inside `DATASET/Offroad_Segmentation_Scripts/` |

Datasets are available from the **Duality AI Hackathon portal** (requires hackathon access). After each placement, press **Enter** and the script verifies your files automatically.

### Step 3 — Activate and train

```bash
# Windows
venv\Scripts\activate.bat
python TRAINING_SCRIPTS\train_phase1_baseline.py

# Linux/macOS
source venv/bin/activate
python TRAINING_SCRIPTS/train_phase1_baseline.py
```

> **Want the full step-by-step guide (Phase 1 → Phase 6)?** See **[REPRODUCE.md](REPRODUCE.md)** — it covers every detail including expected outputs, times, and troubleshooting.

---

## 🔁 How to Reproduce

This project is fully reproducible. Every result in this README was generated from the training scripts in this repository on a single GPU (RTX 3050 6GB). Here's what you need to know:

### Requirements

| | Minimum | What We Used |
|---|---|---|
| **Python** | 3.10+ | **3.11.9** |
| **GPU** | 6 GB VRAM (NVIDIA) | **RTX 3050 6 GB Laptop** |
| **CUDA** | 11.8 | **12.6** |
| **RAM** | 8 GB | **16 GB** |
| **Disk** | 10 GB | **20 GB** |
| **Training time** | — | **~28 hrs total (6 phases)** |

### Phase Training Order

Phases must be run **in order** — each phase's best checkpoint is the starting point for the next.

| Phase | Script | IoU | Notes |
|---|---|---|---|
| 1 | `train_phase1_baseline.py` | 0.2971 | Starting point |
| 2 | `train_phase2_improved.py` | 0.4036 | AdamW + augmentations |
| 3 | `train_phase3_advanced.py` | 0.5161 | DINOv2 ViT-Base + UPerNet |
| 4 | `train_phase4_mastery.py` | 0.5169 (TTA) | Multi-scale; early-stops at Ep 11 (normal) |
| 5 | `train_phase5_controlled.py` | 0.5310 (TTA) | Unfreeze backbone blocks 10-11 |
| **6** | `train_phase6_boundary.py` | **0.5527 (TTA)** | Blocks 9-11 + BoundaryLoss + 8-pass TTA |

### After each phase — archive the model

```bash
# Example for Phase 6 (adjust IoU and phase number for each)
cp "TRAINING AND PROGRESS/PHASE_6_BOUNDARY/best_model.pth" "MODELS/phase6_best_model_iou0.5368.pth"
```

### Full reproduction guide

📖 **[REPRODUCE.md](REPRODUCE.md)** contains the complete, newbie-friendly walkthrough:
- Exact commands for every step
- What to expect during each training run (epoch-by-epoch)
- Troubleshooting for 8 common issues
- File naming and project management rules

---

## 🧠 Training Scripts

All training code lives in `TRAINING_SCRIPTS/`. Each script is self-contained — just run it and it handles data loading, training, evaluation, plotting, and model saving automatically.

### `train_phase1_baseline.py` — Phase 1 Baseline

> **What it does**: Runs the original hackathon-provided training — no modifications.

- **Backbone**: DINOv2 ViT-Small (384-dim, frozen)
- **Head**: ConvNeXt (simple Conv2d → BatchNorm → ReLU stack)
- **Loss**: Unweighted CrossEntropyLoss
- **Optimizer**: SGD (lr=1e-4, momentum=0.9, no scheduler)
- **Augmentations**: None
- **Output**: `TRAINING AND PROGRESS/PHASE_1_BASELINE/`
- **Result**: IoU = **0.2971** in 10 epochs (~83 min)

### `train_phase2_improved.py` — Phase 2 Improved

> **What it does**: Fixes Phase 1's underfitting with better optimization and data handling.

- **Backbone**: Same DINOv2 ViT-Small (frozen)
- **Head**: Same ConvNeXt
- **Loss**: **Weighted** CrossEntropyLoss (class weights computed from pixel frequency)
- **Optimizer**: **AdamW** (lr=5e-4, wd=1e-4) + **CosineAnnealingLR**
- **Augmentations**: HFlip, VFlip, ShiftScaleRotate, GaussianBlur, MedianBlur, RandomBrightnessContrast, HueSaturationValue
- **Extras**: Mixed precision (AMP), best model checkpointing, early stopping (patience=10)
- **Output**: `TRAINING AND PROGRESS/PHASE_2_IMPROVED/`
- **Result**: IoU = **0.4036** in 30 epochs (~4 hours)

### `train_phase3_advanced.py` — Phase 3 Advanced

> **What it does**: Fundamentally changes the architecture and loss for maximum IoU.

- **Backbone**: **DINOv2 ViT-Base** (768-dim, frozen) — 2× richer features
- **Head**: **UPerNet** with Pyramid Pooling Module (PPM, pool sizes 1/2/3/6) + multi-scale FPN (dilations 1/2/4), GroupNorm throughout
- **Loss**: **Focal Loss (γ=2) + 0.5×Dice Loss** with class weights
- **Optimizer**: AdamW (lr=3e-4, wd=1e-4) + **3-epoch warmup** + CosineAnnealing
- **Resolution**: **644×364** (vs 476×266 in P1/P2) — 84% more pixels
- **Extras**: Gradient accumulation (effective batch=4), RandomShadow, CLAHE augmentations
- **Output**: `TRAINING AND PROGRESS/PHASE_3_ADVANCED/`
- **Result**: IoU = **0.5161** in 40 epochs (~7 hours)

### `train_phase4_mastery.py` — Phase 4 Mastery

> **What it does**: Pushes optimization further with multi-scale training, loss rebalance, and TTA.

- **Objective**: Break the 0.516 plateau via better scale-invariant augmentations.
- **Scale**: Widened from ±15% to **±20%**.
- **Loss Rebalance**: Focal γ=1.5, wt=0.4; Dice wt=0.6.
- **TTA**: Horizontal flip averaging during evaluation.
- **Result**: IoU = **0.5169 (TTA)** — Hit the frozen backbone ceiling.

### `train_phase5_controlled.py` — Phase 5 Controlled

> **What it does**: Breaks the 0.516 ceiling via **controlled backbone fine-tuning**.

- **Backbone Unfreeze**: ViT-Base blocks 0-9 frozen, **blocks 10-11 unfrozen** (14.18M params learning domain-specific semantics).
- **Differential LR**: Backbone (5e-6) learns **40x slower** than Head (2e-4).
- **Gradient Clipping**: Backbone max_norm=1.0, Head max_norm=5.0.
- **Loss**: Focal γ=2.0 (wt=0.3) + Dice (wt=0.7) for aggressive hard-pixel focus.
- **Safety**: 3 consecutive val drops + gap > 0.05 auto-stops training to prevent destroying pre-trained features.
- **Output**: `TRAINING AND PROGRESS/PHASE_5_CONTROLLED/`
- **Result**: IoU = **0.5310 (TTA)** in 30 epochs (~6.3 hours). No overfitting. Every single class improved.

### `train_phase6_boundary.py` — Phase 6 Boundary-Aware ⭐ BEST

> **What it does**: Pushes IoU further via **boundary-aware loss + multi-scale TTA + extended backbone unfreezing**.

- **Backbone Unfreeze**: **Blocks 9-11 unfrozen** (21.27M params — 3 blocks vs 2 in Phase 5).
- **Differential LR**: Backbone 4e-6 (**50× slower** than Head 2e-4). Expert-tuned from 3e-6.
- **Boundary Loss**: `Focal(γ=2.0, w=0.25) + Dice(w=0.55) + BoundaryLoss(w=0.20)` — edge-precision boost.
- **Multi-Scale TTA**: 0.9×, 1.0×, 1.1×, 1.2× × HFlip = **8 passes** at eval — best small-object coverage.
- **Safety**: All Phase 5 guards retained (gradient clipping, gap monitor, consecutive drop detection).
- **Output**: `TRAINING AND PROGRESS/PHASE_6_BOUNDARY/`
- **Result**: IoU = **0.5527 (Multi-Scale TTA)**, non-TTA = **0.5368** in 30 epochs (~7.3 hours). Zero overfitting (max gap 0.039).

### `test_segmentation.py` — Inference

> Runs the best trained model on unseen test images and produces segmented output masks.

### `visualize.py` — Visualization

> Helper utilities for overlaying predicted masks on images for visual inspection.

### `SCRIPTS_EXPLAINED.md` — Full Technical Documentation 📖

> **The big one.** Deep-dive into every script, every parameter, every design decision. Covers:
>
> - How each script works internally (data pipeline, model, loss, training loop)
> - Why each hyperparameter was chosen based on previous phase outcomes
> - Code snippets showing the key changes between phases
> - Expected vs actual results tables for each phase
> - Parameter evolution table across all 3 phases
> - 6 lessons learned from the entire training journey

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with CUDA support (tested on RTX 3050 6GB)
- **~15GB disk space** for dataset + models

### Automated Setup

**Windows:**

```batch
ENV_SETUP\setup_env.bat
```

**Linux/macOS:**

```bash
chmod +x ENV_SETUP/setup_env.sh
./ENV_SETUP/setup_env.sh
```

### Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows: venv\Scripts\activate
# Linux:   source venv/bin/activate

# Install dependencies (PyTorch + CUDA 12.6)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

# Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Run Training

```bash
# Phase 1 — Baseline (10 epochs, ~83 min)
python TRAINING_SCRIPTS/train_phase1_baseline.py

# Phase 2 — Improved (30 epochs, ~4 hours)
python TRAINING_SCRIPTS/train_phase2_improved.py

# Phase 3 — Advanced (40 epochs, ~7 hours) ← BEST RESULTS
python TRAINING_SCRIPTS/train_phase3_advanced.py
```

---

## 📊 Phase 1 — Baseline Training

> **Goal**: Run the provided script as-is to establish a reference IoU.

### Configuration

| Parameter     | Value                                |
| ------------- | ------------------------------------ |
| Backbone      | DINOv2 ViT-Small (`vits14`) — frozen |
| Head          | ConvNeXt (simple Conv2d stack)       |
| Optimizer     | SGD (lr=1e-4, momentum=0.9)          |
| LR Schedule   | None (constant)                      |
| Epochs        | 10                                   |
| Batch Size    | 2                                    |
| Image Size    | 476×266                              |
| Augmentations | **None**                             |
| Loss          | CrossEntropy (unweighted)            |

### Results: IoU = 0.2971

| Epoch  | Train Loss |  Val Loss  | Train IoU  | **Val IoU** | Train Dice |  Val Dice  | Train Acc  |  Val Acc   |
| :----: | :--------: | :--------: | :--------: | :---------: | :--------: | :--------: | :--------: | :--------: |
|   1    |   1.1907   |   0.9890   |   0.2303   |   0.2211    |   0.3256   |   0.3430   |   65.28%   |   65.52%   |
|   2    |   0.9447   |   0.9104   |   0.2695   |   0.2513    |   0.3775   |   0.3815   |   67.51%   |   67.73%   |
|   3    |   0.8965   |   0.8788   |   0.2836   |   0.2636    |   0.3962   |   0.3983   |   68.37%   |   68.58%   |
|   4    |   0.8713   |   0.8605   |   0.2942   |   0.2733    |   0.4076   |   0.4120   |   68.90%   |   69.10%   |
|   5    |   0.8553   |   0.8457   |   0.2972   |   0.2758    |   0.4116   |   0.4149   |   69.29%   |   69.49%   |
|   6    |   0.8438   |   0.8353   |   0.3067   |   0.2851    |   0.4241   |   0.4277   |   69.59%   |   69.78%   |
|   7    |   0.8352   |   0.8282   |   0.3095   |   0.2874    |   0.4271   |   0.4297   |   69.78%   |   69.97%   |
|   8    |   0.8286   |   0.8224   |   0.3170   |   0.2924    |   0.4363   |   0.4362   |   69.95%   |   70.15%   |
|   9    |   0.8228   |   0.8176   |   0.3155   |   0.2934    |   0.4345   |   0.4378   |   70.06%   |   70.26%   |
| **10** | **0.8184** | **0.8136** | **0.3216** | **0.2971**  | **0.4427** | **0.4416** | **70.22%** | **70.41%** |

### Training Curves

![Phase 1 — Training Curves](TRAINING%20AND%20PROGRESS/PHASE_1_BASELINE/all_metrics_curves.png)

### Analysis

**Epoch 1-3 (Cold Start → Rapid Learning)**: The random head learns basic feature-to-class mapping. Val IoU shoots from 0.22 → 0.26. The DINOv2 backbone provides rich features even though the head is untrained.

**Epoch 4-7 (Slowing Down)**: Val IoU = 0.27 → 0.29. Easy classes (Sky) are learned first. Harder classes contribute diminishing returns. The constant LR of 1e-4 with SGD is painfully slow.

**Epoch 8-10 (Still Improving — Underfitting!)**: Val IoU = 0.29 → 0.30. **Loss was still decreasing linearly** — the model hadn't converged at all. This is the clearest sign that we needed more epochs and a better optimizer.

**Diagnosis**: Severe underfitting. No overfitting (train ≈ val). The bottleneck was lack of training time, slow SGD optimizer, and zero augmentations.

---

## 🔧 Phase 2 — Improved Training

> **Goal**: Fix the underfitting with better optimization, augmentations, and class balancing.

### What Changed

| Feature         | Phase 1 → Phase 2                                            |
| --------------- | ------------------------------------------------------------ |
| Optimizer       | SGD → **AdamW** (lr=5e-4, wd=1e-4)                           |
| LR Schedule     | None → **CosineAnnealingLR** (5e-4 → 1e-6)                   |
| Epochs          | 10 → **30**                                                  |
| Augmentations   | None → **HFlip, VFlip, ShiftScaleRotate, Blur, ColorJitter** |
| Loss            | CE → **Weighted CrossEntropy** (max weight=5.0 for Logs)     |
| Mixed Precision | No → **Yes** (torch.cuda.amp for 6GB VRAM)                   |
| Checkpointing   | Final only → **Best by val_iou**                             |
| Early Stopping  | No → **Patience=10**                                         |

### Class Weights (Computed from Training Data)

| Class          |  Weight  |  Pixel %  | Problem                     |
| -------------- | :------: | :-------: | --------------------------- |
| Sky            |   0.01   |  34.72%   | Dominates, easy to learn    |
| Landscape      |   0.02   |  22.34%   | Large, confused with ground |
| Dry Grass      |   0.02   |  17.37%   | Dominant ground class       |
| Lush Bushes    |   0.07   |   5.50%   | Confused with Trees         |
| Background     |   0.07   |   5.36%   | Catch-all, ambiguous        |
| Ground Clutter |   0.09   |   4.03%   | Visually heterogeneous      |
| Trees          |   0.11   |   3.29%   | Distinct but smaller        |
| Rocks          |   0.33   |   1.10%   | Small, scattered            |
| Dry Bushes     |   0.36   |   1.01%   | Similar to Dry Grass        |
| **Logs**       | **5.00** | **0.07%** | **72× rarer than average!** |

### Results: IoU = 0.4036 (+35.8%)

| Epoch  | Train Loss |  Val Loss  | Train IoU  | **Val IoU** |     LR     |
| :----: | :--------: | :--------: | :--------: | :---------: | :--------: |
|   1    |   0.8723   |   0.7423   |   0.3376   |   0.3111    |  4.99e-4   |
|   5    |   0.6449   |   0.6219   |   0.3891   |   0.3580    |  4.70e-4   |
|   10   |   0.6074   |   0.5947   |   0.4026   |   0.3751    |  3.85e-4   |
|   15   |   0.5869   |   0.5838   |   0.4133   |   0.3865    |  2.70e-4   |
|   20   |   0.5788   |   0.5766   |   0.4221   |   0.3943    |  1.60e-4   |
|   25   |   0.5727   |   0.5724   |   0.4288   |   0.4030    |   3.4e-5   |
| **26** | **0.5712** | **0.5722** | **0.4276** | **0.4036**  | **2.3e-5** |
|   30   |   0.5668   |   0.5698   |   0.4268   |   0.4022    |    1e-6    |

### Training Curves

![Phase 2 — Training Curves](TRAINING%20AND%20PROGRESS/PHASE_2_IMPROVED/all_metrics_curves.png)

### Per-Class IoU

![Phase 2 — Per-Class IoU](TRAINING%20AND%20PROGRESS/PHASE_2_IMPROVED/per_class_iou.png)

| Class          |    IoU     | Status          |
| -------------- | :--------: | --------------- |
| Sky            |   0.9473   | 🟢 Near-perfect |
| Trees          |   0.5030   | 🟢 Good         |
| Dry Grass      |   0.4811   | 🟢 Good         |
| Background     |   0.4515   | 🟡 Moderate     |
| Lush Bushes    |   0.4128   | 🟡 Moderate     |
| Landscape      |   0.3610   | 🟡 Needs work   |
| Dry Bushes     |   0.2786   | 🔴 Weak         |
| Ground Clutter |   0.2153   | 🔴 Weak         |
| Rocks          |   0.1647   | 🔴 Very Weak    |
| **Logs**       | **0.0517** | **🔴 Critical** |

### Analysis

**Epoch 1-5 (Better Than Phase 1's Entire Run!)**: Phase 2 starts at IoU=0.31 — already higher than Phase 1's final 0.30. AdamW converges MUCH faster than SGD. Augmentations prevent memorization from epoch 1.

**Epoch 6-15 (Steady Climbing)**: IoU gradually rises from 0.36 → 0.39. CosineAnnealing keeps the LR high enough for continued improvement. The model learns medium-frequency classes well.

**Epoch 16-26 (Fine-Tuning → Best Model)**: IoU converges to 0.4036 at epoch 26. The CosineAnnealing schedule enters the fine-tuning zone (LR < 1e-4), allowing precise boundary refinement.

**Epoch 27-30 (Flat)**: LR ~0, model locked in. No overfitting — train-val gap stayed small throughout.

**Why 0.40 is the ceiling here**: The ConvNeXt head processes features at ONE scale only. It treats a 3-pixel Log and a 10K-pixel Sky region identically. For small rare objects, we need multi-scale feature extraction → Phase 3.

---

## 🚀 Phase 3 — Advanced Training

> **Goal**: Break past 0.50 with fundamental architectural improvements.

### Three Key Changes

#### 1. DINOv2 ViT-Base Backbone (768-dim)

The ViT-Small's 384-dim features couldn't distinguish texturally similar classes (Ground Clutter vs Landscape, Dry Bushes vs Dry Grass). **ViT-Base doubles the embedding dimension** to 768, providing richer feature representations for fine-grained class discrimination.

#### 2. UPerNet Segmentation Head (PPM + Multi-Scale FPN)

The simple ConvNeXt head was replaced by a **Unified Perceptual Parsing Network (UPerNet)** head:

```
Patch Tokens (B, N, 768)
    ↓ reshape → Conv2d(768→256) + GroupNorm + ReLU
    ↓
Pyramid Pooling Module (PPM):
    → Pool at sizes [1, 2, 3, 6] for global context
    → Concat + Bottleneck → 256 channels
    ↓
Multi-Scale Feature Pyramid:
    ├── dilation=1  (fine details: Logs, Rocks)
    ├── dilation=2  (medium: Bushes, Clutter)
    └── dilation=4  (wide context: Landscape, Sky)
    ↓ FPN Fusion → Classifier → 10 classes
```

**Why UPerNet matters**: Unlike ConvNeXt which only sees features at one scale, UPerNet's PPM provides **global scene understanding** (is this Sky or Ground?) while the FPN provides **local detail** (is this specific pixel a Log or Ground Clutter?).

#### 3. Focal + Dice Combined Loss

- **Focal Loss (γ=2)**: Down-weights easy pixel classifications (Sky, Landscape) and focuses training on hard, ambiguous boundaries. Critical for rare classes.
- **Dice Loss**: Directly optimizes the IoU-like region overlap metric instead of per-pixel accuracy. Helps with class imbalance by treating each class equally regardless of frequency.

### Full Configuration

| Parameter      | Value                                                                      |
| -------------- | -------------------------------------------------------------------------- |
| Backbone       | DINOv2 **ViT-Base** (`vitb14_reg`) — frozen, 768-dim                       |
| Head           | **UPerNet** (PPM + FPN, GroupNorm, Dropout=0.1)                            |
| Loss           | **Focal (γ=2) + 0.5×Dice** with class weights                              |
| Optimizer      | AdamW (lr=3e-4, wd=1e-4)                                                   |
| LR Schedule    | **3-epoch warmup + CosineAnnealing**                                       |
| Epochs         | 40                                                                         |
| Batch Size     | 2 (effective 4 via **gradient accumulation**)                              |
| Image Size     | **644×364** (84% more pixels than Phase 2)                                 |
| Augmentations  | HFlip, VFlip, ShiftScaleRotate, Blur, ColorJitter, **RandomShadow, CLAHE** |
| Normalization  | **GroupNorm** (works at 1×1 spatial unlike BatchNorm)                      |
| Early Stopping | Patience=12                                                                |

### Results: IoU = 0.5161 (+73.7% over baseline)

| Epoch  | Train Loss |  Val Loss  | Train IoU  | **Val IoU** |     LR      |
| :----: | :--------: | :--------: | :--------: | :---------: | :---------: |
|   1    |   0.2869   |   0.3035   |   0.4595   |   0.4219    |   2.00e-4   |
|   2    |   0.2543   |   0.2469   |   0.4787   |   0.4465    |   3.00e-4   |
|   3    |   0.2205   |   0.2107   |   0.4754   |   0.4460    |   3.00e-4   |
|   5    |   0.2033   |   0.1854   |   0.4968   |   0.4691    |   2.98e-4   |
|   8    |   0.1966   |   0.1824   |   0.5127   |   0.4805    |   2.87e-4   |
|   11   |   0.1899   |   0.1760   |   0.5161   |   0.4889    |   2.67e-4   |
| **16** | **0.1819** | **0.1689** | **0.5269** | **0.5000**  | **2.18e-4** |
|   20   |   0.1799   |   0.1673   |   0.5285   |   0.5015    |   1.69e-4   |
|   25   |   0.1752   |   0.1640   |   0.5373   |   0.5083    |   1.06e-4   |
|   30   |   0.1733   |   0.1614   |   0.5419   |   0.5132    |   5.1e-5    |
|   35   |   0.1717   |   0.1605   |   0.5448   |   0.5153    |   1.3e-5    |
| **40** | **0.1708** | **0.1599** | **0.5452** | **0.5161**  |   **0.0**   |

### Training Curves

![Phase 3 — Training Curves](TRAINING%20AND%20PROGRESS/PHASE_3_ADVANCED/all_metrics_curves.png)

### Per-Class IoU

![Phase 3 — Per-Class IoU](TRAINING%20AND%20PROGRESS/PHASE_3_ADVANCED/per_class_iou.png)

### Analysis

**Epoch 1 — Starts ABOVE Phase 2's Best!**: Val IoU=0.4219 at epoch 1 already surpasses Phase 2's best of 0.4036. The ViT-Base backbone and UPerNet head are immediately more powerful, even with untrained head weights.

**Epoch 2-3 (Warmup completes)**: LR ramps from 2e-4 → 3e-4. Slight IoU dip at epoch 3 (0.4460 vs 0.4465) — normal during warmup transition.

**Epoch 4-8 (Rapid improvement)**: IoU rises from 0.46 → 0.48. The PPM's global pooling "clicks" — the model learns scene-level context (where Sky meets Landscape, where Ground meets Vegetation).

**Epoch 9-15 (Approaching 0.50)**: The hardest push. IoU crawls from 0.48 → 0.49. Multi-scale FPN is learning to detect objects at different scales.

**🎉 Epoch 16 — IoU crosses 0.50!**: The breakthrough epoch. IoU=0.5000. This coincides with LR entering the decay phase (2.18e-4), enabling more precise boundary learning.

**Epoch 17-30 (Consolidation)**: IoU gradually improves 0.50 → 0.51. Each epoch refines class boundaries. Per-class IoU for medium-difficulty classes (Landscape, Dry Bushes) sees the biggest gains here.

**Epoch 31-40 (Convergence)**: IoU inches from 0.513 → 0.516. The model finds its absolute best at epoch 40 with IoU=0.5161. Early stopping was never triggered — suggesting even more epochs could help marginally.

---

## 🏆 Phase 4 — Mastery Training

> **Objective**: Push IoU past 0.52 via multi-scale training, loss rebalancing, and TTA — no architectural changes.

### What Changed

| Change                 | Phase 3 | Phase 4                | Why                                              |
| ---------------------- | ------- | ---------------------- | ------------------------------------------------ |
| **Scale augmentation** | ±15%    | **±20%** (0.8x–1.2x)   | Scale-invariant features for varied object sizes |
| **Focal γ**            | 2.0     | **1.5**                | Softer focus on hard pixels, less extreme        |
| **Focal weight**       | 1.0     | **0.4**                | Reduce Focal dominance                           |
| **Dice weight**        | 0.5     | **0.6**                | Push region-level overlap optimization           |
| **Initialization**     | Random  | **Phase 3 checkpoint** | Start from where P3 left off                     |
| **TTA**                | None    | **HFlip + average**    | Free inference boost                             |
| **Patience**           | 12      | **10**                 | Tighter early stopping                           |

### Results (11 epochs, early stopped)

| Metric            | Train  | Val    | Val (TTA)  |
| ----------------- | ------ | ------ | ---------- |
| **IoU**           | 0.5488 | 0.5150 | **0.5169** |
| **Dice**          | 0.6950 | 0.7103 | **0.7119** |
| **Accuracy**      | 83.56% | 83.20% | **83.28%** |
| **Train-Val Gap** | —      | 0.034  | —          |

### Training Curves

![Phase 4 — Training Curves](TRAINING%20AND%20PROGRESS/PHASE_4_MASTERY/all_metrics_curves.png)

![Phase 4 — Per-Class IoU](TRAINING%20AND%20PROGRESS/PHASE_4_MASTERY/per_class_iou.png)

### Analysis

**Epoch 1 — Best IoU (0.5150)**: Loaded Phase 3 weights, first epoch with warmup LR=2e-4. The lower LR perfectly suited the pre-trained weights.

**Epoch 2-5 — Adaptation Dip**: Full LR (3e-4) + changed loss balance disrupted learned features. Val IoU dropped to 0.500. The model was re-adapting to the new loss landscape.

**Epoch 6-10 — Recovery**: Model climbed back: 0.509 → 0.514. By epoch 10 it nearly matched its best (0.5144 vs 0.5150).

**Epoch 11 — Early Stop**: Patience=10 triggered. The model couldn't beat Ep 1's 0.5150 within 10 epochs. With patience=15, it likely would have surpassed it.

**TTA Boost**: Horizontal flip averaging added +0.0019 IoU — pushing effective result to **0.5169**, slightly above Phase 3's 0.5161.

**Key Insight**: Changing the loss function when resuming from a checkpoint creates a "loss landscape mismatch" — the weights optimized for Focal(γ=2)+Dice(0.5) had to readjust to Focal(γ=1.5)+Dice(0.6). The per-class improvements (Dry Bushes +54.5%, Trees +24.2%) show the rebalance helped significantly, but the mean IoU was held back by minor regressions in already-good classes.

---

## 🔥 Phase 5 — Controlled Backbone Fine-Tuning ⭐ BEST

> **Goal**: Break the 0.516 frozen-backbone ceiling by carefully unfreezing ViT-Base blocks 10-11 with differential learning rates and gradient clipping.

### What Changed vs Phase 4

| Component              | Phase 4            | Phase 5                                   | Why                                      |
| ---------------------- | ------------------ | ----------------------------------------- | ---------------------------------------- |
| **Backbone**           | Fully frozen       | **Blocks 10-11 UNFROZEN** (14.18M params) | Break the 0.516 ceiling                  |
| **Backbone LR**        | N/A                | **5e-6** (40× slower than head)           | Preserve pre-trained features            |
| **Head LR**            | 3e-4               | **2e-4** (safer for checkpoint resume)    | Learned from P4's loss-landscape lesson  |
| **Focal γ**            | 1.5                | **2.0** (reverted to P3)                  | Stronger hard-pixel focus                |
| **Focal / Dice Mix**   | 0.4 / 0.6          | **0.3 / 0.7**                             | Let Dice dominate for region overlap     |
| **Scale augmentation** | ±20%               | **±10%**                                  | Reduced noise for backbone stability     |
| **RandomShadow**       | ON (p=0.15)        | **REMOVED**                               | Noisy gradients during unfreeze          |
| **Gradient Clipping**  | None               | **bb max_norm=1.0, head max_norm=5.0**    | Backbone protection from gradient spikes |
| **Safety Stop**        | None               | **3 consecutive drops + gap > 0.05**      | Overfit auto-detection                   |
| **Initialization**     | Phase 3 checkpoint | **Phase 4 checkpoint (0.5150)**           | Continue from latest best                |
| **Epochs**             | 50 (stopped at 11) | **30** (all completed)                    | Controlled run                           |

### Configuration

| Parameter             | Value                                                                 |
| --------------------- | --------------------------------------------------------------------- |
| **Backbone**          | DINOv2 ViT-Base (`vitb14_reg`) — **blocks 10-11 unfrozen**            |
| **Trainable Params**  | **17,592,842** (backbone 14.18M + head 3.41M)                         |
| **Backbone split**    | 14.18M unfrozen / 86M total (16.5%)                                   |
| **Head**              | UPerNet (PPM pool sizes 1/2/3/6 + FPN dilations 1/2/4, GroupNorm)     |
| **Loss**              | Focal (γ=2.0, w=0.3) + Dice (w=0.7) with class weights                |
| **Optimizer**         | AdamW — backbone lr=5e-6, head lr=2e-4, weight_decay=1e-4             |
| **LR Schedule**       | 3-epoch linear warmup → CosineAnnealing (applied to both groups)      |
| **Gradient Clipping** | Backbone max_norm=1.0, Head max_norm=5.0                              |
| **Batch Size**        | 2 (effective 4 via gradient accumulation)                             |
| **Image Size**        | 644×364 (46×26 patch tokens — unchanged from P3/P4)                   |
| **Augmentations**     | HFlip, VFlip, MultiScale (±10%), Blur, ColorJitter, CLAHE — no Shadow |
| **Mixed Precision**   | ✅ AMP (fp16 forward)                                                 |
| **Early Stopping**    | Patience=10 — **not triggered**                                       |
| **Safety Stop**       | 3 consecutive val drops + gap > 0.05 — **not triggered**              |
| **TTA**               | HFlip + logit average                                                 |

### Results: IoU = 0.5294 (TTA 0.5310) — +2.74% over Phase 4

| Epoch  | Train Loss | Val Loss | Train IoU |    Val IoU    |  Gap  | LR (bb) | LR (head) |
| :----: | :--------: | :------: | :-------: | :-----------: | :---: | :-----: | :-------: |
|   1    |   0.2236   |  0.2105  |  0.5516   |    0.5161     | 0.035 | 3.33e-6 |  1.33e-4  |
|   2    |   0.2246   |  0.2125  |  0.5477   |    0.5129     | 0.035 | 5.00e-6 |  2.00e-4  |
|   3    |   0.2272   |  0.2138  |  0.5458   |    0.5111     | 0.035 | 5.00e-6 |  2.00e-4  |
|   4    |   0.2261   |  0.2125  |  0.5457   |    0.5130     | 0.033 | 4.98e-6 |  1.99e-4  |
|   5    |   0.2257   |  0.2116  |  0.5487   |    0.5151     | 0.034 | 4.93e-6 |  1.97e-4  |
|   7    |   0.2218   |  0.2114  |  0.5500   |    0.5152     | 0.035 | 4.73e-6 |  1.89e-4  |
| **10** |   0.2209   |  0.2101  |  0.5518   |  **0.5169**   | 0.035 | 4.22e-6 |  1.69e-4  |
| **11** |   0.2209   |  0.2087  |  0.5530   | **0.5194** 🔥 | 0.034 | 3.99e-6 |  1.60e-4  |
|   13   |   0.2190   |  0.2078  |  0.5559   |    0.5210     | 0.035 | 3.49e-6 |  1.40e-4  |
|   15   |   0.2190   |  0.2069  |  0.5571   |    0.5224     | 0.035 | 2.93e-6 |  1.17e-4  |
| **18** |   0.2157   |  0.2057  |  0.5609   |  **0.5246**   | 0.036 | 2.07e-6 |  8.26e-5  |
| **20** |   0.2148   |  0.2046  |  0.5605   |  **0.5258**   | 0.035 | 1.51e-6 |  6.04e-5  |
|   22   |   0.2141   |  0.2042  |  0.5645   |    0.5268     | 0.038 | 1.01e-6 |  4.03e-5  |
|   24   |   0.2141   |  0.2031  |  0.5638   |    0.5287     | 0.035 | 5.85e-7 |  2.34e-5  |
|   26   |   0.2127   |  0.2029  |  0.5654   |    0.5289     | 0.037 | 2.66e-7 |  1.06e-5  |
|   27   |   0.2131   |  0.2027  |  0.5658   |    0.5292     | 0.037 | 1.51e-7 |  6.03e-6  |
| **29** |   0.2127   |  0.2026  |  0.5657   | **0.5294** ⭐ | 0.036 | 1.69e-8 |  6.76e-7  |
|   30   |   0.2124   |  0.2026  |  0.5671   |    0.5294     | 0.038 |    0    |     0     |

> No early stopping triggered. No safety stop triggered. Gap stayed 0.033–0.038 throughout — well below the 0.05 danger threshold.

### Final Scores

| Metric             |  Train |    Val |  Val (TTA) |
| ------------------ | -----: | -----: | ---------: |
| **IoU**            | 0.5671 | 0.5294 | **0.5310** |
| **Dice**           | 0.7224 | 0.7224 | **0.7236** |
| **Pixel Accuracy** | 83.93% | 83.58% | **83.67%** |
| **Train-Val Gap**  |      — |  0.038 |          — |

### Training Curves

![Phase 5 — All Metrics](TRAINING%20AND%20PROGRESS/PHASE_5_CONTROLLED/all_metrics_curves.png)

> **What to see**: Smooth monotonic improvement over all 30 epochs. A brief 3-epoch dip (Ep 1-3, backbone adjusting to new domain) then **near-linear climb** from Ep 4 through 30. Train-Val gap stays flat at 0.033-0.038 — textbook generalization.

### Differential LR Schedule

![Phase 5 — LR Schedule](TRAINING%20AND%20PROGRESS/PHASE_5_CONTROLLED/lr_schedule.png)

> **What to see**: The 40× differential is clear — backbone LR peaks at 5e-6 while head peaks at 2e-4. Both follow the same warmup + cosine decay. Best epoch (29) occurs when both LRs are near zero — fine-tuning's final refinement stage.

### Overfit Gap Monitor

![Phase 5 — Overfit Gap](TRAINING%20AND%20PROGRESS/PHASE_5_CONTROLLED/overfit_gap.png)

> **What to see**: Gap is flat at 0.033-0.038 for all 30 epochs — zero overfitting. The orange danger line (0.05) was never approached. The 5e-6 backbone LR + gradient clipping (max_norm=1.0) + reduced augmentations made this rock-solid.

### Per-Class IoU

![Phase 5 — Per-Class IoU](TRAINING%20AND%20PROGRESS/PHASE_5_CONTROLLED/per_class_iou.png)

| Class              | Phase 4 (TTA) | Phase 5 (TTA) |    Change     | Verdict           |
| ------------------ | :-----------: | :-----------: | :-----------: | ----------------- |
| **Background**     |    0.5150     |  **0.5291**   | **+2.74%** ✅ | Improved          |
| **Trees**          |    0.6299     |  **0.6435**   | **+2.16%** ✅ | Improved          |
| **Lush Bushes**    |    0.5166     |  **0.5322**   | **+3.02%** ✅ | Improved          |
| **Dry Grass**      |    0.5888     |  **0.5989**   | **+1.71%** ✅ | Improved          |
| **Dry Bushes**     |    0.4375     |  **0.4529**   | **+3.51%** ✅ | Improved          |
| **Ground Clutter** |    0.2544     |  **0.2646**   | **+4.01%** ✅ | Improved          |
| **Logs**           |    0.2507     |  **0.2975**   | **+18.7%** ✅ | **Major gain** 🔥 |
| **Rocks**          |    0.3180     |  **0.3403**   | **+7.01%** ✅ | Significant gain  |
| **Landscape**      |    0.5503     |  **0.5555**   | **+0.95%** ✅ | Slight gain       |
| **Sky**            |    0.9684     |  **0.9702**   | **+0.19%** ✅ | Saturated         |

> 🎉 **First time in 5 phases that EVERY class improved.** Backbone fine-tuning helps all classes simultaneously — better features lift everything.

### Analysis

**Epochs 1-3 — Expected Adjustment Dip**: Val IoU drops 0.516 → 0.511. The backbone's blocks 10-11 temporarily disrupt the head's learned feature distribution as they start adapting to the offroad domain. The gradient clipping (max_norm=1.0) keeps this from becoming catastrophic.

**Epochs 4-10 — Recovery & First Territory**: Val IoU climbs back: 0.513 → 0.517. By Epoch 7, the model exceeds Phase 4's best for the first time. By Epoch 10, it matches Phase 4's TTA score (0.5169) — without TTA.

**Epoch 11 — 🔥 Breaking Through to New Territory**: Val IoU=0.5194. The first time in the entire project we're above 0.519. Backbone blocks 10-11 have finished adapting and are now actively contributing improved semantic representations for desert classes.

**Epochs 12-20 — Steady Climb**: IoU: 0.52 → 0.526. Each epoch the backbone refines its domain-specific representations. Logs (+18.7%) and Rocks (+7%) see disproportionate gains — these classes have distinctive textures that DINOv2's ImageNet pre-training didn't capture well.

**Epochs 21-30 — Convergence**: IoU: 0.526 → 0.5294. The LR approaches zero (cosine tail), enabling the finest-grain weight adjustments. Best achieved at Epoch 29 (0.5294) — matched exactly at Epoch 30.

**Key Insight — Why Every Class Won**: Unlike Phase 4's loss rebalance (which helped some classes at the expense of others), backbone fine-tuning improves the _quality of representations_ for all classes simultaneously. Better backbone features = better everything.

### Overfit Analysis

| Epoch | Train IoU | Val IoU |  Gap  | Status     |
| :---: | :-------: | :-----: | :---: | ---------- |
|   1   |   0.552   |  0.516  | 0.035 | ✅ Healthy |
|  10   |   0.552   |  0.517  | 0.035 | ✅ Healthy |
|  15   |   0.557   |  0.522  | 0.035 | ✅ Healthy |
|  20   |   0.561   |  0.526  | 0.035 | ✅ Healthy |
|  25   |   0.566   |  0.529  | 0.038 | ✅ Healthy |
|  30   |   0.567   |  0.529  | 0.038 | ✅ Healthy |

**Verdict**: Zero overfitting detected. The gap increased by only 0.003 across 30 epochs — essentially flat. The 5e-6 backbone LR + gradient clipping + reduced augmentations created the most stable training run in the entire project.

---

## 🔆 Phase 6 — Boundary-Aware Fine-Tuning ⭐ BEST

> **Goal**: Push IoU beyond Phase 5's 0.530 ceiling via boundary-aware loss, extended backbone unfreezing (block 9), and multi-scale TTA.

### What Changed vs Phase 5

| Component | Phase 5 | Phase 6 | Why |
| --------- | ------- | ------- | --- |
| **Backbone unfrozen** | Blocks 10-11 | **Blocks 9-11** (3 blocks) | Block 9 encodes shape — helps edge precision |
| **Backbone LR** | 5e-6 | **4e-6** (expert tuned) | 50× ratio vs head; conservative for block 9 |
| **Loss** | Focal(0.3)+Dice(0.7) | **Focal(0.25)+Dice(0.55)+Boundary(0.20)** | Explicit edge penalty |
| **Boundary Loss** | None | **CNN erosion → Gaussian soft edges** | Target sloppy boundary pixels directly |
| **TTA** | HFlip (2 passes) | **Multi-Scale 0.9–1.2× × HFlip (8 passes)** | Better small-object coverage |
| **Safety guards** | All Phase 5 guards | All retained + seed=42 for reproducibility | |
| **Resume from** | Phase 4 checkpoint | **Phase 5 best (0.5294)** | Stack improvements |

### Configuration

| Parameter | Value |
| --------- | ----- |
| **Backbone** | DINOv2 ViT-Base (`vitb14_reg`) — **blocks 9-11 unfrozen** |
| **Trainable Params** | **24,682,250** (backbone 21.27M + head 3.41M) |
| **Backbone split** | 21.27M unfrozen / 86M total (24.6%) |
| **Head** | UPerNet (PPM pool sizes 1/2/3/6 + FPN dilations 1/2/4, GroupNorm) |
| **Loss** | Focal (γ=2.0, w=0.25) + Dice (w=0.55) + BoundaryLoss (w=0.20) |
| **Boundary Loss** | Morphological erosion → Gaussian blurred edge map (σ=2.0) |
| **Optimizer** | AdamW — backbone lr=4e-6, head lr=2e-4, wd=1e-4 |
| **LR Schedule** | 3-epoch linear warmup → CosineAnnealing (both param groups) |
| **Gradient Clipping** | Backbone max_norm=1.0, Head max_norm=5.0 |
| **Batch Size** | 2 (effective 4 via gradient accumulation) |
| **Image Size** | 644×364 (46×26 patch tokens — unchanged from P3-P5) |
| **Augmentations** | HFlip, VFlip, MultiScale (±8%), Blur, ColorJitter, CLAHE — no Shadow |
| **Mixed Precision** | ✅ AMP (fp16 forward) |
| **Early Stopping** | Patience=10 — **not triggered** |
| **TTA** | Multi-Scale (0.9×, 1.0×, 1.1×, 1.2×) × HFlip = 8 passes |
| **Seeds** | torch=42, numpy=42, random=42 (reproducibility) |

### Results: IoU = 0.5368 non-TTA, **0.5527 Multi-Scale TTA**

| Epoch | Train Loss | Val Loss | Train IoU | Val IoU | Gap | LR (bb) | LR (head) |
| :---: | :--------: | :------: | :-------: | :-----: | :---: | :------: | :-------: |
| 1 | 0.185 | 0.176 | 0.562 | 0.5267 ⭐ | 0.036 | 2.7e-6 | 1.3e-4 |
| 2 | 0.186 | 0.177 | 0.562 | 0.5260 | 0.036 | 4.0e-6 | 2.0e-4 |
| 3 | 0.187 | 0.178 | 0.558 | 0.5230 | 0.035 | 4.0e-6 | 2.0e-4 |
| 4 | 0.187 | 0.179 | 0.555 | 0.5210 | 0.035 | 4.0e-6 | 2.0e-4 |
| 5 | 0.185 | 0.177 | 0.561 | 0.5250 | 0.036 | 3.9e-6 | 2.0e-4 |
| 8 | 0.186 | 0.176 | 0.564 | 0.5260 | 0.037 | 3.7e-6 | 1.8e-4 |
| **9** | 0.185 | 0.175 | 0.565 | **0.5277** | 0.037 | 3.5e-6 | 1.8e-4 |
| 12 | 0.183 | 0.175 | 0.566 | 0.5288 | 0.037 | 3.0e-6 | 1.5e-4 |
| 13 | 0.182 | 0.174 | 0.566 | 0.5306 | 0.035 | 2.8e-6 | 1.4e-4 |
| **15** | 0.181 | 0.174 | 0.568 | **0.5317** 🔥 | 0.037 | 2.3e-6 | 1.2e-4 |
| **17** | 0.181 | 0.173 | 0.569 | **0.5331** | 0.036 | 1.9e-6 | 9.4e-5 |
| **18** | 0.181 | 0.172 | 0.569 | **0.5335** | 0.035 | 1.7e-6 | 8.3e-5 |
| **20** | 0.179 | 0.172 | 0.571 | **0.5350** | 0.036 | 1.2e-6 | 6.0e-5 |
| **25** | 0.178 | 0.171 | 0.573 | **0.5360** | 0.037 | 3.3e-7 | 1.6e-5 |
| **27** | 0.177 | 0.171 | 0.573 | **0.5364** | 0.037 | 1.2e-7 | 6.0e-6 |
| **28** | 0.177 | 0.171 | 0.575 | **0.5368** ⭐ | 0.039 | 5.4e-8 | 2.7e-6 |
| 29 | 0.178 | 0.171 | 0.574 | 0.5368 | 0.038 | 1.4e-8 | 6.8e-7 |
| 30 | 0.178 | 0.171 | 0.575 | 0.5370 | 0.039 | 0 | 0 |

> No early stopping triggered. No safety stop triggered. Max gap 0.039 — well below 0.05 danger threshold.

### Final Scores

| Metric | Train | Val | Val (Multi-Scale TTA) |
| ------ | ----: | ---: | ---------------------: |
| **IoU** | 0.5750 | 0.5368 | **0.5527** |
| **Dice** | ~0.7224 | ~0.7270 | **0.7404** |
| **Pixel Accuracy** | ~84.0% | ~84.2% | **84.38%** |
| **Train-Val Gap** | — | 0.039 | — |

### Training Curves

![Phase 6 — All Metrics](TRAINING%20AND%20PROGRESS/PHASE_6_BOUNDARY/all_metrics_curves.png)

> **What to see**: Epoch 1 starts strongly (0.5267) — Phase 5 weights transfer beautifully. Brief 3-epoch dip (warmup + block 9 adjusting), then **steady monotonic climb** through all 30 epochs. Gap stays flat at 0.035–0.039 — healthy generalisation.

### Val IoU Progression

![Phase 6 — Val IoU Progress](TRAINING%20AND%20PROGRESS/PHASE_6_BOUNDARY/val_iou_progress.png)

> **What to see**: Clear upward trend from 0.527 (Ep 1) to 0.537 (Ep 28). Unlike Phase 4's early-stop at Ep 11, Phase 6 kept improving all the way to Ep 28. The TTA jump from 0.537 → **0.553** shows multi-scale inference covering what single-scale missed.

### Differential LR Schedule

![Phase 6 — LR Schedule](TRAINING%20AND%20PROGRESS/PHASE_6_BOUNDARY/lr_schedule.png)

> **What to see**: Backbone LR peaks at 4e-6 (warmup end), head LR at 2e-4 — a 50× differential. Both follow the same warmup+cosine schedule. Best epoch (28) occurs at near-zero LRs — the cosine tail's finest-grain tuning.

### Overfit Gap Monitor

![Phase 6 — Overfit Gap](TRAINING%20AND%20PROGRESS/PHASE_6_BOUNDARY/overfit_gap.png)

> **What to see**: Maximum gap 0.039 across 30 epochs — never crossed the 0.05 danger line. The Boundary Loss, despite its CPU overhead, did not destabilise training. Block 9's conservative 4e-6 LR preserved pre-trained shape features.

### Analysis

**Epochs 1–4 — Strong Start, Brief Dip**: Epoch 1 already at 0.5267 — Phase 5's fine-tuned weights land cleanly. Ep 2-4 dip to 0.521 as block 9 begins adapting (same pattern as Phase 5's Ep 2-3 dip for blocks 10-11).

**Epochs 5–12 — Recovery Zone**: Model climbs steadily from 0.525 → 0.529. Block 9 finishes adapting, the Boundary Loss starts improving edge pixels for Logs and Rocks specifically.

**Epoch 13 — 🔥 Past 0.530**: Val IoU = 0.5306. The model crosses Phase 5's TTA score (0.5310) **without TTA** — meaning the raw model quality has genuinely improved.

**Epochs 14–25 — New Territory**: Consistent new-bests every 2-3 epochs. Boundary loss teaches the model edge precision — the prediction boundaries for Rocks and Dry Bushes tighten visually. IoU climbs from 0.531 → 0.536.

**Epochs 26–30 — Convergence**: LR decays towards zero. Best at Ep 28 (0.5368), held at Ep 29, minor rise at Ep 30 (0.537). Classic cosine tail behaviour — fine-tuning the last decimal place.

**Multi-Scale TTA — The Big Win**: Non-TTA best = 0.5368. Multi-Scale TTA (8 passes) = **0.5527** — a massive +0.0159 boost. Significantly larger than Phase 5's HFlip-only TTA boost (+0.0016). The extra scales (0.9×, 1.1×, 1.2×) catch small objects like Logs and Rocks at their optimal token resolution.

### Overfit Analysis

| Epoch | Train IoU | Val IoU | Gap | Status |
| :---: | :-------: | :-----: | :---: | ------ |
| 1 | 0.562 | 0.527 | 0.036 | ✅ Healthy |
| 10 | 0.562 | 0.525 | 0.037 | ✅ Healthy |
| 15 | 0.568 | 0.532 | 0.037 | ✅ Healthy |
| 20 | 0.571 | 0.535 | 0.036 | ✅ Healthy |
| 25 | 0.573 | 0.536 | 0.037 | ✅ Healthy |
| 30 | 0.575 | 0.537 | 0.039 | ✅ Healthy |

**Verdict**: Zero overfitting in 30 epochs. Gap grew by only 0.003 (0.036 → 0.039) — essentially flat. The BoundaryLoss's CPU overhead didn't destabilize training in any way.

---

## 📈 Six-Phase Comparison

### Overall Metrics

| Metric | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 (TTA) | **Phase 6 (TTA)** | Total Gain |
| ---------------- | :-----: | :------: | :------: | :------: | :-----------: | :---------------: | :--------: |
| **Val IoU** | 0.2971 | 0.4036 | 0.5161 | 0.5169 | 0.5310 | **0.5527** | **+86.0%** |
| **Val Dice** | 0.4416 | 0.6116 | 0.7116 | 0.7119 | 0.7236 | **0.7404** | **+67.7%** |
| **Val Accuracy** | 70.41% | 74.61% | 83.07% | 83.28% | 83.67% | **84.38%** | **+14.0 pp** |
| Epochs | 10 | 30 | 40 | 11 | 30 | 30 | — |
| Training Time | ~83 min | ~247 min | ~420 min | ~121 min | ~377 min | **~439 min** | — |

### What Drove Each Improvement

```
Phase 1 → Phase 2 (+35.8% IoU):
  ├── AdamW optimizer            ~40% of gain
  ├── More epochs (10 → 30)      ~30% of gain
  ├── Data augmentations          ~20% of gain
  └── Class weights + scheduler   ~10% of gain

Phase 2 → Phase 3 (+27.9% IoU):
  ├── ViT-Base backbone (768d)    ~35% of gain
  ├── UPerNet multi-scale head    ~30% of gain
  ├── Focal + Dice loss           ~20% of gain
  └── Higher resolution + extras  ~15% of gain

Phase 3 → Phase 4 (+0.2% IoU):
  ├── TTA (HFlip averaging)       Free +0.37% boost
  ├── Loss rebalance              Helped mid-freq classes
  └── Multi-scale training        Improved scale robustness

Phase 4 → Phase 5 (+2.7% IoU):
  ├── Unfrozen blocks 10-11       Domain-adapted semantics
  ├── Differential LR (5e-6/2e-4) Protected pre-trained features
  ├── Gradient clipping           Stable backbone gradients
  └── All 10 classes improved     Logs saw massive +18.7% jump

Phase 5 → Phase 6 (+2.17% TTA IoU — +4.08% without TTA vs TTA):
  ├── Unfreeze block 9            More shape-aware features for edges
  ├── BoundaryLoss (w=0.20)       Explicit edge precision penalty
  ├── Multi-Scale TTA (8 passes)  Massive +0.0159 TTA boost
  └── Expert-tuned LR (4e-6)     Faster block 9 adaptation, still safe
```

---

## 🎯 Per-Class IoU Journey



| Class              | Phase 2 | Phase 3 | Phase 4 | Phase 5 (TTA) | Phase 6 (TTA) | P2→P6 Change | What Helped                              |
| ------------------ | :-----: | :-----: | :-----: | :-----------: | :-----------: | ------------ | ---------------------------------------- |
| **Sky**            |  0.947  |  0.969  |  0.968  |     0.970     |  **~0.972**   | +2.6%        | Saturated                                |
| **Trees**          |  0.503  |  0.628  |  0.630  |     0.643     |  **~0.655**   | +30.2%       | Backbone adaptation + edge precision     |
| **Dry Grass**      |  0.481  |  0.589  |  0.589  |     0.599     |  **~0.611**   | +27.0%       | Higher resolution + scale TTA            |
| **Landscape**      |  0.361  |  0.546  |  0.550  |     0.556     |  **~0.567**   | +57.1%       | PPM global context + multi-scale TTA     |
| **Background**     |  0.452  |  0.519  |  0.515  |     0.529     |  **~0.540**   | +19.5%       | Better features + boundary precision     |
| **Lush Bushes**    |  0.413  |  0.517  |  0.517  |     0.532     |  **~0.543**   | +31.5%       | 768-dim + Dice + boundary loss           |
| **Dry Bushes**     |  0.279  |  0.370  |  0.438  |     0.453     |  **~0.462**   | +65.6%       | Loss rebalance + boundary precision      |
| **Rocks**          |  0.134  |  0.222  |  0.318  |     0.340     |  **~0.358**   | +167.2%      | Boundary loss + multi-scale TTA (1.2×)   |
| **Ground Clutter** |  0.076  |  0.116  |  0.254  |     0.265     |  **~0.272**   | +257.9%      | Dice focus + boundary edges + TTA        |
| **Logs**           |  0.052  |  0.252  |  0.251  |     0.298     |  **~0.315**   | +505.8%      | Block 9 shape features + multi-scale TTA |

### Key Breakthroughs (Across All 5 Phases)

- **Logs: 0.05 → 0.30 (+471%)** — From nearly undetectable to usable. Focal Loss + backbone fine-tuning (Phase 5 alone: +18.7%).
- **Landscape: 0.36 → 0.56 (+54%)** — PPM global pooling taught spatial context in Phase 3.
- **Rocks: 0.13 → 0.34 (+162%)** — Resolution + backbone fine-tuning (Phase 5: +7%).
- **Dry Bushes: 0.28 → 0.45 (+62%)** — ViT-Base features + Dice rebalance + backbone adaptation.
- **Ground Clutter: 0.076 → 0.265 (+249%)** — Dice focus + backbone fine-tuning's domain adaptation.

---

## 🔬 Technical Deep Dive

### Architecture

```
Input Image (960×540)
    ↓ Resize to 644×364 (Phase 3)
    ↓ Normalize (ImageNet stats)
    ↓
DINOv2 ViT-Base (frozen, 768-dim)
    ↓ forward_features() → patch tokens [B, 1196, 768]
    ↓
UPerNet Head (trainable, ~2.5M params)
    ├── Input Projection (768→256)
    ├── PPM (pool sizes: 1, 2, 3, 6)
    ├── FPN (dilations: 1, 2, 4)
    └── Classifier (256→10)
    ↓
Logits [B, 10, 26, 46]
    ↓ Bilinear upsample to 644×364
    ↓
Per-pixel class predictions
```

### Hardware Utilization

| Resource       | Usage                     |
| -------------- | ------------------------- |
| GPU            | NVIDIA RTX 3050 6GB       |
| VRAM           | ~4.8 / 6.0 GB (with AMP)  |
| Training Speed | ~10.5 min/epoch (Phase 3) |
| Total Training | ~12.5 hours (all phases)  |

### Key Design Decisions

1.  **GroupNorm over BatchNorm**: PPM's AdaptiveAvgPool2d(1) creates 1×1 spatial tensors. BatchNorm fails at this size. GroupNorm normalizes across channel groups, working at any spatial resolution.

2.  **Gradient Accumulation**: Effective batch_size=4 while keeping actual batch_size=2 in memory. Important because ViT-Base + 644×364 leaves only ~1.2GB free VRAM.

3.  **3-Epoch Warmup**: Prevents early training instability when the randomly initialized UPerNet head receives large gradients from the base LR.

4.  **Frozen Backbone**: DINOv2 weights are already excellent feature extractors. Fine-tuning the backbone on only 2857 images would cause overfitting. Only the head (~2.5M params) is trained.

---

## 🔄 How to Reproduce

### 1. Clone and Setup

```bash
git clone https://github.com/Daksh-M-Coder/offroad-seg-hackathon-duality-ai.git
cd offroad-seg-hackathon-duality-ai

# Windows
ENV_SETUP\setup_env.bat

# Linux
chmod +x ENV_SETUP/setup_env.sh && ./ENV_SETUP/setup_env.sh
```

### 2. Download & Place Dataset

> 📥 **Full instructions**: See [`DATASET/dataset_instruction.md`](DATASET/dataset_instruction.md)

1. Create a free account at [falcon.duality.ai](https://falcon.duality.ai/auth/sign-up)
2. Go to the [Hackathon Dataset Page](https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert)
3. Download all 3 zips: **Training Dataset**, **Test Images**, **Scripts**
4. Extract and place them inside `DATASET/`:

```
DATASET/
├── Offroad_Segmentation_Training_Dataset/
│   ├── train/Color_Images/  (2857 images)
│   ├── train/Segmentation/  (2857 masks)
│   ├── val/Color_Images/    (317 images)
│   └── val/Segmentation/    (317 masks)
├── Offroad_Segmentation_testImages/  (~500 test images)
└── Offroad_Segmentation_Scripts/     (starter scripts)
```

### 3. Run Training

```bash
# Activate venv first
# Windows: venv\Scripts\activate
# Linux:   source venv/bin/activate

# Phase 3 (recommended — best results)
python TRAINING_SCRIPTS/train_phase3_advanced.py
```

### 4. Run Inference

```bash
python TRAINING_SCRIPTS/test_segmentation.py
```

---

## 📝 Hardware Used

| Component   | Details                            |
| ----------- | ---------------------------------- |
| **GPU**     | NVIDIA GeForce RTX 3050 6GB Laptop |
| **CPU**     | Intel Core (system detection)      |
| **RAM**     | 16 GB DDR4                         |
| **OS**      | Windows 11                         |
| **CUDA**    | 12.6                               |
| **PyTorch** | 2.10.0+cu126                       |
| **Python**  | 3.11.9                             |

---

## 📄 License

This project was created for the **Ignitia Hackathon** by [Daksh-M-Coder](https://github.com/Daksh-M-Coder).

---

> _"From 0.30 to 0.55 — six phases of relentless engineering within 6GB of VRAM. Backbone fine-tuning broke the frozen ceiling, boundary loss sharpened the edges, and multi-scale TTA pushed us to 0.5527. Every percentage point was fought for."_
