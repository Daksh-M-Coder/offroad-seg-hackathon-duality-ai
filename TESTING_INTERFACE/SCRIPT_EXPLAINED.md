# Testing Interface — Script Explained

> **File**: `TESTING_INTERFACE/app.py`  
> **Purpose**: Interactive Gradio dashboard for visually testing the **Phase 6** segmentation model on training/validation images and custom uploads.

---

## What It Does

The testing interface gives a **visual, interactive way** to see the model's predictions — something the training scripts can't provide. It lets you:

1. Browse the **10 fixed pre-indexed samples for every class** and run segmentation on them.
2. **Randomly pick** any image from the full dataset pool for any class.
3. **Upload your own** offroad image for inference.
4. See a **colour-coded overlay**, raw mask, and per-class IoU bar chart for every prediction.
5. Automatically **log every result** as a timestamped Markdown file in `TESTING_INTERFACE/RESULTS/`.

---

## Model Loaded

The interface loads the **Phase 6 best checkpoint** (`MODELS/phase6_best_model_iou0.5368.pth`):

| Component      | Detail                                                                          |
| -------------- | ------------------------------------------------------------------------------- |
| **Backbone**   | DINOv2 ViT-Base (`dinov2_vitb14_reg`), **blocks 9-11 fine-tuned**              |
| **Head**       | UPerNet (PPM + multi-scale FPN, GroupNorm)                                      |
| **Input size** | 644×364 (46×26 patch tokens)                                                   |
| **TTA**        | Horizontal flip + logit average (same as training eval)                         |
| **Output**     | 10-class segmentation map                                                        |
| **Best IoU**   | **0.5368** (Val, non-TTA) / **0.5527** (Val, Multi-Scale TTA)                  |

The `backbone_state_dict` stored in the checkpoint (containing fine-tuned blocks **9, 10, and 11**) is automatically restored when loading. Block 9 contains shape-encoding features that Phase 6 adapted specifically for desert terrain edge precision.

### Model Changelog

| Version | File | Val IoU (TTA) | Key Change |
|---|---|---|---|
| Phase 5 | `phase5_best_model_iou0.5294.pth` | 0.5310 | Blocks 10-11 fine-tuned |
| **Phase 6** ← current | `phase6_best_model_iou0.5368.pth` | **0.5527** | Blocks 9-11 + BoundaryLoss + Multi-Scale TTA |

---

## Interface Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Header: model info, best IoU, class count                  │
├────────────────────────────┬────────────────────────────────┤
│  Original | Overlay | Mask │  Class Legend (colour patches) │
├────────────────────────────┴────────────────────────────────┤
│  Per-Class IoU Bar Chart    │  Metrics Detail (text panel)  │
├─────────────────────────────────────────────────────────────┤
│  💾 Result Log (saved filename)                             │
├─────────────────────────────────────────────────────────────┤
│  TABS:                                                       │
│  [📂 Class Samples]  [🎲 Random Pick]  [📤 Upload]          │
└─────────────────────────────────────────────────────────────┘
```

---

## The Three Tabs

### Tab 1 — Class Samples (Fixed 10 per class)

- **Dropdown**: Select any of the 10 semantic classes by name.
- **Slider (1–10)**: Pick which of the 10 fixed samples to use.
- **Purpose**: Reproducible testing — the same sample always gives the same image. Use this to compare model behaviour across phases or fine-tuning runs.
- **Ground truth available**: ✅ Yes — IoU and Dice are computed against the real mask.

The 10 samples per class are selected at startup using a fixed seed (`random.seed(42 + class_id)`) from all images in the train+val sets where the class covers at least 500 pixels.

### Tab 2 — Random Class Sample

- **Dropdown**: Select a class.
- **Button**: Each click picks a **random** image from the full pool of that class (100s of options).
- **Purpose**: Stress-test the model on different images of the same class; catch failure modes.
- **Ground truth available**: ✅ Yes.

### Tab 3 — Upload Your Own Image

- **Image uploader**: Drag-and-drop or browse any image.
- **Purpose**: Test on completely unseen images, judge images, or custom screenshots.
- **Ground truth available**: ❌ No — pixel coverage per class shown instead.

---

## Outputs Explained

### Segmentation Overlay

The original image blended with the colour-coded prediction mask at **55% mask opacity**. Shows which region the model assigned to each class. Dark areas = Background (near-black), Green = Trees/Bushes, Sky Blue = Sky, etc.

### Prediction Mask

Pure colour-coded mask with **no image blending** — useful to see exact prediction boundaries. Phase 6's BoundaryLoss makes these edges noticeably sharper than Phase 5, especially for Logs and Rocks.

### Class Legend

A static reference panel showing **which colour = which class** for all 10 categories.

### Per-Class IoU Bar Chart

- Each bar = one class, coloured with its segmentation colour.
- Red dashed line = mean IoU for this image.
- Values printed above each bar.
- Only meaningful when ground truth is available — otherwise set to NaN.

### Metrics Text Panel

Shows:

- Mean IoU, Pixel Accuracy (when GT available)
- Per-class table: IoU, Dice, Predicted pixels, GT pixels

---

## Class Colour Palette

| ID  | Class          | Colour                       |
| --- | -------------- | ---------------------------- |
| 0   | Background     | Near-black (20,20,20)        |
| 1   | Trees          | Forest green (34,139,34)     |
| 2   | Lush Bushes    | Lime green (0,220,0)         |
| 3   | Dry Grass      | Tan (210,180,140)            |
| 4   | Dry Bushes     | Brown (139,90,43)            |
| 5   | Ground Clutter | Olive (128,128,0)            |
| 6   | Logs           | Peru (205,133,63)            |
| 7   | Rocks          | Gray (150,150,150)           |
| 8   | Landscape      | Sienna (160,82,45)           |
| 9   | Sky            | Light sky blue (135,206,250) |

---

## Result Logging

Every single prediction (no matter which tab) automatically saves a Markdown file and 3 images:

```
TESTING_INTERFACE/
├── RESULTS/
│   ├── 0001_20260310_011500.md     ← metrics + embedded images
│   ├── 0002_20260310_011523.md
│   └── ...
└── IMGS/
    ├── 0001_20260310_011500_raw.png
    ├── 0001_20260310_011500_overlay.png
    ├── 0001_20260310_011500_mask.png
    └── ...
```

**Filename format**: `{4-digit sequence}_{YYYYMMDD_HHMMSS}_{type}`

The sequence counter **persists across restarts** — it scans existing RESULTS/IMGS/ on startup to find the highest existing number and continues from there. No gaps, no collisions.

Each `.md` result file contains:

- Source (which tab, which class, which file)
- **Model version**: Phase 6 (`phase6_best_model_iou0.5368.pth`)
- Device (CUDA/CPU)
- Metrics table (IoU, Dice, pixel counts per class)
- Timestamp
- **Embedded images** (raw, overlay, mask) as a comparison table

---

## How to Run

```bash
cd "PIXEL IMG SEGMENTATION"
venv\Scripts\python.exe TESTING_INTERFACE\app.py
```

Then open your browser at `http://localhost:7860`

### Requirements

The following additional package is needed:

```
gradio>=4.0
```

Install with:

```bash
venv\Scripts\pip install gradio
```

---

## Preprocessing Pipeline

1. **Resize** input image to 644×364 (to match training resolution)
2. **Normalize** with ImageNet stats: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
3. **Backbone forward pass** → DINOv2 extracts patch tokens (shape: [1, 46×26, 768])
4. **TTA flip** → forward pass on horizontally flipped image
5. **UPerNet forward pass** on tokens → class logits (shape: [1, 10, 26, 46])
6. **Bilinear upsample** to 644×364
7. **Average original + flipped logits** (TTA)
8. **Argmax** → predicted class per pixel (shape: [364, 644])
9. **Colour map** → RGB visualization

---

## Parameters Chosen and Why

| Parameter                      | Value           | Reason                                                          |
| ------------------------------ | --------------- | --------------------------------------------------------------- |
| `SAMPLES_PER_CLASS`            | 10              | Enough variety without memory overhead                          |
| `alpha` (overlay opacity)      | 0.55            | Clear mask visibility while keeping original visible            |
| Min pixels for class inclusion | 500             | Avoid marking tiny class appearances — need real representation |
| Fixed seed for samples         | `42 + class_id` | Reproducible selection across runs                              |
| TTA                            | HFlip average   | Same as used during training eval, free ~+0.3% IoU boost        |

---

## Known Limitations

1. **Ground Clutter & Logs**: Still the hardest classes — Phase 6 improved them (Logs ~0.315 TTA IoU, Ground Clutter ~0.272) but they remain challenging due to dataset scarcity.
2. **Upload tab**: No ground-truth mask available — only pixel coverage is shown.
3. **Load time**: Model loading takes ~30s on first start (DINOv2 downloads from torch.hub if not cached).
4. **Dataset indexing**: ~15-30s on startup to scan all training/validation images.
5. **Per-class IoU values in interface** reflect single-scale HFlip TTA (not Multi-Scale). The Multi-Scale TTA (0.5527) requires 8 passes and is too slow for interactive use.

---
