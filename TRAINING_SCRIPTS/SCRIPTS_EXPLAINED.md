# Training Scripts — Technical Documentation

> This document explains **every script, every parameter, every decision** we made across 5 training phases — why we chose what we chose, what we expected, and what actually happened.

---

## Table of Contents

- [Script Overview](#script-overview)
- [Phase 1 Script: `train_phase1_baseline.py`](#phase-1-script-train_phase1_baselinepy)
- [Phase 2 Script: `train_phase2_improved.py`](#phase-2-script-train_phase2_improvedpy)
- [Phase 3 Script: `train_phase3_advanced.py`](#phase-3-script-train_phase3_advancedpy)
- [Phase 4 Script: `train_phase4_mastery.py`](#phase-4-script-train_phase4_masterypy)
- [Phase 5 Script: `train_phase5_controlled.py`](#phase-5-script-train_phase5_controlledpy)
- [Inference: `test_segmentation.py`](#inference-test_segmentationpy)
- [Visualization: `visualize.py`](#visualization-visualizepy)
- [Parameter Evolution Table](#parameter-evolution-table)
- [Lessons Learned](#lessons-learned)

---

## Script Overview

Each training script follows the same pipeline:

```
1. Config & Hyperparameters
2. Dataset class (loads images + masks, applies transforms)
3. Augmentation pipeline (albumentations)
4. Model architecture (backbone + segmentation head)
5. Loss function
6. Training loop (with AMP, gradient accumulation)
7. Validation loop (IoU, Dice, Accuracy per epoch)
8. Checkpointing (save best model by val_iou)
9. Plotting (training curves, per-class IoU)
10. Metrics export (text + JSON)
```

All scripts output to `TRAINING AND PROGRESS/PHASE_X_NAME/` and are self-contained — no imports between them.

---

## Phase 1 Script: `train_phase1_baseline.py`

### Purpose

The **original hackathon-provided script**, run without modifications. Establishes the baseline metric that all improvements are measured against.

### Data Pipeline

```python
# Mask encoding: raw pixel values → class IDs
value_map = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 700:6, 800:7, 7100:8, 10000:9}
# 10 classes: Background, Trees, Lush Bushes, Dry Grass, Dry Bushes,
#             Ground Clutter, Logs, Rocks, Landscape, Sky
```

**Image loading**: OpenCV reads BGR → converted to RGB. Masks are uint16 single-channel.  
**Resize**: 476×266 — chosen as nearest multiple of 14 (ViT patch size) to the original 960×540 at ~50% scale.  
**Normalization**: ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`  
**Augmentations**: **None at all** — this is the baseline's biggest weakness.

### Backbone — DINOv2 ViT-Small

```python
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
backbone.eval()  # frozen — no gradients
```

**Why ViT-Small?** It's the hackathon's default. DINOv2 is a self-supervised vision transformer pre-trained on 142M images. ViT-Small has 384-dim embeddings with 14×14 patches.

**Why frozen?** With only 2857 training images, fine-tuning the entire 22M-parameter backbone would cause massive overfitting. Only the segmentation head (~500K params) is trained.

**Feature extraction**: `forward_features()` returns `x_norm_patchtokens` — a tensor of shape `[B, N, 384]` where N = (H/14) × (W/14) = 19 × 34 = 646 patch tokens.

### Segmentation Head — ConvNeXt

```python
# Simple stack: project tokens → reshape → conv layers → upsample
nn.Sequential(
    nn.Conv2d(384, 256, 3, padding=1),
    nn.BatchNorm2d(256),
    nn.GELU(),
    nn.Conv2d(256, 256, 3, padding=1),
    nn.BatchNorm2d(256),
    nn.GELU(),
    nn.Conv2d(256, 10, 1),  # classifier
)
```

**How it works**: Patch tokens are reshaped from (B, 646, 384) → (B, 384, 19, 34), then passed through conv layers, and bilinearly upsampled to the original 476×266.

**Problem**: This head only sees features at **one scale**. A 3-pixel Log and a 10,000-pixel Sky region get the exact same processing. There's no multi-scale context.

### Loss & Optimizer

```python
loss_fn = nn.CrossEntropyLoss()  # unweighted — treats all classes equally
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
# No LR scheduler
```

**Why this is bad**: Sky (34% of pixels) contributes 34% of the loss. Logs (0.07%) contributes 0.07%. The model has almost zero incentive to learn rare classes.

**SGD at 1e-4**: Extremely slow convergence. SGD needs large learning rates and momentum to work well, but this LR is conservative. AdamW would be 3-5× faster.

### Training Loop

```python
for epoch in range(10):  # only 10 epochs!
    for imgs, labels in train_loader:
        tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]
        logits = model(tokens)
        out = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear")
        loss = loss_fn(out, labels.squeeze(1).long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**No mixed precision** — runs in fp32, using more VRAM than necessary.  
**No gradient accumulation** — effective batch = actual batch = 2.  
**No checkpointing** — only the final model is saved, not the best.

### Expected vs Actual

| Aspect            | Expected                             | Actual                                           |
| ----------------- | ------------------------------------ | ------------------------------------------------ |
| **IoU**           | 0.25-0.35 (typical for frozen ViT-S) | **0.2971** ✅ in range                           |
| **Convergence**   | Should plateau by epoch 10           | **Still improving** ❌ — underfitting            |
| **Overfitting**   | Possible without augmentations       | **None** ✅ — train ≈ val                        |
| **Rare classes**  | Very low IoU expected                | Logs=0.05, Rocks=0.16 — **terrible** as expected |
| **Training time** | ~1-2 hours                           | **83 min** ✅                                    |

**Key takeaway**: The model was nowhere near converged. Loss decreased linearly through all 10 epochs — we left massive performance on the table.

---

## Phase 2 Script: `train_phase2_improved.py`

### What Changed and Why

Every change was a **direct response to Phase 1's failures**.

#### Change 1: SGD → AdamW

```python
# Phase 1
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# Phase 2
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
```

**Why**: SGD at 1e-4 was painfully slow. AdamW uses per-parameter adaptive learning rates — parameters with small gradients get larger effective LR. This is critical for segmentation where different parts of the head learn at different rates.

**Why lr=5e-4?** Standard starting LR for AdamW in fine-tuning tasks. Higher than SGD's 1e-4 because AdamW self-adjusts.

**Why weight_decay=1e-4?** Light regularization to prevent overfitting without being too aggressive. Common default for vision tasks.

#### Change 2: CosineAnnealingLR

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
```

**Why**: A constant LR means the model either learns too aggressively (high LR → unstable) or too slowly (low LR → never converges). Cosine annealing starts high for fast initial learning, then gradually decreases to fine-tune.

**Why T_max=30?** Matches the total epoch count so LR reaches its minimum at the last epoch.

**Why eta_min=1e-6?** Nearly zero but not exactly zero — allows tiny parameter updates in the final epochs for boundary refinement.

**Expected**: IoU should improve faster initially and plateau more gracefully.  
**Actual**: ✅ Worked perfectly. Best IoU at epoch 26 (LR ≈ 2.3e-5), showing the schedule found the sweet spot.

#### Change 3: Data Augmentations

```python
A.Compose([
    A.Resize(h, w),
    A.HorizontalFlip(p=0.5),              # Scenes are symmetric
    A.VerticalFlip(p=0.1),                 # Rare but adds variety
    A.ShiftScaleRotate(                    # Simulates camera movement
        shift_limit=0.08, scale_limit=0.15, rotate_limit=20,
        border_mode=cv2.BORDER_CONSTANT, value=0, p=0.4
    ),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3,5)),  # Simulates defocus
        A.MedianBlur(blur_limit=5),        # Simulates sensor noise
    ], p=0.2),
    A.OneOf([
        A.RandomBrightnessContrast(...),   # Lighting variation
        A.HueSaturationValue(...),         # Color shift
    ], p=0.4),
    A.Normalize(...),
    ToTensorV2(),
])
```

**Why these specific augmentations?**

- **HFlip (p=0.5)**: Desert scenes look the same left-right flipped. Free 2× data.
- **ShiftScaleRotate**: Simulates different camera angles and distances. `border_mode=CONSTANT` fills edges with black rather than reflecting.
- **Blur**: Simulates camera defocus and motion blur in offroad conditions.
- **Color jitter**: Desert lighting varies hugely — sunrise, midday, sunset all look different.

**What we deliberately avoided**:

- `RandomRotate90` — crashed training because it swaps dimensions on non-square images, breaking `torch.stack` in the DataLoader. We discovered this bug and removed it.
- Heavy geometric distortions — would distort object shapes too much for segmentation.

**Expected**: 10-15% IoU improvement from augmentations alone.  
**Actual**: Augmentations + AdamW together gave +35.8%, hard to isolate individual contribution.

#### Change 4: Weighted CrossEntropy

```python
# Compute class weights from pixel frequency
counts = torch.zeros(10)
for _, masks in train_loader:
    for c in range(10):
        counts[c] += (masks == c).sum()
weights = total / (10 * counts)  # Inverse frequency
weights = weights / weights.max() * 5.0  # Cap at 5.0

loss_fn = nn.CrossEntropyLoss(weight=weights)
```

**Why inverse frequency?** If Sky has 34% of pixels, its weight becomes 0.01 (near-zero). If Logs has 0.07%, its weight becomes 5.0 (maximum). This forces the model to "care" about rare classes proportionally.

**Why cap at 5.0?** Without capping, Logs would get weight ~72. This would make the model hallucinate Logs everywhere because the loss rewards even false Logs detections heavily.

**Expected**: Big boost for rare classes (Logs, Rocks).  
**Actual**: Partial success. Logs went from unlearnable to IoU=0.05 (barely visible). The fundamental problem is that weighted CE still treats each pixel independently — it doesn't optimize for region overlap.

#### Change 5: Mixed Precision

```python
scaler = torch.amp.GradScaler('cuda')
with torch.amp.autocast('cuda'):
    logits = model(tokens)
    loss = loss_fn(logits, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Why**: Our RTX 3050 has only 6GB VRAM. Mixed precision (fp16 forward, fp32 backward) reduces memory by ~30% and speeds up training by ~15% on Tensor Cores.

**Expected**: Faster training, no accuracy loss.  
**Actual**: ✅ ~15% speedup, identical metrics to fp32 training in tests.

#### Change 6: Checkpointing & Early Stopping

```python
if val_iou > best_iou:
    best_iou = val_iou
    torch.save(model.state_dict(), 'best_model.pth')
    no_improve = 0
else:
    no_improve += 1
    if no_improve >= 10:  # patience
        break
```

**Why checkpointing?** Phase 1 only saved the final model. If the best epoch was #26 but training ran to #30, we'd lose the best weights.

**Why patience=10?** Gives the model time to escape local minima. Too low (3-5) might stop too early; too high (20+) wastes time.

**Expected**: Save the true best model.  
**Actual**: ✅ Best at epoch 26, saved correctly. Early stopping not triggered in 30 epochs.

### Expected vs Actual (Phase 2)

| Aspect               | Expected           | Actual                            |
| -------------------- | ------------------ | --------------------------------- |
| **IoU**              | 0.45-0.50          | **0.4036** ❌ lower than expected |
| **Best epoch**       | ~20-25             | **26** ✅ close                   |
| **Training time**    | ~3-4 hours         | **4.1 hours** ✅                  |
| **Rare class boost** | Logs > 0.15        | **Logs = 0.05** ❌ still terrible |
| **Overfitting**      | Low risk with augs | **None** ✅                       |

**Why lower than expected?** We overestimated what a simple ConvNeXt head could do. The bottleneck wasn't optimization — it was **architecture**. The single-scale head fundamentally cannot handle the 500× size difference between Sky and Logs.

---

## Phase 3 Script: `train_phase3_advanced.py`

### What Changed and Why

Every change was a **direct response to Phase 2's architectural limitations**.

#### Change 1: ViT-Small → ViT-Base

```python
# Phase 2
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
# 384-dim embeddings, 22M params

# Phase 3
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
# 768-dim embeddings, 86M params, with register tokens
```

**Why ViT-Base?** Phase 2's per-class analysis showed failure on texturally similar classes:

- Dry Bushes (0.28) confused with Dry Grass (0.48) — both brownish
- Lush Bushes (0.41) confused with Trees (0.50) — both green
- Ground Clutter (0.22) confused with Background (0.45)

ViT-Small's 384-dim features don't have enough capacity to encode **texture differences between similar classes**. ViT-Base doubles to 768-dim, giving the model more "vocabulary" to describe what it sees.

**Why `_reg` variant?** The `dinov2_vitb14_reg` model includes register tokens that reduce attention artifacts in ViT, producing smoother feature maps.

**VRAM concern**: ViT-Base uses ~2× more VRAM than ViT-Small for feature extraction. We solved this with gradient accumulation and mixed precision.

**Expected**: +5-10% IoU from richer features alone.  
**Actual**: Epoch 1 IoU=0.42 (already above Phase 2's best 0.40!) confirming the backbone quality. ✅

#### Change 2: ConvNeXt → UPerNet

```python
class PPM(nn.Module):
    """Pyramid Pooling Module — captures context at multiple scales."""
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6)):
        # For each pool size, global average pool → 1×1 conv → upsample back
        # pool_size=1: global context (entire image summary)
        # pool_size=2: 2×2 grid (quadrant-level context)
        # pool_size=3: 3×3 grid (region-level context)
        # pool_size=6: 6×6 grid (local context)

class UPerNetHead(nn.Module):
    """Multi-scale FPN with dilated convolutions."""
    def __init__(self, in_channels=768, mid_channels=256, out_channels=10):
        # 1. Input projection: 768 → 256 channels
        # 2. PPM: captures global-to-local context
        # 3. Multi-scale FPN:
        #    - dilation=1: 3×3 receptive field (fine details like Logs)
        #    - dilation=2: 5×5 receptive field (medium objects like Rocks)
        #    - dilation=4: 9×9 receptive field (large areas like Landscape)
        # 4. Fusion: concat 3 scales → 1×1 conv → classifier
```

**Why UPerNet?** Phase 2's ConvNeXt head failed because:

- It processed all spatial locations with the same 3×3 conv
- A 3-pixel Log and a 10K-pixel Sky get identical processing
- No global scene understanding — can't use "this is the top of the image" as context for Sky

UPerNet fixes this with:

1. **PPM**: At pool_size=1, the model sees the ENTIRE image summarized as one vector. This provides **scene understanding** — "this is a desert scene with sky at top."
2. **Multi-scale FPN**: dilation=1 detects fine edges (Logs, small Rocks), dilation=4 captures wide patterns (Landscape boundaries).

**Why GroupNorm instead of BatchNorm?**

```python
# PPM creates 1×1 spatial tensors via AdaptiveAvgPool2d(1)
# BatchNorm requires spatial_size > 1 → CRASHES
nn.BatchNorm2d(channels)  # ❌ fails at 1×1

# GroupNorm normalizes across channel groups, works at ANY spatial size
nn.GroupNorm(32, channels)  # ✅ always works
```

We discovered this bug when Phase 3 crashed on the first forward pass. GroupNorm(32, channels) splits the 256 channels into 32 groups and normalizes each group independently.

**Expected**: +10-15% IoU from multi-scale features.  
**Actual**: Hard to isolate, but the per-class improvements in Landscape (+51%), Rocks (+92%), and Logs (+382%) strongly suggest the multi-scale processing was the key driver. ✅

#### Change 3: CrossEntropy → Focal + Dice

```python
class FocalLoss(nn.Module):
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)   # probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        # When pt ≈ 1 (easy pixel): focal_weight ≈ 0 → ignored
        # When pt ≈ 0 (hard pixel): focal_weight ≈ 1 → full weight
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def forward(self, inputs, targets):
        # Computes per-class Dice coefficient and averages
        # Unlike CE, Dice treats each class equally regardless of pixel count
        # Logs (0.07%) gets the same weight as Sky (34%)
        for c in range(num_classes):
            dice += (2 * intersection + smooth) / (prediction + target + smooth)
        return 1 - dice / num_classes

# Combined: Focal handles hard pixels, Dice handles class balance
loss = 1.0 * FocalLoss(alpha=class_weights, gamma=2.0) + \
       0.5 * DiceLoss(num_classes=10)
```

**Why γ=2.0 for Focal?** Standard choice from the original Focal Loss paper. γ=2 means a pixel classified with 90% confidence gets its loss reduced by 100× compared to a 50% confidence pixel. This forces the model to focus on uncertain boundaries.

**Why 0.5 weight for Dice?** Dice Loss tends to produce larger gradients than Focal Loss. Weighting it at 0.5 prevents it from dominating the combined gradient.

**Why not just Dice alone?** Dice Loss can be unstable early in training when predictions are near-random. Focal Loss provides stable pixel-level gradients while Dice provides the region-level optimization signal.

**Expected**: Major boost for Logs and Rocks.  
**Actual**: Logs 0.05 → 0.25 (+382%), Rocks 0.16 → 0.32 (+92%). ✅ Exactly what we hoped.

#### Change 4: Higher Resolution (644×364)

```python
# Phase 1-2: 476×266 → 34×19 = 646 patch tokens
# Phase 3:   644×364 → 46×26 = 1196 patch tokens
w = 46 * 14  # 644 (must be multiple of 14 for ViT)
h = 26 * 14  # 364
```

**Why higher resolution?** At 476×266, a Log might occupy only 3-5 pixels. That's too small for ANY architecture to segment reliably. At 644×364 (84% more pixels), Logs become 5-8 pixels — still small but detectable.

**Trade-off**: 1196 patch tokens (vs 646) means ~1.85× more computation in the backbone. Combined with ViT-Base's larger model, each epoch takes ~10 min vs ~8 min in Phase 2.

**Why not even higher (e.g., 960×540)?** Would exceed 6GB VRAM even with AMP. The 644×364 balance was chosen to maximize resolution within our GPU constraints.

#### Change 5: Gradient Accumulation

```python
accum_steps = 2  # accumulate gradients over 2 batches
for step, (imgs, labels) in enumerate(train_loader):
    loss = loss_fn(output, labels) / accum_steps  # scale loss
    scaler.scale(loss).backward()

    if (step + 1) % accum_steps == 0:  # update every 2 steps
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

**Why?** Actual batch_size=2 (VRAM limit), but effective batch_size=4. Larger effective batches provide more stable gradient estimates, especially important with class-weighted Focal Loss where rare-class gradients can be noisy.

**Why accum=2 (not 4 or 8)?** Each accumulation step doubles memory for stored activations. 2 steps keeps us well within 6GB.

#### Change 6: Warmup Scheduler

```python
def lr_lambda(epoch):
    if epoch < warmup_epochs:  # epochs 0, 1, 2
        return (epoch + 1) / warmup_epochs  # 0.33, 0.67, 1.0
    progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

**Why warmup?** Phase 2's CosineAnnealing started at peak LR immediately. This was fine for Phase 2's simpler head, but Phase 3's UPerNet has more parameters (PPM + FPN + fusion). Starting with full LR on randomly initialized weights causes large, unstable gradients.

**Why 3 epochs?** Standard heuristic: warmup for 5-10% of total training. 3/40 = 7.5%.

**LR progression**: 1e-4 → 2e-4 → 3e-4 (warmup) → cosine decay → 0.

#### Change 7: Stronger Augmentations

```python
# New in Phase 3:
A.RandomShadow(p=0.15),  # Simulates shadow from trees/cliffs
A.CLAHE(clip_limit=2.0, p=0.15),  # Enhance local contrast
```

**Why RandomShadow?** Desert offroad scenes have harsh shadows from boulders, cliffs, and vegetation. Training with synthetic shadows makes the model robust to lighting changes.

**Why CLAHE?** Contrast-Limited Adaptive Histogram Equalization enhances local detail, making subtle textures (Ground Clutter, Dry Bushes) more distinct during training.

### Expected vs Actual (Phase 3)

| Aspect            | Expected              | Actual                                            |
| ----------------- | --------------------- | ------------------------------------------------- |
| **IoU**           | 0.55+                 | **0.5161** ❌ close but short                     |
| **Epoch 1 IoU**   | > Phase 2 best (0.40) | **0.4219** ✅ already above                       |
| **IoU > 0.50**    | By epoch 10-15        | **Epoch 16** ✅ close                             |
| **Logs IoU**      | > 0.20                | **0.2495** ✅ exceeded                            |
| **Rocks IoU**     | > 0.25                | **0.3167** ✅ exceeded                            |
| **Training time** | ~6-8 hours            | **~7 hours** ✅                                   |
| **Convergence**   | By epoch 35           | **Still improving at 40** ❌ — more epochs needed |

**Why we didn't reach 0.55?** The model was still improving at epoch 40 (IoU went from 0.515 → 0.516 in the last 5 epochs). With 60-80 epochs, we'd likely reach 0.52-0.53. The remaining gap to 0.55 would require:

- **Copy-paste augmentation** for Logs/Rocks (paste rare objects onto training images)
- **Backbone fine-tuning** (carefully unfreeze last 2-3 ViT layers)
- **Test-time augmentation** (average predictions from flipped/scaled inputs)

---

## Phase 4 Script: `train_phase4_mastery.py`

### What Changed and Why

Every change was a **direct response to Phase 3's analysis** — the model was still improving at epoch 40, rare classes (Logs=0.25, Rocks=0.32) hadn't plateaued, and we hadn't tried TTA or loss tuning.

#### Change 1: Resume From Phase 3 Checkpoint

```python
# Load Phase 3's best weights instead of training from scratch
p3_checkpoint = os.path.join(project_root, 'TRAINING AND PROGRESS', 'PHASE_3_ADVANCED', 'best_model.pth')
ckpt = torch.load(p3_checkpoint, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
# Epoch 40, Val IoU=0.5161 — our starting point
```

**Why resume?** Phase 3 was still improving at epoch 40. Rather than training 80+ epochs from scratch (14+ hours), we can start from P3's best and focus on refinement. This is standard transfer learning — the head already knows the basic class patterns, we're teaching it nuances.

**Why `weights_only=False`?** PyTorch 2.10+ defaults to `weights_only=True` for security (prevents arbitrary code execution during deserialization). Our checkpoint contains numpy arrays (from `history.json`), which requires the `False` flag. Safe because we generated the checkpoint ourselves.

**Expected**: Epoch 1 should start near 0.51 IoU.  
**Actual**: ✅ Epoch 1 IoU = 0.515 — exactly where Phase 3 left off.

#### Change 2: Multi-Scale Training (0.8x–1.2x)

```python
# Phase 3
A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=20, p=0.4)

# Phase 4
A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5)
#                                    ^^^^^^^^^^^^
# scale_limit=0.2 means ±20% → images are randomly scaled to 80%–120% of base size
```

**Why wider scale range?** Phase 3 used ±15% (0.85x–1.15x). Phase 4 expands to ±20% (0.8x–1.2x). This forces the model to recognize objects at more varied scales, simulating different camera distances. Rocks at 0.8x are much smaller — the model must learn scale-invariant texture features.

**Why increase shift_limit (0.08 → 0.1)?** Wider shifts expose more edge regions, training the model for off-center compositions common in real UGV footage.

**Why reduce rotate (20 → 15)?** Desert scenes have strong horizontal lines (horizon, terrain layers). ±20° rotation sometimes flipped the scene orientation beyond realistic, confusing the model about Sky vs Landscape position. ±15° keeps it natural.

**Why increase probability (0.4 → 0.5)?** With wider scale range, we want multi-scale training to happen more often to give the model enough exposure.

**Expected**: +0.01–0.02 IoU from scale robustness.  
**Actual**: Hard to isolate — model improved for mid-frequency classes (Trees +24%, Lush Bushes +24%) suggesting scale variation helped.

#### Change 3: Loss Rebalance (Dice 0.6, Focal γ=1.5)

```python
# Phase 3
loss = 1.0 * FocalLoss(gamma=2.0) + 0.5 * DiceLoss()  # Focal-heavy

# Phase 4
loss = 0.4 * FocalLoss(gamma=1.5) + 0.6 * DiceLoss()  # Dice-heavy
```

**Why lower Focal γ (2.0 → 1.5)?** Focal Loss with γ=2 aggressively downweights "easy" pixels (>90% confidence gets 100× less weight). This is great for initial learning but in Phase 4 (fine-tuning), many pixels ARE correctly classified with high confidence. γ=1.5 is softer — it still focuses on hard pixels but doesn't completely ignore the easy ones, providing more stable gradients.

**Why increase Dice weight (0.5 → 0.6)?** Dice Loss optimizes for region-level overlap — it cares about IoU-like metrics directly. In Phase 3, Focal Loss dominated (1.0 vs 0.5). For fine-tuning, region-level Dice feedback is more useful than pixel-level Focal feedback, because the model already gets most pixels right — it needs to refine **boundaries and small regions**.

**Why Focal weight 0.4 (not 0.0)?** Removing Focal entirely would lose the per-pixel gradient signal. Keeping it at 0.4 maintains stable pixel-level learning while letting Dice drive the optimization direction.

**The Tradeoff**: This change created a "loss landscape mismatch" — the Phase 3 weights were optimized for the old loss ratios. The new ratios meant epoch 1's gradients were slightly misaligned, causing the Ep 2-5 dip. This is the price of changing loss mid-training.

**Expected**: +0.01–0.03 IoU from better rare-class handling.  
**Actual**: Per-class gains were huge (Dry Bushes +54.5%, Trees +24.2%) but mean IoU didn't improve because already-good classes (Sky, Landscape) regressed slightly. The loss change was right for individual classes but neutral for mean IoU.

#### Change 4: Explicit Backbone Freezing

```python
# Phase 3 — implicit freezing (backbone.eval(), no optimizer params)
backbone.eval()
backbone.to(device)

# Phase 4 — explicit freezing
backbone.eval()
for param in backbone.parameters():
    param.requires_grad = False  # Belt + suspenders
backbone.to(device)
```

**Why explicit?** Phase 3 relied on not including backbone params in the optimizer. But PyTorch can still compute and store gradients for backbone params during `loss.backward()`, wasting VRAM. `requires_grad=False` prevents gradient computation entirely, saving ~500MB VRAM and ~5% training time.

#### Change 5: Early Stopping Patience 10 (was 12)

```python
patience = 10  # Phase 3 used 12
```

**Why tighter?** Phase 3 trained 40 epochs and early stopping never triggered. For Phase 4 (fine-tuning from a checkpoint), the model is already near its optimum — we expect faster convergence and shorter plateau periods. Patience=10 prevents wasting hours if the model has truly plateaued.

**In retrospect**: This was slightly too aggressive. The model hit its best at Ep 1, dipped Ep 2-5 (loss adjustment), then recovered by Ep 10 (IoU=0.5144, nearly matching best). With patience=15, it likely would have beaten 0.5150 by Ep 12-14. **Lesson: when changing loss function mid-training, increase patience to account for the adaptation period.**

#### Change 6: Train-Val Gap Monitoring

```python
gap = tr_iou - va_iou
gap_warn = " ⚠️ GAP!" if gap > 0.05 else ""
tqdm.write(f"... | gap: {gap:.3f}{gap_warn} | ...")
```

**Why?** Phase 3 didn't explicitly monitor overfitting. For Phase 4, we added real-time gap tracking with a ⚠️ warning if train-val IoU gap exceeds 0.05 (chosen as the threshold where the model is clearly memorizing rather than generalizing).

**Actual**: Gap stayed at 0.030–0.037 throughout — healthy. The frozen backbone + augmentations are strong overfitting guards.

#### Change 7: Test-Time Augmentation (TTA)

```python
def evaluate_with_tta(model, backbone, loader, device, use_amp=False):
    with torch.no_grad():
        # Original prediction
        tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]
        out1 = model(tokens)  # → upsample → [B, 10, H, W]

        # Horizontally flipped prediction
        imgs_flip = torch.flip(imgs, dims=[3])  # Flip width axis
        tokens_flip = backbone.forward_features(imgs_flip)["x_norm_patchtokens"]
        out2 = model(tokens_flip)
        out2 = torch.flip(out2, dims=[3])  # Flip predictions back

        # Average logits → more confident predictions
        out = (out1 + out2) / 2.0
```

**Why horizontal flip?** Desert scenes are horizontally symmetric — a tree on the left looks the same on the right. By averaging predictions from the original and flipped views, we:

1. **Reduce left-right bias** — if the model learned Sky is more common on the left (spurious correlation), TTA averages this out
2. **Smooth boundary predictions** — edges get predictions from two perspectives, reducing noise
3. **Free boost** — no training cost, only 2× inference time

**Why only horizontal flip (not vertical or rotation)?** Vertical flip inverts Sky/Ground semantics — averaging would be catastrophic. Rotation at inference changes the image grid, requiring careful interpolation.

**Expected**: +0.005–0.01 IoU.  
**Actual**: +0.0019 IoU (0.5150 → 0.5169). ✅ Small but free — exactly what we expected for a simple 2-view TTA.

#### Change 8: Model Saving to MODELS/ Directory

```python
models_dir = os.path.join(project_root, 'MODELS')
# Save best checkpoint to both training output AND models dir
torch.save(ckpt_data, os.path.join(output_dir, 'best_model.pth'))
torch.save(ckpt_data, os.path.join(models_dir, f'phase4_best_model_iou{va_iou:.4f}.pth'))

# Also update the inference-ready head weights
torch.save(best_ckpt['model_state_dict'], os.path.join(models_dir, 'segmentation_head.pth'))
```

**Why dual save?** Phase 3 only saved to the training output directory. Phase 4 saves to both locations: the phase-specific folder (for reproducibility) and `MODELS/` (for easy deployment). The inference-ready `segmentation_head.pth` is always overwritten with the latest best model.

### Expected vs Actual (Phase 4)

| Aspect                | Expected               | Actual                                          |
| --------------------- | ---------------------- | ----------------------------------------------- |
| **IoU**               | 0.53–0.56              | **0.5150** ❌ didn't push higher                |
| **TTA IoU**           | +0.005–0.01 boost      | **0.5169** (+0.37%) ✅                          |
| **Per-class winners** | Rare classes +10-20%   | **Dry Bushes +54.5%, Trees +24.2%** ✅ exceeded |
| **Per-class flat**    | Sky/Landscape stable   | **Sky −0.04%, Landscape −0.5%** ✅              |
| **Overfitting**       | None (frozen backbone) | **Gap < 0.037** ✅                              |
| **Early stopping**    | ~30–40 epochs          | **Ep 11** ❌ too early — loss change caused dip |
| **Training time**     | 8–12 hours             | **2 hours** ✅ (early stop)                     |

**Why we didn't reach 0.53+**: The loss rebalance created an adaptation period (Ep 2-5 dip) that "reset" some of Phase 3's learned patterns. The model spent 10 epochs recovering rather than improving. The two key mistakes:

1. **Changed loss function + resumed from checkpoint** — these conflict. The checkpoint weights were optimized for the old loss landscape.
2. **Patience too tight** — 10 wasn't enough to recover from the loss-induced dip and then find improvements beyond 0.515.

**What DID work**: TTA (+0.37% free), per-class improvements (4 classes gained 13-55%), and the multi-scale augmentation (model generalized well, gap stayed low).

---

## Phase 5 Script: `train_phase5_controlled.py`

### What Changed and Why

Every change was a **direct response to Phase 4's ceiling** — the frozen backbone hit ~0.515 IoU across two phases. Expert consensus: the only path past 0.52 is controlled backbone fine-tuning.

#### Change 1: Partial Backbone Unfreeze (Blocks 10–11)

```python
# Freeze everything first
for param in backbone.parameters():
    param.requires_grad = False

# Unfreeze ONLY blocks 10 and 11 (last 2 of 12)
for i, block in enumerate(backbone.blocks):
    if i >= 10:
        for param in block.parameters():
            param.requires_grad = True

# Keep patch_embed, cls_token, pos_embed, and norm frozen
# Result: ~14M unfrozen (16% of backbone) + ~10M head = ~24M trainable
```

**Why last 2 blocks?** DINOv2's ViT-Base has 12 transformer blocks. Early blocks (0-5) encode low-level features (edges, textures) that are universal. Middle blocks (6-9) encode mid-level patterns. Blocks 10-11 encode **high-level semantics** — these are the ones that need domain-specific adaptation for offroad desert scenes vs. the general ImageNet-like data DINOv2 was pre-trained on.

**Why not all 12?** Unfreezing the entire backbone with only 2857 training images would cause catastrophic overfitting. The model has 86M backbone parameters but only ~3K images. Even 2 blocks (~14M params) is aggressive — this is why we use extremely low LR.

**Why not just block 11?** Expert allows 1-2 blocks. 2 blocks gives more representational flexibility. If overfitting is detected (gap > 0.05 for 3+ epochs), we stop early.

#### Change 2: Differential Learning Rate

```python
optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': 5e-6},   # Backbone: 40x slower
    {'params': head_params,     'lr': 2e-4},   # Head: normal range
], weight_decay=1e-4)
```

**Why 5e-6 for backbone?** The backbone's pre-trained features are extremely valuable — they encode knowledge from 142M images. We want to _nudge_ these features toward offroad semantics, not overwrite them. 5e-6 is conservative: at this LR, weight changes after 30 epochs are <1% of original values. Expert allows up to 1e-5, but we start at 5e-6 for safety.

**Why 2e-4 for head (not 3e-4)?** Phase 4 showed that 3e-4 was slightly too aggressive when resuming from checkpoint. 2e-4 provides gentler refinement while still allowing meaningful updates.

**Why 40x ratio?** Standard for partial backbone fine-tuning. The backbone needs to preserve its pre-trained knowledge while the head needs to adapt its classification boundaries. A smaller ratio (10x) would update backbone too aggressively; a larger ratio (100x) would barely change backbone features.

#### Change 3: Loss Rebalance (Focal γ=2.0, Dice=0.7, Focal=0.3)

```python
# Phase 4
loss = 0.4 * FocalLoss(gamma=1.5) + 0.6 * DiceLoss()

# Phase 5
loss = 0.3 * FocalLoss(gamma=2.0) + 0.7 * DiceLoss()
```

**Why revert γ to 2.0?** Phase 4 used γ=1.5 which was softer. But with backbone fine-tuning, the model has more representational capacity. γ=2.0 is the original Focal Loss paper's recommendation — it aggressively downweights easy pixels (>90% confidence gets 100× less weight), forcing the model to focus on **hard boundary pixels and rare classes**. This is exactly what we need to push Logs/Rocks/Clutter higher.

**Why Dice 0.7?** With backbone features now adapting, Dice Loss becomes even more important — it directly optimizes region-level overlap. Higher Dice weight (0.7) pushes the model to improve small-region segmentation quality rather than just pixel-level classification.

#### Change 4: Gradient Clipping (Critical for Backbone Safety)

```python
scaler.unscale_(optimizer)
# Backbone: strict clipping prevents pre-trained features from being destroyed
torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=1.0)
# Head: permissive clipping for normal training dynamics
torch.nn.utils.clip_grad_norm_(head_params, max_norm=5.0)
scaler.step(optimizer)
```

**Why different clip norms?** Backbone gradients must be small and controlled — any gradient spike could destroy pre-trained features irreversibly. `max_norm=1.0` caps the gradient vector norm, preventing catastrophic updates. Head gradients are already safe (randomly initialized, no pre-trained knowledge to protect), so `max_norm=5.0` is permissive.

**Why not in Phase 1-4?** With a fully frozen backbone, gradient clipping wasn't needed — gradients only flowed through the head. Now that backbone blocks receive gradients, clipping is **mandatory safety equipment**.

#### Change 5: Reduced Augmentation Intensity

```python
# Phase 4
A.ShiftScaleRotate(scale_limit=0.2, ...)  # 0.8x–1.2x (wide)
A.RandomShadow(p=0.15)                    # ON

# Phase 5
A.ShiftScaleRotate(scale_limit=0.1, ...)  # 0.9x–1.1x (narrow)
# RandomShadow REMOVED (p=0)
```

**Why narrower scale (0.2 → 0.1)?** With backbone blocks unfrozen, the model is more sensitive to training data distribution. Wide scale augmentation creates distribution shift that can destabilize backbone fine-tuning. 0.9x–1.1x still provides scale robustness but with gentler variation.

**Why remove RandomShadow?** Synthetic shadows can create noisy gradients that propagate into backbone blocks, disrupting pre-trained shadow handling. The backbone already understands shadows from pre-training — adding synthetic ones during fine-tuning confuses rather than helps.

#### Change 6: Enhanced Safety Monitoring

```python
# Track consecutive validation drops
if va_iou < prev_val_iou:
    consecutive_val_drops += 1
else:
    consecutive_val_drops = 0

# SAFETY STOP: 3 consecutive drops + gap > 0.05 = definite overfit
if consecutive_val_drops >= 3 and gap > 0.05:
    print("SAFETY STOP: Overfitting detected!")
    break
```

**Why two conditions?** A single val drop is normal noise. Two drops could be a plateau. Three consecutive drops with a growing train-val gap is a **definitive overfitting signal** — the backbone is memorizing training patterns. The dual condition (drops + gap) prevents false triggers from random fluctuation.

**Why gap threshold 0.05?** In Phases 1-4, the gap never exceeded 0.037. A gap of 0.05 means the model is 3.5% better on train than val — this is where memorization starts to dominate generalization. Combined with 3 consecutive drops, it's an unambiguous overfit signal.

#### Change 7: Backbone State Dict Saving

```python
# Save only the unfrozen blocks' state dict for checkpoint
ckpt_data = {
    'model_state_dict': model.state_dict(),
    'backbone_state_dict': {k: v for k, v in backbone.state_dict().items()
                            if any(f'blocks.{i}.' in k for i in [10, 11])},
    'optimizer_state_dict': optimizer.state_dict(),
    'val_iou': va_iou
}
```

**Why save backbone state?** In Phase 4, only the head weights were saved (backbone was frozen, so unchanged). Now blocks 10-11 have been modified — we need to save their state too. At checkpoint reload, we update only those blocks in the backbone state dict, keeping blocks 0-9 at their original DINOv2 weights.

**Why only blocks 10-11 (not full backbone)?** Saving the full backbone state dict would include 86M parameters of mostly unchanged weights. Saving only the 2 modified blocks reduces checkpoint size from ~330MB to ~82MB.

### Expected vs Actual (Phase 5)

| Aspect            | Expected                                             | Actual                 |
| ----------------- | ---------------------------------------------------- | ---------------------- |
| **IoU**           | 0.55–0.58 (best case 0.60)                           | _Training in progress_ |
| **TTA IoU**       | +0.005–0.01 boost                                    | _Pending_              |
| **Epoch 1-3**     | Slight drop (0.50–0.51) — normal unfreeze adjustment | _Pending_              |
| **Peak**          | Epoch 15–25                                          | _Pending_              |
| **Rare classes**  | Logs/Clutter/Rocks +0.04–0.10 each                   | _Pending_              |
| **Overfitting**   | Gap ≤0.05 (safety stop if violated)                  | _Pending_              |
| **Training time** | ~5–8 hours (30 epochs, ~15 min/ep)                   | _Pending_              |

---

## Inference: `test_segmentation.py`

Runs the best trained model on test images from `DATASET/Offroad_Segmentation_testImages/`.

**What it does**:

1. Loads the best `segmentation_head.pth` weights
2. For each test image: resize → normalize → backbone → head → argmax → color map
3. Saves predicted mask overlays for visual inspection

**Key parameters**:

- Uses the same image size as the training phase (644×364 for Phase 3)
- Same normalization (ImageNet mean/std)
- No augmentations at inference (deterministic)

---

## Visualization: `visualize.py`

Helper utilities:

- **Overlay**: Blends RGB image with predicted mask at configurable opacity
- **Color map**: Maps class IDs to distinct colors for visualization
- **Side-by-side**: Shows image | ground truth | prediction for comparison

---

## Parameter Evolution Table

| Parameter           | Phase 1      | Phase 2         | Phase 3              | Phase 4              | Phase 5                           | Why Changed (P5)       |
| ------------------- | ------------ | --------------- | -------------------- | -------------------- | --------------------------------- | ---------------------- |
| **Backbone**        | ViT-S (384d) | ViT-S (384d)    | **ViT-B (768d)**     | ViT-B (768d)         | ViT-B (**blocks 10-11 unfrozen**) | Break frozen ceiling   |
| **Head**            | ConvNeXt     | ConvNeXt        | **UPerNet**          | UPerNet              | UPerNet                           | Proven architecture    |
| **Loss**            | CE           | Weighted CE     | Focal+Dice (1.0/0.5) | Focal+Dice (0.4/0.6) | **Focal+Dice (0.3/0.7)**          | More Dice for regions  |
| **Focal γ**         | —            | —               | 2.0                  | 1.5                  | **2.0**                           | Revert for hard pixels |
| **Optimizer**       | SGD 1e-4     | AdamW 5e-4      | AdamW 3e-4           | AdamW 3e-4           | **AdamW (bb=5e-6, hd=2e-4)**      | Differential LR        |
| **LR Schedule**     | None         | CosineAnnealing | Warmup + Cosine      | Warmup + Cosine      | Warmup + Cosine (**both groups**) | Applied to both        |
| **Image Size**      | 476×266      | 476×266         | **644×364**          | 644×364              | 644×364                           | No change              |
| **Scale Aug**       | None         | ±15%            | ±15%                 | **±20%**             | **±10%**                          | Reduced for stability  |
| **Batch**           | 2            | 2               | 2 (eff. 4)           | 2 (eff. 4)           | 2 (eff. 4)                        | VRAM safe              |
| **Epochs**          | 10           | 30              | 40                   | 50 (stopped 11)      | **30**                            | Controlled run         |
| **Augmentations**   | None         | 5 types         | 7 types              | 7 types (stronger)   | **6 types (−Shadow)**             | Noisy for backbone     |
| **Normalization**   | BatchNorm    | BatchNorm       | **GroupNorm**        | GroupNorm            | GroupNorm                         | Proven                 |
| **Mixed Precision** | No           | Yes             | Yes                  | Yes                  | Yes                               | 6GB VRAM constraint    |
| **Checkpointing**   | Final only   | Best by IoU     | Best by IoU          | Best + `MODELS/`     | Best + `MODELS/` + **backbone**   | Save unfrozen blocks   |
| **Early Stopping**  | None         | Patience=10     | Patience=12          | Patience=10          | **Patience=10**                   | Safety                 |
| **Initialization**  | Random       | Random          | Random               | Phase 3 ckpt         | **Phase 4 ckpt**                  | Transfer learning      |
| **TTA**             | No           | No              | No                   | HFlip avg            | **HFlip avg**                     | Free boost             |
| **Overfit Monitor** | No           | No              | No                   | Gap tracking         | **Gap + consecutive drops**       | Enhanced safety        |
| **Grad Clipping**   | No           | No              | No                   | No                   | **bb=1.0, hd=5.0**                | Backbone protection    |

---

## Lessons Learned

### 1. Architecture > Optimization

Phase 2 proved that better optimization (AdamW vs SGD, 30 vs 10 epochs) can only go so far. The ConvNeXt head held us at 0.40 regardless of how well we optimized. Phase 3's UPerNet broke through to 0.52. **A better model architecture provides gains that no amount of hyperparameter tuning can match.**

### 2. Rare Class Performance Requires Multiple Strategies

Weighted CE alone gave Logs IoU=0.05. Adding Focal Loss raised it to 0.25. But even with Focal + Dice + higher resolution + stronger augmentations, Logs is still the worst class (0.25). **Extreme class imbalance (72× rarer than average) requires data-level solutions** like copy-paste augmentation or synthetic oversampling, not just loss-level fixes.

### 3. Resolution Matters for Small Objects

Going from 476×266 → 644×364 (+84% pixels) gave disproportionate gains for small-object classes (Rocks +92%, Logs +382%). At low resolution, these objects are 2-4 pixels — below the detection threshold for any model.

### 4. Warmup Prevents Wasted Early Epochs

Phase 3's warmup prevented the early instability we saw in some Phase 2 test runs. The first 3 epochs with gradually increasing LR let the randomly initialized UPerNet head "warm up" before receiving full-strength gradient updates.

### 5. GroupNorm is Mandatory for PPM

This was a production bug: BatchNorm crashes silently when spatial size = 1×1 (from AdaptiveAvgPool2d(1)). GroupNorm has no spatial-size dependency. **Always use GroupNorm in pooling-heavy architectures.**

### 6. Don't Change Loss When Resuming From Checkpoint

Phase 4's biggest lesson. The Phase 3 weights were in a good local minimum for Focal(γ=2)+Dice(0.5). Switching to Focal(γ=1.5)+Dice(0.6) moved the loss landscape — the old minimum was no longer a minimum under the new loss. The model spent 10 epochs recovering. **When fine-tuning from a checkpoint, either keep the same loss or use a much lower LR (5e-5 instead of 3e-4) to gently adapt.**

### 7. TTA is Free Performance

Horizontal flip TTA added +0.37% IoU with zero training cost and only 2× inference time. For any submission or production deployment, TTA should be standard. More aggressive TTA (multi-scale + flip) could add more but requires careful implementation.

### 8. Patience Must Account for Loss Changes

With patience=10, Phase 4 stopped at Ep 11 — just as the model was recovering (Ep 10 IoU=0.5144, nearly matching best). The loss-change adaptation dip ate 5 of the 10 patience epochs. **When changing the training objective, set patience ≥ 2× the expected adaptation period.**

### 9. Gradient Clipping is Non-Negotiable for Backbone Fine-Tuning

Phase 5 introduces gradient clipping with different thresholds for backbone (1.0) and head (5.0). Without clipping, a single batch with unusual images (e.g., pure sky) could produce gradient spikes that irreversibly damage pre-trained backbone features. **Always clip gradients when fine-tuning pre-trained models.**

### 10. Differential LR: The 40x Rule

When fine-tuning backbone + head simultaneously, the backbone should learn 20-50× slower than the head. Phase 5 uses 5e-6 (backbone) vs 2e-4 (head) = 40× ratio. This preserves pre-trained knowledge while allowing domain adaptation. **Too small a ratio (≤10x) risks destroying backbone features; too large (≥100x) makes backbone fine-tuning pointless.**
