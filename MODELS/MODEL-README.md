# Trained Model Weights

This directory contains the best checkpoints from our 6-phase training journey on the **Offroad Semantic Scene Segmentation** task.

> Total reduction in error: **86.0%** (IoU 0.2971 → 0.5527 TTA)

## Overview of Checkpoints

| File | Phase | Val IoU | TTA IoU | Size | Architecture / Highlights |
|---|---|---|---|---|---|
| `phase2_best_model_iou0.4036.pth` | 2 | 0.4036 | — | 28 MB | DINOv2 ViT-Small (frozen) + ConvNeXt. AdamW + Augs + Class Weights. |
| `phase3_best_model_iou0.5161.pth` | 3 | 0.5161 | — | 39 MB | DINOv2 ViT-Base (frozen) + UPerNet. Focal+Dice Loss. 644×364. |
| `phase4_best_model_iou0.5150.pth` | 4 | 0.5150 | 0.5169 | 39 MB | Same arch. Multi-scale augs + Rebalanced loss weights. |
| `phase5_best_model_iou0.5294.pth` | 5 | 0.5294 | 0.5310 | 201 MB | ⭐ **First Unfreeze**. Blocks 10-11 unfrozen with differential LR. |
| `phase6_best_model_iou0.5368.pth` | 6 | **0.5368** | **0.5527** | 282 MB | 🏆 **Best Overall**. Blocks 9-11 unfrozen + Boundary Loss + Multi-Scale TTA. |

### Where is Phase 1?
The Phase 1 script (provided baseline) only saved the final epoch, not the best validation checkpoint. Because it scored the lowest (IoU 0.2971) and was immediately superseded by Phase 2, we did not retain its 100MB file to save repository space.

---

## What is inside a checkpoint?

Starting from Phase 2, every `.pth` file is a dictionary containing all states needed for both inference and resuming training:

```python
checkpoint = {
    'epoch': 28,
    'model_state_dict': { ... },       # Model weights (backbone + head)
    'optimizer_state_dict': { ... },   # AdamW momentum/variance
    'val_iou': 0.5368,
    'val_dice': 0.5977,
    'val_acc': 73.15,
    'config': {                        # Hyperparameters used for this run
        'resolution': (364, 644),
        'batch_size': 2,
        'learning_rate_head': 0.0002,
        'learning_rate_backbone': 0.000004
    }
}
```

> **Why did the size increase in Phase 5 and 6?**  
> In Phases 1-4, the backbone was fully frozen, meaning the optimizer only tracked gradients for the UPerNet head (~39MB). In Phase 5, we unfrozen blocks 10-11. In Phase 6, we unfroze blocks 9-11 and added Boundary Loss structures, causing the optimizer state dictionary to expand significantly as it now tracks momentum and variance for millions of backbone parameters.

---

## How to Use for Inference

The visual dashboard (`TESTING_INTERFACE/app.py`) handles model loading automatically. If you want to load the Phase 6 model programmatically:

```python
import torch
import torch.nn as nn
from transformers import Dinov2Model

# 1. Initialize empty architecture
# (Must match the architecture used in Phase 3-6)
backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")
# ... initialize UPerNet head ... 

class FullModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

model = FullModel(backbone, upernet_head)

# 2. Load the checkpoint
checkpoint = torch.load('MODELS/phase6_best_model_iou0.5368.pth', map_location='cuda')

# 3. Handle the backbone state dictionary
# Phase 6 unfreezes blocks 9-11. The checkpoint therefore contains updated 
# weights for blocks 9, 10, and 11. The core DINOv2 weights for blocks 0-8 
# are left unchanged and rely on 'facebook/dinov2-base'.
current_backbone_state = model.backbone.state_dict()
current_backbone_state.update(checkpoint['backbone_state_dict']) # Override blocks 9-11
model.backbone.load_state_dict(current_backbone_state)

# 4. Load the head state dictionary
model.head.load_state_dict(checkpoint['model_state_dict'])

model.eval()
print(f"Loaded Phase 6 model! (Val IoU: {checkpoint['val_iou']:.4f})")
```

*(For exact implementation details including the UPerNet head class and Multi-Scale Test-Time Augmentation (TTA), see `TESTING_INTERFACE/app.py` or `TRAINING_SCRIPTS/train_phase6_boundary.py`)*
