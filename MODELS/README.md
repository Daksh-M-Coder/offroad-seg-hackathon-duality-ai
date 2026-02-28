# Trained Model Weights

All trained model checkpoints from our 3-phase training journey.

## Files

| File                              | Phase   | Val IoU    | Size  | Description                                                                                                                                                                                            |
| --------------------------------- | ------- | ---------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `phase2_best_model_iou0.4036.pth` | Phase 2 | **0.4036** | 28MB  | Best checkpoint from Phase 2 (epoch 26). Contains model state dict + optimizer state + config. Uses DINOv2 ViT-Small backbone + ConvNeXt head.                                                         |
| `phase3_best_model_iou0.5161.pth` | Phase 3 | **0.5161** | 39MB  | ⭐ **Best overall model.** Best checkpoint from Phase 3 (epoch 40). Contains model state dict + optimizer state + config. Uses DINOv2 ViT-Base backbone + UPerNet head.                                |
| `segmentation_head.pth`           | Phase 3 | **0.5161** | 9.3MB | Head-only weights extracted from Phase 3 best checkpoint. This is what `test_segmentation.py` loads for inference. Smaller file because it only contains UPerNet head parameters (no optimizer state). |

## How to Use

### For Inference (test images)

```python
# test_segmentation.py loads segmentation_head.pth automatically
python TRAINING_SCRIPTS/test_segmentation.py
```

### For Resuming Training

```python
checkpoint = torch.load('MODELS/phase3_best_model_iou0.5161.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print(f"Resuming from epoch {checkpoint['epoch']}, IoU={checkpoint['val_iou']:.4f}")
```

## Why No Phase 1 Model?

Phase 1's baseline script only saved the final epoch weights (no best checkpoint). The final model (IoU=0.2971) has been superseded by Phase 2 and 3, so it isn't preserved here.
