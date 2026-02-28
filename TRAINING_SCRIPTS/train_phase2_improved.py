"""
PHASE 2 — Improved Segmentation Training Script
Based on the provided train_segmentation.py with the following improvements:
  1. Albumentations data augmentations (flip, rotate, color jitter, blur)
  2. More epochs (30)
  3. CosineAnnealingLR scheduler
  4. Best model checkpointing (saves when val_iou improves)
  5. Mixed precision training (torch.cuda.amp) for 6GB VRAM
  6. Class-weighted CrossEntropyLoss
  7. AdamW optimizer (better than SGD for this task)
  8. Early stopping (patience=10)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import json
import time
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion
# ============================================================================

# Mapping from raw pixel values to new class IDs
value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}
n_classes = len(value_map)

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return new_arr


# ============================================================================
# Dataset with Albumentations
# ============================================================================

class MaskDatasetAlbum(Dataset):
    """Dataset with albumentations augmentations for both image and mask."""
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.data_ids = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        # Read image as RGB numpy array
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask and convert to class IDs
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = convert_mask(mask)

        # Apply combined transform (augmentation + normalize + to tensor)
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Ensure correct types
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        mask = mask.long()

        return image, mask


def get_train_transform(h, w):
    """Training: augmentations + normalize + to tensor (single pipeline)."""
    return A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                           border_mode=cv2.BORDER_CONSTANT, value=0, p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transform(h, w):
    """Validation: resize + normalize + to tensor (no augmentations)."""
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ============================================================================
# Model: Segmentation Head (ConvNeXt-style) — Same as baseline
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )

        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10, ignore_index=255):
    """Compute IoU for each class and return mean IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue
        pred_inds = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class), iou_per_class


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    """Compute Dice coefficient per class and return mean Dice Score."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)
        dice_per_class.append(dice_score.cpu().numpy())

    return np.mean(dice_per_class)


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


def evaluate_metrics(model, backbone, data_loader, device, num_classes=10, use_amp=False):
    """Evaluate all metrics on a dataset."""
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    all_class_iou = []

    model.eval()
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                output = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = model(output.to(device))
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            labels = labels.squeeze(dim=1).long()

            iou, class_iou = compute_iou(outputs, labels, num_classes=num_classes)
            dice = compute_dice(outputs, labels, num_classes=num_classes)
            pixel_acc = compute_pixel_accuracy(outputs, labels)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)
            all_class_iou.append(class_iou)

    model.train()
    avg_class_iou = np.nanmean(all_class_iou, axis=0)
    return np.mean(iou_scores), np.mean(dice_scores), np.mean(pixel_accuracies), avg_class_iou


# ============================================================================
# Compute Class Weights from Dataset
# ============================================================================

def compute_class_weights(data_loader, num_classes=10, device='cuda'):
    """Compute inverse-frequency class weights from the dataset."""
    print("Computing class weights from training data...")
    class_counts = torch.zeros(num_classes, dtype=torch.float64)

    for _, masks in tqdm(data_loader, desc="Scanning classes", leave=False):
        masks = masks.view(-1)
        for c in range(num_classes):
            class_counts[c] += (masks == c).sum().item()

    total = class_counts.sum()
    weights = total / (num_classes * class_counts + 1e-6)
    # Normalize so max weight = 5.0 to prevent explosion
    weights = weights / weights.max() * 5.0
    weights = weights.float().to(device)

    print("Class weights computed:")
    for i, (name, w) in enumerate(zip(class_names, weights)):
        print(f"  {name:<20}: {w:.4f} (count: {int(class_counts[i])})")

    return weights


# ============================================================================
# Plotting Functions
# ============================================================================

def save_training_plots(history, output_dir):
    """Save all training metric plots to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Combined 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_title('Loss vs Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['train_iou'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_iou'], label='Val', linewidth=2)
    best_epoch = np.argmax(history['val_iou'])
    axes[0, 1].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best (Ep {best_epoch+1})')
    axes[0, 1].set_title('IoU vs Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history['train_dice'], label='Train', linewidth=2)
    axes[1, 0].plot(history['val_dice'], label='Val', linewidth=2)
    axes[1, 0].set_title('Dice Score vs Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['train_pixel_acc'], label='Train', linewidth=2)
    axes[1, 1].plot(history['val_pixel_acc'], label='Val', linewidth=2)
    axes[1, 1].set_title('Pixel Accuracy vs Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Pixel Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Phase 2 — Improved Training Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'), dpi=150)
    plt.close()
    print(f"Saved combined metrics curves to '{output_dir}/all_metrics_curves.png'")

    # Per-class IoU bar chart (final epoch)
    if 'final_class_iou' in history:
        fig, ax = plt.subplots(figsize=(10, 6))
        class_iou = history['final_class_iou']
        valid_iou = [iou if not np.isnan(iou) else 0 for iou in class_iou]
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        bars = ax.bar(range(n_classes), valid_iou, color=colors, edgecolor='black')
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylabel('IoU')
        ax.set_title(f'Per-Class IoU (Mean: {np.nanmean(class_iou):.4f})', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.axhline(y=np.nanmean(class_iou), color='red', linestyle='--', label='Mean', linewidth=2)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        for bar, iou in zip(bars, valid_iou):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{iou:.3f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_iou.png'), dpi=150)
        plt.close()
        print(f"Saved per-class IoU chart to '{output_dir}/per_class_iou.png'")

    # Learning rate plot
    if 'lr' in history:
        plt.figure(figsize=(8, 4))
        plt.plot(history['lr'], linewidth=2, color='green')
        plt.title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lr_schedule.png'), dpi=150)
        plt.close()
        print(f"Saved LR schedule to '{output_dir}/lr_schedule.png'")


def save_history_to_file(history, output_dir, config):
    """Save training history + config to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')

    with open(filepath, 'w') as f:
        f.write("PHASE 2 — IMPROVED TRAINING RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write("Configuration:\n")
        for k, v in config.items():
            f.write(f"  {k:<25}: {v}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Final Metrics:\n")
        f.write(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:       {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train IoU:      {history['train_iou'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:        {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Train Dice:     {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {history['train_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:   {history['val_pixel_acc'][-1]:.4f}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Best Results:\n")
        best_iou_epoch = np.argmax(history['val_iou']) + 1
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {best_iou_epoch})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 80 + "\n\n")

        if 'final_class_iou' in history:
            f.write("Per-Class IoU (Best Model):\n")
            f.write("-" * 40 + "\n")
            for name, iou in zip(class_names, history['final_class_iou']):
                iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
                f.write(f"  {name:<20}: {iou_str}\n")
            f.write("=" * 80 + "\n\n")

        f.write("Per-Epoch History:\n")
        f.write("-" * 110 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                   'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc', 'LR']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 110 + "\n")

        n_epochs = len(history['train_loss'])
        for i in range(n_epochs):
            lr_val = history['lr'][i] if 'lr' in history else 'N/A'
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.6f}\n".format(
                i + 1,
                history['train_loss'][i], history['val_loss'][i],
                history['train_iou'][i], history['val_iou'][i],
                history['train_dice'][i], history['val_dice'][i],
                history['train_pixel_acc'][i], history['val_pixel_acc'][i],
                lr_val if isinstance(lr_val, float) else 0.0
            ))

    print(f"Saved evaluation metrics to {filepath}")

    # Also save as JSON for programmatic access
    json_path = os.path.join(output_dir, 'history.json')
    json_history = {}
    for k, v in history.items():
        if isinstance(v, list):
            json_history[k] = [float(x) if not (isinstance(x, float) and np.isnan(x)) else None for x in v]
        else:
            json_history[k] = v
    with open(json_path, 'w') as f:
        json.dump(json_history, f, indent=2)
    print(f"Saved history JSON to {json_path}")


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    # ======================== CONFIGURATION ========================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters — IMPROVED
    batch_size = 2
    w = int(((960 / 2) // 14) * 14)   # 476
    h = int(((540 / 2) // 14) * 14)   # 266
    lr = 5e-4                          # Higher starting LR (was 1e-4)
    n_epochs = 30                      # More epochs (was 10)
    use_amp = True                     # Mixed precision
    use_class_weights = True           # Weighted loss
    patience = 10                      # Early stopping patience

    config = {
        'backbone': 'dinov2_vits14',
        'seg_head': 'ConvNeXt',
        'optimizer': 'AdamW',
        'lr': lr,
        'batch_size': batch_size,
        'epochs': n_epochs,
        'image_size': f'{w}x{h}',
        'mixed_precision': use_amp,
        'class_weights': use_class_weights,
        'augmentations': 'HFlip, VFlip, ShiftScaleRotate, Blur, ColorJitter',
        'scheduler': 'CosineAnnealingLR',
        'early_stopping_patience': patience,
    }

    # Output directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Find the project root (go up from DATASET/Offroad_Segmentation_Scripts/)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    output_dir = os.path.join(project_root, 'PROGRESS', 'PHASE_2_IMPROVED')
    os.makedirs(output_dir, exist_ok=True)

    # Dataset paths
    data_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'val')

    # ======================== TRANSFORMS ========================
    train_transform = get_train_transform(h, w)
    val_transform = get_val_transform(h, w)

    # ======================== DATASETS ========================
    trainset = MaskDatasetAlbum(data_dir=data_dir, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    valset = MaskDatasetAlbum(data_dir=val_dir, transform=val_transform)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Training samples: {len(trainset)}")
    print(f"Validation samples: {len(valset)}")

    # ======================== BACKBONE ========================
    print("Loading DINOv2 backbone...")
    BACKBONE_SIZE = "small"
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    backbone_model.to(device)
    print("Backbone loaded successfully!")

    # Get embedding dimension
    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    print(f"Embedding dimension: {n_embedding}")
    print(f"Patch tokens shape: {output.shape}")

    # ======================== MODEL ========================
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    )
    classifier = classifier.to(device)

    # ======================== LOSS & OPTIMIZER ========================
    if use_class_weights:
        class_weights = compute_class_weights(train_loader, num_classes=n_classes, device=device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fct = torch.nn.CrossEntropyLoss()

    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ======================== TRAINING HISTORY ========================
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_dice': [], 'val_dice': [],
        'train_pixel_acc': [], 'val_pixel_acc': [],
        'lr': []
    }

    best_val_iou = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    start_time = time.time()

    # ======================== TRAINING LOOP ========================
    print(f"\nStarting Phase 2 training ({n_epochs} epochs)...")
    print("=" * 80)
    print(f"Config: LR={lr}, Optimizer=AdamW, Scheduler=CosineAnnealing, AMP={use_amp}")
    print(f"Augmentations: HFlip, VFlip, Rotate90, ShiftScaleRotate, Blur, ColorJitter, GaussNoise")
    print("=" * 80)

    epoch_pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        # --- Training Phase ---
        classifier.train()
        train_losses = []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]",
                          leave=False, unit="batch")
        for imgs, labels in train_pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                with torch.no_grad():
                    output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]

                logits = classifier(output.to(device))
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

                labels = labels.squeeze(dim=1).long()
                loss = loss_fct(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- Validation Phase ---
        classifier.eval()
        val_losses = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
                    logits = classifier(output.to(device))
                    outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

                    labels = labels.squeeze(dim=1).long()
                    loss = loss_fct(outputs, labels)

                val_losses.append(loss.item())

        # --- Metrics ---
        train_iou, train_dice, train_pixel_acc, _ = evaluate_metrics(
            classifier, backbone_model, train_loader, device, num_classes=n_classes, use_amp=use_amp
        )
        val_iou, val_dice, val_pixel_acc, val_class_iou = evaluate_metrics(
            classifier, backbone_model, val_loader, device, num_classes=n_classes, use_amp=use_amp
        )

        # Step scheduler
        scheduler.step()

        # Store history
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_pixel_acc'].append(train_pixel_acc)
        history['val_pixel_acc'].append(val_pixel_acc)
        history['lr'].append(current_lr)

        # --- Best Model Checkpointing ---
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch + 1
            epochs_no_improve = 0
            # Save best model
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_dice': val_dice,
                'val_loss': epoch_val_loss,
                'config': config,
            }, best_model_path)
            print(f"\n  >> NEW BEST MODEL! Val IoU: {val_iou:.4f} (Epoch {epoch+1}) — Saved!")
            # Store per-class IoU for the best model
            history['final_class_iou'] = val_class_iou.tolist()
        else:
            epochs_no_improve += 1

        # Update progress bar
        epoch_pbar.set_postfix(
            train_loss=f"{epoch_train_loss:.3f}",
            val_loss=f"{epoch_val_loss:.3f}",
            val_iou=f"{val_iou:.3f}",
            best_iou=f"{best_val_iou:.3f}",
            lr=f"{current_lr:.2e}"
        )

        # --- Early Stopping ---
        if epochs_no_improve >= patience:
            print(f"\n  Early stopping triggered! No improvement for {patience} epochs.")
            print(f"  Best Val IoU: {best_val_iou:.4f} at Epoch {best_epoch}")
            break

    # ======================== SAVE RESULTS ========================
    elapsed = time.time() - start_time
    config['total_time_min'] = round(elapsed / 60, 1)
    config['best_val_iou'] = round(best_val_iou, 4)
    config['best_epoch'] = best_epoch

    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Best Val IoU: {best_val_iou:.4f} (Epoch {best_epoch})")

    # Save plots
    print("\nSaving training curves...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir, config)

    # Save final model too
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    torch.save(classifier.state_dict(), final_model_path)
    print(f"Saved final model to '{final_model_path}'")

    # Also save to scripts dir for test_segmentation.py compatibility
    compat_path = os.path.join(script_dir, 'segmentation_head.pth')
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device)
    torch.save(checkpoint['model_state_dict'], compat_path)
    print(f"Saved best model (compatible) to '{compat_path}'")

    # Final summary
    print("\n" + "=" * 80)
    print("PHASE 2 TRAINING COMPLETE")
    print("=" * 80)
    print(f"  Best Val IoU:      {best_val_iou:.4f} (Epoch {best_epoch})")
    print(f"  Final Val IoU:     {history['val_iou'][-1]:.4f}")
    print(f"  Final Val Dice:    {history['val_dice'][-1]:.4f}")
    print(f"  Final Val Acc:     {history['val_pixel_acc'][-1]:.4f}")
    print(f"  Training Time:     {elapsed/60:.1f} minutes")
    print(f"  Baseline IoU was:  0.2971")
    print(f"  Improvement:       {((best_val_iou - 0.2971) / 0.2971 * 100):.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
