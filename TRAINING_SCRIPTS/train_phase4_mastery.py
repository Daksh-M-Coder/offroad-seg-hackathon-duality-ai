"""
PHASE 4 — Mastery Training Script
Key improvements over Phase 3:
  1. Multi-scale training (0.8x–1.2x random resize via ShiftScaleRotate)
  2. Loss rebalance: Dice weight 0.6 (was 0.5), Focal gamma 1.5 (was 2.0)
  3. Resume from Phase 3 best checkpoint (transfer learning)
  4. Early stopping patience 10 (was 12)
  5. 50 epochs for deeper convergence
  6. Test-time augmentation (TTA) with horizontal flip
  Target: Val IoU 0.53–0.56+
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image
import cv2
import os
import json
import time
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

plt.switch_backend('Agg')

# ============================================================================
# Config
# ============================================================================

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}
n_classes = len(value_map)
class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return new_arr


# ============================================================================
# Dataset
# ============================================================================

class SegDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.data_ids = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = cv2.imread(os.path.join(self.image_dir, data_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.masks_dir, data_id), cv2.IMREAD_UNCHANGED)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = convert_mask(mask)

        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        return image, mask.long()


def get_train_transform(h, w):
    """Phase 4 train augmentations — adds multi-scale via larger scale_limit."""
    return A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        # Multi-scale: scale_limit=0.2 means 0.8x–1.2x random scaling
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15,
                           border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=25, p=1.0),
        ], p=0.4),
        A.RandomShadow(p=0.15),
        A.CLAHE(clip_limit=2.0, p=0.15),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transform(h, w):
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ============================================================================
# Losses: Focal (gamma=1.5) + Dice (weight=0.6)
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, num_classes=10):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        targets_onehot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        dice_total = 0.0
        for c in range(self.num_classes):
            p = probs[:, c]
            t = targets_onehot[:, c]
            intersection = (p * t).sum()
            dice_total += (2.0 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)

        return 1.0 - dice_total / self.num_classes


class CombinedLoss(nn.Module):
    def __init__(self, alpha=None, focal_weight=0.4, dice_weight=0.6, gamma=1.5, num_classes=10):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss(num_classes=num_classes)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        return self.focal_weight * self.focal(inputs, targets) + \
               self.dice_weight * self.dice(inputs, targets)


# ============================================================================
# UPerNet Segmentation Head (identical to Phase 3)
# ============================================================================

class PPM(nn.Module):
    """Pyramid Pooling Module — uses GroupNorm to avoid BatchNorm 1x1 failure."""
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        out_channels = in_channels // len(pool_sizes)
        num_groups = min(32, out_channels)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(num_groups, out_channels),
                nn.ReLU(inplace=True),
            ) for ps in pool_sizes
        ])

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        out = [x]
        for stage in self.stages:
            feat = stage(x)
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            out.append(feat)
        return torch.cat(out, dim=1)


class UPerNetHead(nn.Module):
    """UPerNet-style head with PPM and FPN. Uses GroupNorm throughout."""
    def __init__(self, in_channels, mid_channels, out_channels, tokenH, tokenW):
        super().__init__()
        self.H = tokenH
        self.W = tokenW
        gn = min(32, mid_channels)

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.GroupNorm(gn, mid_channels),
            nn.ReLU(inplace=True),
        )

        self.ppm = PPM(mid_channels, pool_sizes=(1, 2, 3, 6))
        ppm_out = mid_channels + (mid_channels // 4) * 4

        self.bottleneck = nn.Sequential(
            nn.Conv2d(ppm_out, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(gn, mid_channels),
            nn.ReLU(inplace=True),
        )

        self.fpn_conv1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(gn, mid_channels),
            nn.ReLU(inplace=True),
        )
        self.fpn_conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(gn, mid_channels),
            nn.ReLU(inplace=True),
        )
        self.fpn_conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(gn, mid_channels),
            nn.ReLU(inplace=True),
        )

        self.fpn_fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 3, mid_channels, 1, bias=False),
            nn.GroupNorm(gn, mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.classifier = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.input_proj(x)

        x_ppm = self.ppm(x)
        x_bn = self.bottleneck(x_ppm)

        p1 = self.fpn_conv1(x_bn)
        p2 = self.fpn_conv2(x_bn)
        p3 = self.fpn_conv3(x_bn)

        fused = torch.cat([p1, p2, p3], dim=1)
        fused = self.fpn_fuse(fused)

        return self.classifier(fused)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    iou_per_class = []
    for c in range(num_classes):
        pi, ti = pred == c, target == c
        inter = (pi & ti).sum().float()
        union = (pi | ti).sum().float()
        iou_per_class.append((inter / union).cpu().numpy() if union > 0 else float('nan'))
    return np.nanmean(iou_per_class), iou_per_class


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    dice_per_class = []
    for c in range(num_classes):
        pi, ti = pred == c, target == c
        inter = (pi & ti).sum().float()
        dice_per_class.append(((2 * inter + smooth) / (pi.sum().float() + ti.sum().float() + smooth)).cpu().numpy())
    return np.mean(dice_per_class)


def compute_pixel_accuracy(pred, target):
    return (torch.argmax(pred, dim=1) == target).float().mean().cpu().numpy()


def evaluate(model, backbone, loader, device, use_amp=False):
    model.eval()
    ious, dices, accs, all_cls = [], [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = model(tokens)
                out = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            labels = labels.squeeze(1).long()
            iou, cls_iou = compute_iou(out, labels)
            ious.append(iou)
            dices.append(compute_dice(out, labels))
            accs.append(compute_pixel_accuracy(out, labels))
            all_cls.append(cls_iou)
    model.train()
    return np.mean(ious), np.mean(dices), np.mean(accs), np.nanmean(all_cls, axis=0)


def evaluate_with_tta(model, backbone, loader, device, use_amp=False):
    """Evaluate with Test-Time Augmentation (horizontal flip + average)."""
    model.eval()
    ious, dices, accs, all_cls = [], [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                # Original prediction
                tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits1 = model(tokens)
                out1 = F.interpolate(logits1, size=imgs.shape[2:], mode="bilinear", align_corners=False)

                # Flipped prediction
                imgs_flip = torch.flip(imgs, dims=[3])
                tokens_flip = backbone.forward_features(imgs_flip)["x_norm_patchtokens"]
                logits2 = model(tokens_flip)
                out2 = F.interpolate(logits2, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                out2 = torch.flip(out2, dims=[3])  # Flip back

                # Average
                out = (out1 + out2) / 2.0

            labels = labels.squeeze(1).long()
            iou, cls_iou = compute_iou(out, labels)
            ious.append(iou)
            dices.append(compute_dice(out, labels))
            accs.append(compute_pixel_accuracy(out, labels))
            all_cls.append(cls_iou)
    model.train()
    return np.mean(ious), np.mean(dices), np.mean(accs), np.nanmean(all_cls, axis=0)


# ============================================================================
# Class Weights
# ============================================================================

def compute_class_weights(loader, num_classes=10, device='cuda'):
    print("Computing class weights...")
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, masks in tqdm(loader, desc="Scanning", leave=False):
        masks = masks.view(-1)
        for c in range(num_classes):
            counts[c] += (masks == c).sum().item()
    total = counts.sum()
    w = total / (num_classes * counts + 1e-6)
    w = w / w.max() * 5.0
    w = w.float().to(device)
    for name, wt in zip(class_names, w):
        print(f"  {name:<20}: {wt:.4f} ({int(counts[class_names.index(name)])} px)")
    return w


# ============================================================================
# Plotting
# ============================================================================

def save_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, key, title in [
        (axes[0,0], 'loss', 'Loss'), (axes[0,1], 'iou', 'IoU'),
        (axes[1,0], 'dice', 'Dice'), (axes[1,1], 'pixel_acc', 'Pixel Accuracy')
    ]:
        ax.plot(history[f'train_{key}'], label='Train', linewidth=2)
        ax.plot(history[f'val_{key}'], label='Val', linewidth=2)
        if key == 'iou':
            best = np.argmax(history['val_iou'])
            ax.axvline(x=best, color='red', linestyle='--', alpha=0.5, label=f'Best Ep {best+1}')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel(title)
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 4 — Mastery Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'), dpi=150)
    plt.close()

    # Per-class IoU
    if 'final_class_iou' in history:
        fig, ax = plt.subplots(figsize=(10, 6))
        vals = [x if not np.isnan(x) else 0 for x in history['final_class_iou']]
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        bars = ax.bar(range(n_classes), vals, color=colors, edgecolor='black')
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylabel('IoU'); ax.set_ylim(0, 1)
        mean_iou = np.nanmean(history['final_class_iou'])
        ax.axhline(y=mean_iou, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_iou:.3f}')
        ax.set_title(f'Per-Class IoU (Mean: {mean_iou:.4f})', fontweight='bold')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{v:.3f}', ha='center', fontsize=8)
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_iou.png'), dpi=150)
        plt.close()

    # LR schedule
    if 'lr' in history:
        plt.figure(figsize=(8, 4))
        plt.plot(history['lr'], linewidth=2, color='green')
        plt.title('Learning Rate Schedule (Phase 4)', fontweight='bold')
        plt.xlabel('Epoch'); plt.ylabel('LR')
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lr_schedule.png'), dpi=150)
        plt.close()

    print(f"Plots saved to {output_dir}")


def save_metrics(history, output_dir, config):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("PHASE 4 — MASTERY TRAINING RESULTS\n" + "="*80 + "\n\n")
        f.write("Configuration:\n")
        for k, v in config.items():
            f.write(f"  {k:<30}: {v}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou'])+1})\n")
        f.write(f"Final Val IoU:     {history['val_iou'][-1]:.4f}\n")
        f.write(f"Final Val Dice:    {history['val_dice'][-1]:.4f}\n")
        f.write(f"Final Val Acc:     {history['val_pixel_acc'][-1]:.4f}\n\n")

        if 'final_class_iou' in history:
            f.write("Per-Class IoU (Best Model):\n" + "-"*40 + "\n")
            for name, iou in zip(class_names, history['final_class_iou']):
                f.write(f"  {name:<20}: {iou:.4f}\n" if not np.isnan(iou) else f"  {name:<20}: N/A\n")
            f.write("="*80 + "\n\n")

        if 'tta_iou' in history:
            f.write(f"\nTTA Val IoU:       {history['tta_iou']:.4f}\n")
            f.write(f"TTA Val Dice:      {history['tta_dice']:.4f}\n")
            f.write(f"TTA Val Acc:       {history['tta_acc']:.4f}\n")
            f.write("TTA Per-Class IoU:\n" + "-"*40 + "\n")
            for name, iou in zip(class_names, history['tta_class_iou']):
                f.write(f"  {name:<20}: {iou:.4f}\n" if not np.isnan(iou) else f"  {name:<20}: N/A\n")
            f.write("="*80 + "\n\n")

        f.write("Per-Epoch History:\n" + "-"*120 + "\n")
        hdr = f"{'Ep':<5}{'TrLoss':<10}{'VaLoss':<10}{'TrIoU':<10}{'VaIoU':<10}{'TrDice':<10}{'VaDice':<10}{'TrAcc':<10}{'VaAcc':<10}{'LR':<12}\n"
        f.write(hdr + "-"*120 + "\n")
        for i in range(len(history['train_loss'])):
            lr = history['lr'][i] if 'lr' in history else 0
            f.write(f"{i+1:<5}{history['train_loss'][i]:<10.4f}{history['val_loss'][i]:<10.4f}"
                    f"{history['train_iou'][i]:<10.4f}{history['val_iou'][i]:<10.4f}"
                    f"{history['train_dice'][i]:<10.4f}{history['val_dice'][i]:<10.4f}"
                    f"{history['train_pixel_acc'][i]:<10.4f}{history['val_pixel_acc'][i]:<10.4f}"
                    f"{lr:<12.6f}\n")
    print(f"Metrics saved to {filepath}")

    # JSON
    jpath = os.path.join(output_dir, 'history.json')
    jh = {}
    for k, v in history.items():
        if isinstance(v, list):
            jh[k] = [float(x) if not (isinstance(x, float) and np.isnan(x)) else None for x in v]
        elif isinstance(v, (np.floating, float)):
            jh[k] = float(v)
        else:
            jh[k] = v
    with open(jpath, 'w') as f:
        json.dump(jh, f, indent=2)


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Config ──
    batch_size = 2
    accum_steps = 2            # gradient accumulation → effective batch=4
    w = 46 * 14  # 644
    h = 26 * 14  # 364
    lr = 3e-4
    n_epochs = 50
    warmup_epochs = 3
    patience = 10
    use_amp = True

    config = {
        'phase': 'Phase 4 — Mastery',
        'backbone': 'dinov2_vitb14_reg (ViT-Base, frozen)',
        'seg_head': 'UPerNet (PPM + multi-scale FPN)',
        'loss': 'Focal (gamma=1.5, weight=0.4) + Dice (weight=0.6)',
        'optimizer': 'AdamW (wd=1e-4)',
        'lr': f'{lr} + warmup({warmup_epochs}ep) + CosineAnnealing',
        'batch_size': f'{batch_size} (effective {batch_size * accum_steps} with grad accum)',
        'epochs': n_epochs,
        'image_size': f'{w}x{h}',
        'mixed_precision': use_amp,
        'augmentations': 'HFlip, VFlip, MultiScale(0.8-1.2x), Blur, ColorJitter, RandomShadow, CLAHE',
        'early_stopping': patience,
        'resume_from': 'Phase 3 best checkpoint (IoU=0.5161)',
        'tta': 'HorizontalFlip average at eval',
    }
    print("Config:", json.dumps(config, indent=2))

    # ── Paths ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    output_dir = os.path.join(project_root, 'TRAINING AND PROGRESS', 'PHASE_4_MASTERY')
    models_dir = os.path.join(project_root, 'MODELS')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    data_dir = os.path.join(project_root, 'DATASET', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(project_root, 'DATASET', 'Offroad_Segmentation_Training_Dataset', 'val')

    # ── Data ──
    trainset = SegDataset(data_dir, get_train_transform(h, w))
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valset = SegDataset(val_dir, get_val_transform(h, w))
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Train: {len(trainset)}, Val: {len(valset)}")

    # ── Backbone ──
    print("Loading DINOv2 ViT-Base backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False  # Explicitly frozen
    backbone.to(device)
    print("Backbone loaded and frozen!")

    # Get embedding dim
    sample_img, _ = next(iter(train_loader))
    with torch.no_grad():
        tokens = backbone.forward_features(sample_img.to(device))["x_norm_patchtokens"]
    n_emb = tokens.shape[2]
    tokenH, tokenW = h // 14, w // 14
    print(f"Embedding: {n_emb}, Tokens: {tokens.shape} ({tokenH}x{tokenW})")

    # ── Model ──
    model = UPerNetHead(
        in_channels=n_emb,
        mid_channels=256,
        out_channels=n_classes,
        tokenH=tokenH,
        tokenW=tokenW
    ).to(device)

    # ── Resume from Phase 3 ──
    p3_checkpoint = os.path.join(project_root, 'TRAINING AND PROGRESS', 'PHASE_3_ADVANCED', 'best_model.pth')
    if os.path.exists(p3_checkpoint):
        print(f"Loading Phase 3 checkpoint: {p3_checkpoint}")
        ckpt = torch.load(p3_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded from Epoch {ckpt.get('epoch', '?')}, Val IoU: {ckpt.get('val_iou', '?')}")
    else:
        print("WARNING: Phase 3 checkpoint not found! Training from scratch.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ── Loss ──
    class_weights = compute_class_weights(train_loader, n_classes, device)
    loss_fn = CombinedLoss(alpha=class_weights, focal_weight=0.4, dice_weight=0.6, gamma=1.5, num_classes=n_classes)

    # ── Optimizer + Scheduler ──
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Warmup + cosine
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ── History ──
    history = {k: [] for k in ['train_loss','val_loss','train_iou','val_iou',
                                'train_dice','val_dice','train_pixel_acc','val_pixel_acc','lr']}
    best_iou, best_epoch, no_improve = 0.0, 0, 0
    start_time = time.time()

    # ── Training ──
    print(f"\nPhase 4 Mastery Training: {n_epochs} epochs, {w}x{h}, ViT-Base, UPerNet")
    print(f"  Focal(gamma=1.5, w=0.4) + Dice(w=0.6) | Multi-scale 0.8x-1.2x")
    print(f"  Resume from Phase 3 (IoU=0.5161)")
    print("="*80)

    for epoch in tqdm(range(n_epochs), desc="Training", unit="epoch"):
        model.train()
        train_losses = []
        optimizer.zero_grad()

        for step, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)):
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                with torch.no_grad():
                    tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = model(tokens)
                out = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                labels = labels.squeeze(1).long()
                loss = loss_fn(out, labels) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_losses.append(loss.item() * accum_steps)

        # ── Validation ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]
                    logits = model(tokens)
                    out = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                    labels = labels.squeeze(1).long()
                    loss = loss_fn(out, labels)
                val_losses.append(loss.item())

        # ── Metrics ──
        tr_iou, tr_dice, tr_acc, _ = evaluate(model, backbone, train_loader, device, use_amp)
        va_iou, va_dice, va_acc, va_cls = evaluate(model, backbone, val_loader, device, use_amp)

        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        tr_loss, va_loss = np.mean(train_losses), np.mean(val_losses)
        for k, v in [('train_loss',tr_loss),('val_loss',va_loss),('train_iou',tr_iou),('val_iou',va_iou),
                      ('train_dice',tr_dice),('val_dice',va_dice),('train_pixel_acc',tr_acc),('val_pixel_acc',va_acc),('lr',cur_lr)]:
            history[k].append(v)

        # ── Checkpoint ──
        if va_iou > best_iou:
            best_iou, best_epoch, no_improve = va_iou, epoch + 1, 0
            # Save to output dir
            ckpt_data = {
                'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': va_iou, 'config': config
            }
            torch.save(ckpt_data, os.path.join(output_dir, 'best_model.pth'))
            # Also save to MODELS/
            torch.save(ckpt_data, os.path.join(models_dir, f'phase4_best_model_iou{va_iou:.4f}.pth'))
            history['final_class_iou'] = va_cls.tolist()
            print(f"\n  >> NEW BEST! IoU: {va_iou:.4f} (Ep {epoch+1})")
        else:
            no_improve += 1

        # Train-val gap monitoring
        gap = tr_iou - va_iou
        gap_warn = " ⚠️ GAP!" if gap > 0.05 else ""
        tqdm.write(f"  Ep {epoch+1:>2} | loss: {tr_loss:.3f}/{va_loss:.3f} | IoU: {tr_iou:.3f}/{va_iou:.3f} | gap: {gap:.3f}{gap_warn} | best: {best_iou:.3f} | lr: {cur_lr:.2e}")

        if no_improve >= patience:
            print(f"\n  Early stopping! Best IoU: {best_iou:.4f} at Ep {best_epoch}")
            break

    # ── TTA Evaluation ──
    print("\n  Running TTA evaluation on best model...")
    best_ckpt = torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    tta_iou, tta_dice, tta_acc, tta_cls = evaluate_with_tta(model, backbone, val_loader, device, use_amp)
    print(f"  TTA Results — IoU: {tta_iou:.4f}, Dice: {tta_dice:.4f}, Acc: {tta_acc:.4f}")
    history['tta_iou'] = tta_iou
    history['tta_dice'] = tta_dice
    history['tta_acc'] = tta_acc
    history['tta_class_iou'] = tta_cls.tolist()

    # ── Save ──
    elapsed = time.time() - start_time
    config['total_time_min'] = round(elapsed / 60, 1)
    config['best_val_iou'] = round(best_iou, 4)
    config['best_epoch'] = best_epoch
    config['tta_val_iou'] = round(float(tta_iou), 4)

    save_plots(history, output_dir)
    save_metrics(history, output_dir, config)

    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))

    # Save head-only weights for inference
    compat_path = os.path.join(models_dir, 'segmentation_head.pth')
    torch.save(best_ckpt['model_state_dict'], compat_path)
    # Also save in scripts dir for test_segmentation.py
    scripts_path = os.path.join(project_root, 'DATASET', 'Offroad_Segmentation_Scripts', 'segmentation_head.pth')
    torch.save(best_ckpt['model_state_dict'], scripts_path)

    print(f"\n{'='*80}")
    print("PHASE 4 MASTERY COMPLETE")
    print(f"  Best IoU:        {best_iou:.4f} (Ep {best_epoch})")
    print(f"  TTA IoU:         {tta_iou:.4f}")
    print(f"  Phase 1 baseline: 0.2971")
    print(f"  Phase 2 improved: 0.4036")
    print(f"  Phase 3 advanced: 0.5161")
    print(f"  Phase 4 mastery:  {best_iou:.4f}")
    print(f"  Total improvement: {((best_iou-0.2971)/0.2971*100):.1f}% over baseline")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"\n  Models saved to: {models_dir}")
    print(f"  Logs saved to:   {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
