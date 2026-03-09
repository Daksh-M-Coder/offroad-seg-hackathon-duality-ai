"""
TESTING INTERFACE — Offroad Segmentation Visual Tester
=======================================================
Gradio-based visual testing dashboard for the Phase 5 segmentation model.

Features:
  - Pre-loaded sample browser: 10 fixed samples per class (100 total)
  - Random class picker: pick any random image of a selected class instantly
  - Upload your own image for inference
  - Labeled segmentation overlay with class colour legend
  - Per-class IoU, Dice, Pixel Accuracy metrics for every prediction
  - Auto-saves every result as a timestamped MD report in TESTING_INTERFACE/RESULTS/

Run:
    cd "PIXEL IMG SEGMENTATION"
    venv\\Scripts\\python.exe TESTING_INTERFACE\\app.py
"""

import os
import sys
import time
import json
import random
import datetime
import textwrap
import threading

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch import nn

# ─── Resolve project root regardless of CWD ───────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(PROJECT_ROOT, "MODELS", "phase5_best_model_iou0.5294.pth")
TRAIN_DIR    = os.path.join(PROJECT_ROOT, "DATASET", "Offroad_Segmentation_Training_Dataset", "train")
VAL_DIR      = os.path.join(PROJECT_ROOT, "DATASET", "Offroad_Segmentation_Training_Dataset", "val")
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "RESULTS")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Class Config ─────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Logs", "Rocks", "Landscape", "Sky"
]
N_CLASSES = len(CLASS_NAMES)

# Vivid, distinct colours for each class (RGB)
COLOR_PALETTE = np.array([
    [20,  20,  20 ],   # 0  Background  – near-black
    [34,  139, 34 ],   # 1  Trees       – forest green
    [0,   220, 0  ],   # 2  Lush Bushes – lime
    [210, 180, 140],   # 3  Dry Grass   – tan
    [139, 90,  43 ],   # 4  Dry Bushes  – brown
    [128, 128, 0  ],   # 5  Grd Clutter – olive
    [205, 133, 63 ],   # 6  Logs        – peru
    [150, 150, 150],   # 7  Rocks       – grey
    [160, 82,  45 ],   # 8  Landscape   – sienna
    [135, 206, 250],   # 9  Sky         – light-sky-blue
], dtype=np.uint8)

# Mask pixel-value → class-id mapping
VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

# Image / token dimensions (must match Phase 5 training)
IMG_W, IMG_H = 644, 364          # 46×14, 26×14
TOKEN_W, TOKEN_H = 46, 26        # 644/14, 364/14
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Samples per class shown in the browser
SAMPLES_PER_CLASS = 10

# ─── Result counter (thread-safe) ─────────────────────────────────────────────
_result_lock   = threading.Lock()
_result_seq    = 0

def _next_seq():
    global _result_seq
    with _result_lock:
        _result_seq += 1
        return _result_seq


# ══════════════════════════════════════════════════════════════════════════════
# Model Architecture  (identical copy from train_phase5_controlled.py)
# ══════════════════════════════════════════════════════════════════════════════

class PPM(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        out_channels = in_channels // len(pool_sizes)
        num_groups   = min(32, out_channels)
        self.stages  = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(num_groups, out_channels),
                nn.ReLU(inplace=True),
            ) for ps in pool_sizes
        ])

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        out  = [x]
        for stage in self.stages:
            feat = F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=False)
            out.append(feat)
        return torch.cat(out, dim=1)


class UPerNetHead(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, tokenH, tokenW):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        gn = min(32, mid_channels)

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.GroupNorm(gn, mid_channels), nn.ReLU(inplace=True))

        self.ppm       = PPM(mid_channels, pool_sizes=(1, 2, 3, 6))
        ppm_out        = mid_channels + (mid_channels // 4) * 4

        self.bottleneck = nn.Sequential(
            nn.Conv2d(ppm_out, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(gn, mid_channels), nn.ReLU(inplace=True))

        self.fpn_conv1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(gn, mid_channels), nn.ReLU(inplace=True))
        self.fpn_conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(gn, mid_channels), nn.ReLU(inplace=True))
        self.fpn_conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(gn, mid_channels), nn.ReLU(inplace=True))

        self.fpn_fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 3, mid_channels, 1, bias=False),
            nn.GroupNorm(gn, mid_channels), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1))

        self.classifier = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x       = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x       = self.input_proj(x)
        x_ppm   = self.ppm(x)
        x_bn    = self.bottleneck(x_ppm)
        p1, p2, p3 = self.fpn_conv1(x_bn), self.fpn_conv2(x_bn), self.fpn_conv3(x_bn)
        return self.classifier(self.fpn_fuse(torch.cat([p1, p2, p3], dim=1)))


# ══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Loading DINOv2 ViT-Base backbone …")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg", verbose=False)
backbone.to(DEVICE).eval()

# Dummy forward to get embedding dim
with torch.no_grad():
    _dummy    = torch.zeros(1, 3, IMG_H, IMG_W).to(DEVICE)
    _tokens   = backbone.forward_features(_dummy)["x_norm_patchtokens"]
    N_EMB     = _tokens.shape[2]   # 768 for ViT-Base

print(f"  Backbone ready  |  embedding dim = {N_EMB}  |  device = {DEVICE}")

print(f"Loading Phase 5 segmentation head from {MODEL_PATH} …")
seg_head = UPerNetHead(N_EMB, 256, N_CLASSES, TOKEN_H, TOKEN_W).to(DEVICE)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
seg_head.load_state_dict(ckpt["model_state_dict"])

# Restore fine-tuned backbone blocks 10-11 if saved
if "backbone_state_dict" in ckpt:
    bb_state = backbone.state_dict()
    bb_state.update(ckpt["backbone_state_dict"])
    backbone.load_state_dict(bb_state)
    print("  ✓ Restored fine-tuned backbone blocks 10-11")

seg_head.eval()
print(f"  Model ready  |  best val IoU = {ckpt.get('val_iou', '?')}  |  epoch = {ckpt.get('epoch', '?')}")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Indexing — build a class → list[image_path, mask_path] map
# ══════════════════════════════════════════════════════════════════════════════

def _convert_mask(mask_arr):
    out = np.zeros_like(mask_arr, dtype=np.uint8)
    for raw, cls in VALUE_MAP.items():
        out[mask_arr == raw] = cls
    return out


CACHE_PATH = os.path.join(SCRIPT_DIR, "dataset_index_cache.json")
MIN_CLASS_PIXELS = 500   # min pixels of a class for the image to count for it

def _index_dataset(split_dir):
    img_dir  = os.path.join(split_dir, "Color_Images")
    mask_dir = os.path.join(split_dir, "Segmentation")
    if not os.path.isdir(img_dir):
        return {}
    class_index: dict[int, list[tuple[str, str]]] = {c: [] for c in range(N_CLASSES)}
    all_images = sorted(os.listdir(img_dir))
    for fname in all_images:
        mask_path = os.path.join(mask_dir, fname)
        img_path  = os.path.join(img_dir,  fname)
        if not os.path.isfile(mask_path):
            continue
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_raw is None:
            continue
        if len(mask_raw.shape) == 3:
            mask_raw = mask_raw[:, :, 0]
        mask_cls = _convert_mask(mask_raw)
        for c in range(N_CLASSES):
            if (mask_cls == c).sum() > MIN_CLASS_PIXELS:
                class_index[c].append((img_path, mask_path))
    return class_index


def _build_index_with_cache(train_dir, val_dir, cache_path):
    """Load class index from JSON cache if it exists, otherwise build and save it."""
    if os.path.isfile(cache_path):
        print(f"  ✓ Loading dataset index from cache: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # JSON keys are strings — convert back to int
        return {int(k): [tuple(p) for p in v] for k, v in raw.items()}

    print("  ⏳ Building index for the first time (this takes ~2-4 min, cached after) …")
    merged: dict[int, list[tuple[str, str]]] = {c: [] for c in range(N_CLASSES)}
    for split_dir in [train_dir, val_dir]:
        idx = _index_dataset(split_dir)
        for c in range(N_CLASSES):
            merged[c].extend(idx.get(c, []))

    # Deduplicate by image path
    deduped = {}
    for c in range(N_CLASSES):
        seen = set()
        deduped[c] = []
        for pair in merged[c]:
            if pair[0] not in seen:
                seen.add(pair[0])
                deduped[c].append(pair)

    # Save to JSON cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({str(k): [list(p) for p in v] for k, v in deduped.items()}, f)
    print(f"  ✓ Index saved to cache: {cache_path}")
    return deduped


# ── Startup: build or load class index ────────────────────────────────────────
print("Indexing dataset (or loading from cache) …")
_all_index = _build_index_with_cache(TRAIN_DIR, VAL_DIR, CACHE_PATH)

# Shuffle with fixed seed per class and split into full pool + fixed 10 browser samples
CLASS_INDEX:   dict[int, list] = {}
CLASS_SAMPLES: dict[int, list] = {}
for c in range(N_CLASSES):
    pool = _all_index.get(c, [])
    random.seed(42 + c)
    random.shuffle(pool)
    CLASS_INDEX[c]   = pool
    CLASS_SAMPLES[c] = pool[:SAMPLES_PER_CLASS]

total = sum(len(v) for v in CLASS_INDEX.values())
print(f"  Indexed {total} pairs across {N_CLASSES} classes")
for c in range(N_CLASSES):
    print(f"  [{c:2d}] {CLASS_NAMES[c]:<18} — {len(CLASS_SAMPLES[c]):>2} browser  | {len(CLASS_INDEX[c]):>4} total")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Core Inference
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess(img_rgb: np.ndarray) -> torch.Tensor:
    """Resize + normalise any RGB image to model input tensor."""
    img = cv2.resize(img_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    img = (img - np.array(MEAN)) / np.array(STD)
    return torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)


def _predict(tensor: torch.Tensor):
    """Run inference + TTA and return (logits, pred_mask_np)."""
    with torch.no_grad():
        tok1    = backbone.forward_features(tensor)["x_norm_patchtokens"]
        log1    = seg_head(tok1)
        out1    = F.interpolate(log1, size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)

        flip    = torch.flip(tensor, dims=[3])
        tok2    = backbone.forward_features(flip)["x_norm_patchtokens"]
        log2    = seg_head(tok2)
        out2    = torch.flip(F.interpolate(log2, size=(IMG_H, IMG_W), mode="bilinear", align_corners=False), dims=[3])

        logits  = (out1 + out2) / 2.0
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return logits, pred


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    h, w   = mask.shape
    rgb    = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(N_CLASSES):
        rgb[mask == c] = COLOR_PALETTE[c]
    return rgb


def _overlay(img_rgb: np.ndarray, mask_rgb: np.ndarray, alpha=0.55) -> np.ndarray:
    base   = cv2.resize(img_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    return (base * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)


# ─── per-class IoU / Dice from logits + GT mask tensor ───────────────────────
def _per_class_metrics(logits: torch.Tensor, gt: np.ndarray):
    pred   = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)
    gt_t   = gt.astype(np.int64)
    ious, dices, preds_px, gt_px = [], [], [], []
    for c in range(N_CLASSES):
        p = (pred == c)
        g = (gt_t  == c)
        inter  = (p & g).sum()
        union  = (p | g).sum()
        iou    = inter / union if union > 0 else float("nan")
        dice   = (2 * inter + 1e-6) / (p.sum() + g.sum() + 1e-6)
        ious.append(iou)
        dices.append(float(dice))
        preds_px.append(int(p.sum()))
        gt_px.append(int(g.sum()))
    pixel_acc = (pred == gt_t).mean()
    mean_iou  = float(np.nanmean(ious))
    return mean_iou, float(pixel_acc), ious, dices, preds_px, gt_px


# ─── legend figure ────────────────────────────────────────────────────────────
def _make_legend_figure():
    fig, ax = plt.subplots(figsize=(3.5, 4.5))
    patches = [mpatches.Patch(color=COLOR_PALETTE[c] / 255, label=CLASS_NAMES[c])
               for c in range(N_CLASSES)]
    ax.legend(handles=patches, loc="center", fontsize=10, frameon=False)
    ax.axis("off")
    ax.set_title("Class Legend", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# ─── metrics bar-chart figure ─────────────────────────────────────────────────
def _make_metrics_figure(ious):
    valid_ious = [v if not (isinstance(v, float) and np.isnan(v)) else 0.0 for v in ious]
    fig, ax = plt.subplots(figsize=(10, 4))
    colours = [COLOR_PALETTE[c] / 255 for c in range(N_CLASSES)]
    bars    = ax.bar(range(N_CLASSES), valid_ious, color=colours, edgecolor="black", linewidth=0.8)
    mean    = float(np.nanmean([v for v in ious if not (isinstance(v, float) and np.isnan(v))]))
    ax.axhline(mean, color="red", linestyle="--", linewidth=1.8, label=f"Mean IoU = {mean:.3f}")
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class IoU (TTA prediction)", fontweight="bold")
    for bar, v in zip(bars, valid_ious):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Result Logger  → RESULTS/<seq>_<timestamp>.md
# ══════════════════════════════════════════════════════════════════════════════

def _save_result_md(
    source_label: str,
    img_fname: str,
    mean_iou: float,
    pixel_acc: float,
    ious: list,
    dices: list,
    pred_px: list,
    gt_px: list,
    has_gt: bool,
):
    seq  = _next_seq()
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{seq:04d}_{ts}.md"
    path  = os.path.join(RESULTS_DIR, fname)

    lines = [
        f"# Result #{seq:04d}",
        f"",
        f"| Field | Value |",
        f"|---|---|",
        f"| **Timestamp** | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |",
        f"| **Source** | {source_label} |",
        f"| **Image** | `{img_fname}` |",
        f"| **Model** | Phase 5 — DINOv2 ViT-Base + UPerNet (IoU 0.5294, TTA 0.5310) |",
        f"| **Device** | {DEVICE} |",
        f"| **TTA** | ✅ HFlip average |",
        f"",
    ]

    if has_gt:
        lines += [
            f"## Overall Metrics (vs Ground Truth)",
            f"",
            f"| Metric | Value |",
            f"|---|---|",
            f"| **Mean IoU** | {mean_iou:.4f} |",
            f"| **Pixel Accuracy** | {pixel_acc:.4f} ({pixel_acc*100:.2f}%) |",
            f"",
            f"## Per-Class Breakdown",
            f"",
            f"| Class | IoU | Dice | Pred Pixels | GT Pixels |",
            f"|---|---|---|---|---|",
        ]
        for c in range(N_CLASSES):
            iou_str = f"{ious[c]:.4f}" if not (isinstance(ious[c], float) and np.isnan(ious[c])) else "N/A (absent)"
            lines.append(
                f"| **{CLASS_NAMES[c]}** | {iou_str} | {dices[c]:.4f} | {pred_px[c]:,} | {gt_px[c]:,} |"
            )
    else:
        lines += [
            f"## Prediction Only",
            f"> Ground-truth mask not available for this image (custom upload or test set).",
            f"",
            f"| Class | Predicted Pixels | % of Image |",
            f"|---|---|---|",
        ]
        total_px = IMG_W * IMG_H
        for c in range(N_CLASSES):
            lines.append(
                f"| **{CLASS_NAMES[c]}** | {pred_px[c]:,} | {pred_px[c]/total_px*100:.1f}% |"
            )

    lines += [
        f"",
        f"---",
        f"*Auto-generated by TESTING_INTERFACE/app.py — Offroad Segmentation Project*",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return fname


# ══════════════════════════════════════════════════════════════════════════════
# Main Inference Handler
# ══════════════════════════════════════════════════════════════════════════════

def _run_inference(img_path: str, mask_path: str | None, source_label: str):
    """
    Core inference function.
    Returns:
        original_img, overlay_img, mask_rgb_img, metrics_fig, metrics_text, result_md_fname
    """
    # ── Load image ──
    img_rgb  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_fname = os.path.basename(img_path)

    # ── Load GT mask if available ──
    gt_mask = None
    if mask_path and os.path.isfile(mask_path):
        raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if raw is not None:
            if len(raw.shape) == 3:
                raw = raw[:, :, 0]
            gt_resized = cv2.resize(_convert_mask(raw), (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
            gt_mask    = gt_resized

    # ── Inference ──
    tensor   = _preprocess(img_rgb)
    logits, pred_mask = _predict(tensor)

    # ── Visualisations ──
    mask_rgb   = _mask_to_rgb(pred_mask)
    overlay    = _overlay(img_rgb, mask_rgb, alpha=0.55)

    # Original resized for display
    original_display = cv2.resize(img_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)

    has_gt = gt_mask is not None

    # ── Metrics ──
    if has_gt:
        mean_iou, px_acc, ious, dices, pred_px, gt_px = _per_class_metrics(logits, gt_mask)
    else:
        # Prediction-only: just count pixels per class
        ious    = [float("nan")] * N_CLASSES
        dices   = [float("nan")] * N_CLASSES
        gt_px   = [0] * N_CLASSES
        pred_px = [int((pred_mask == c).sum()) for c in range(N_CLASSES)]
        mean_iou, px_acc = float("nan"), float("nan")

    # ── Metrics figure ──
    metrics_fig = _make_metrics_figure(ious)

    # ── Metrics text ──
    def _iou_str(v):
        return f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "absent"

    if has_gt:
        lines_m = [
            f"✅  Mean IoU    : {mean_iou:.4f}",
            f"✅  Pixel Acc  : {px_acc*100:.2f}%",
            f"",
            f"{'Class':<20} {'IoU':>7} {'Dice':>7} {'PredPx':>9} {'GTPx':>9}",
            "─" * 56,
        ]
        for c in range(N_CLASSES):
            lines_m.append(
                f"{CLASS_NAMES[c]:<20} {_iou_str(ious[c]):>7} {dices[c]:>7.4f} {pred_px[c]:>9,} {gt_px[c]:>9,}"
            )
    else:
        total_px = IMG_W * IMG_H
        lines_m = [
            "ℹ️  No ground-truth mask available — prediction coverage shown",
            "",
            f"{'Class':<20} {'Pred Pixels':>12} {'% Area':>8}",
            "─" * 42,
        ]
        for c in range(N_CLASSES):
            lines_m.append(
                f"{CLASS_NAMES[c]:<20} {pred_px[c]:>12,} {pred_px[c]/total_px*100:>7.1f}%"
            )

    metrics_text = "\n".join(lines_m)

    # ── Save result MD ──
    md_fname = _save_result_md(
        source_label, img_fname, mean_iou, px_acc,
        ious, dices, pred_px, gt_px, has_gt
    )
    result_info = f"💾 Saved → RESULTS/{md_fname}"

    return (
        Image.fromarray(original_display),
        Image.fromarray(overlay),
        Image.fromarray(mask_rgb),
        metrics_fig,
        metrics_text,
        result_info,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Gradio Callbacks
# ══════════════════════════════════════════════════════════════════════════════

def cb_fixed_sample(class_id: int, sample_idx: int):
    """Load one of the 10 fixed samples for a given class."""
    c     = int(class_id)
    idx   = int(sample_idx) % SAMPLES_PER_CLASS
    pairs = CLASS_SAMPLES.get(c, [])
    if not pairs:
        return [None] * 5 + ["❌ No samples indexed for this class."] + [None]
    img_path, mask_path = pairs[idx]
    label = f"Fixed Sample — {CLASS_NAMES[c]} #{idx+1}"
    return list(_run_inference(img_path, mask_path, label))


def cb_random_sample(class_id: int):
    """Pick any random image from the full pool for this class."""
    c     = int(class_id)
    pool  = CLASS_INDEX.get(c, [])
    if not pool:
        return [None] * 5 + ["❌ No samples for this class."] + [None]
    img_path, mask_path = random.choice(pool)
    label = f"Random Sample — {CLASS_NAMES[c]}"
    return list(_run_inference(img_path, mask_path, label))


def cb_upload(pil_image):
    """Run inference on a user-uploaded image (no GT mask)."""
    if pil_image is None:
        return [None] * 5 + ["❌ Please upload an image first."] + [None]
    # Save to temp
    tmp_path = os.path.join(RESULTS_DIR, "_uploaded_temp.png")
    pil_image.save(tmp_path)
    return list(_run_inference(tmp_path, None, "Custom Upload"))


# ══════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════════════════════════

LEGEND_FIG = _make_legend_figure()

with gr.Blocks(
    title="🏜️ Offroad Segmentation Tester",
    theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
) as demo:

    # ── Header ──
    gr.Markdown(
        """
        # 🏜️ Offroad Semantic Segmentation — Visual Testing Interface
        **Model**: Phase 5 — DINOv2 ViT-Base + UPerNet &nbsp;|&nbsp;
        **Best Val IoU**: 0.5294 &nbsp;|&nbsp; **TTA IoU**: 0.5310 &nbsp;|&nbsp;
        **Classes**: 10 offroad categories

        Use the tabs below to explore fixed class samples, pick random images, or upload your own.
        Every prediction is saved automatically to `TESTING_INTERFACE/RESULTS/`.
        """
    )

    # ── Output area (shared across all tabs) ──
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                out_original = gr.Image(label="📷 Original Image",      type="pil", height=260)
                out_overlay  = gr.Image(label="🎨 Segmentation Overlay", type="pil", height=260)
                out_mask     = gr.Image(label="🗺️ Prediction Mask",      type="pil", height=260)
        with gr.Column(scale=1):
            out_legend = gr.Plot(label="Class Legend", value=LEGEND_FIG)

    with gr.Row():
        with gr.Column(scale=2):
            out_metrics_fig  = gr.Plot(label="📊 Per-Class IoU Chart")
        with gr.Column(scale=1):
            out_metrics_text = gr.Textbox(label="🔢 Metrics Detail", lines=16, elem_classes=["metric-box"])

    out_saved = gr.Textbox(label="💾 Result Log", interactive=False)

    gr.Markdown("---")

    # ── Tabs ──
    with gr.Tabs():

        # ── Tab 1: Fixed Class Samples ────────────────────────────────────────
        with gr.Tab("📂 Class Samples (Fixed 10 per class)"):
            gr.Markdown(
                "Pick a **class** and a **sample index** (1–10). "
                "These are the same 10 fixed images every time — great for reproducible comparison."
            )
            with gr.Row():
                dd_class  = gr.Dropdown(
                    choices=[(f"{CLASS_NAMES[c]}  ({len(CLASS_SAMPLES[c])} samples)", c)
                             for c in range(N_CLASSES)],
                    value=0, label="🏷️ Class", scale=2
                )
                sl_sample = gr.Slider(1, SAMPLES_PER_CLASS, value=1, step=1,
                                      label="Sample #", scale=1)
            btn_fixed = gr.Button("▶ Run Segmentation", variant="primary")

            btn_fixed.click(
                cb_fixed_sample,
                inputs=[dd_class, sl_sample],
                outputs=[out_original, out_overlay, out_mask,
                         out_metrics_fig, out_metrics_text, out_saved],
            )

        # ── Tab 2: Random Class Sample ────────────────────────────────────────
        with gr.Tab("🎲 Random Class Sample"):
            gr.Markdown(
                "Pick a **class** and click **Random Pick** to get a random image "
                "from the full dataset pool. Each click gives a different image."
            )
            dd_class_rnd = gr.Dropdown(
                choices=[(f"{CLASS_NAMES[c]}  ({len(CLASS_INDEX[c])} available)", c)
                         for c in range(N_CLASSES)],
                value=0, label="🏷️ Class"
            )
            btn_random = gr.Button("🎲 Random Pick", variant="primary")

            btn_random.click(
                cb_random_sample,
                inputs=[dd_class_rnd],
                outputs=[out_original, out_overlay, out_mask,
                         out_metrics_fig, out_metrics_text, out_saved],
            )

        # ── Tab 3: Upload Your Own ────────────────────────────────────────────
        with gr.Tab("📤 Upload Your Own Image"):
            gr.Markdown(
                "Upload **any** offroad image. The model will segment it. "
                "No ground-truth metrics will be computed (no mask available), "
                "but you will see per-class pixel coverage."
            )
            in_upload  = gr.Image(label="Upload Image", type="pil", height=280)
            btn_upload = gr.Button("▶ Segment This Image", variant="primary")

            btn_upload.click(
                cb_upload,
                inputs=[in_upload],
                outputs=[out_original, out_overlay, out_mask,
                         out_metrics_fig, out_metrics_text, out_saved],
            )

    # ── Footer ──
    gr.Markdown(
        """
        ---
        **Offroad Semantic Scene Segmentation** · Duality AI × Ignitia Hackathon  
        Phase 5 Training: DINOv2 ViT-Base (blocks 10-11 unfrozen) + UPerNet · Focal γ=2.0 + Dice 0.7 · 30 epochs  
        Results auto-saved to `TESTING_INTERFACE/RESULTS/<seq>_<timestamp>.md`
        """
    )


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
