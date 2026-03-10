
---

# 1️⃣ Look at the Per-Class IoU Table

Your current weakest classes:

|Class|IoU|
|---|---|
|Ground Clutter|**0.2646**|
|Logs|**0.2975**|
|Rocks|**0.3403**|

Everything else is **0.45–0.97**.

So the **bottleneck is clear**:

```text
Small + visually ambiguous objects
```

Specifically:

- logs
    
- rocks
    
- ground clutter
    

These three classes are **dragging the mean IoU down**.

---

# 2️⃣ Why These Classes Are Hard

From your dataset description:

### Logs

- thin elongated objects
    
- few pixels
    
- similar color to ground
    

Result:

```text
small patch coverage → IoU penalty
```

---

### Rocks

- irregular shapes
    
- similar texture to soil
    

Result:

```text
rock ↔ ground confusion
```

---

### Ground Clutter

Worst class.

This class is basically **everything messy on ground**.

Typical problems:

- visually inconsistent
    
- overlaps with grass
    
- overlaps with rocks
    

Meaning the model has **no stable visual definition**.

---

# 3️⃣ Why Backbone Fine-Tuning Helped

Your Phase 5 result shows this clearly.

Biggest gains:

|Class|Gain|
|---|---|
|Logs|**+18.7%**|
|Rocks|**+7%**|
|Ground Clutter|**+4%**|

This proves:

```text
Backbone features were the bottleneck earlier
```

Now that is **mostly solved**.

But there is a **second bottleneck**.

---

# 4️⃣ The New Bottleneck: Object Boundaries

Look at your metrics:

```text
Pixel Accuracy ≈ 83.6%
IoU ≈ 53%
```

That gap means:

```text
Many pixels correct
BUT object shapes slightly wrong
```

Example:

Real log:

```
██████
```

Prediction:

```
 ████
```

Accuracy high, IoU lower.

This is classic **boundary error**.

---

# 5️⃣ Second Limitation: Patch Resolution

Your backbone:

DINOv2 ViT-Base with **14-pixel patches**.

Your image size:

```
644 × 364
```

Patch grid:

```
46 × 26 tokens
```

So a log may only be:

```
3–6 tokens wide
```

That’s **very coarse**.

So the model literally **cannot see fine edges well**.

This is a known limitation of ViT segmentation.

---

# 6️⃣ Third Limitation: Class Imbalance

Your dataset likely has:

```
sky + landscape + trees = majority pixels
logs + rocks = rare pixels
```

Even with focal loss:

Rare classes still struggle.

---

# 7️⃣ So the True Limiting Factors

From your report we can confidently say:

```text
1️⃣ Small object size (logs)
2️⃣ Texture similarity (rock vs ground)
3️⃣ Boundary precision errors
4️⃣ Patch resolution limits
```

Not:

❌ training instability  
❌ overfitting  
❌ optimizer issues

Your training pipeline is **healthy**.

---

# 8️⃣ So What Should Phase 6 Actually Target?

The strategy must attack those **four limitations**.

Not random changes.

---

### Move 1 — Better TTA

Add scale TTA:

```
0.9x
1.0x
1.1x
1.2x
flip
```

Expected:

```
+0.5–1.2 IoU
```

Safe.

---

### Move 2 — Sliding Window Inference

Instead of whole image prediction.

Predict tiles.

Benefits:

```
better small object detection
better patch alignment
```

Expected:

```
+1–2 IoU
```

Still zero training risk.

---

### Move 3 — Boundary Loss

Add:

```
Dice + Focal + Boundary
```

Targets:

```
object edges
```

Expected:

```
+1 IoU
```

---

### Move 4 — Slightly Expand Backbone Training

Current:

```
blocks 10–11
```

Phase 6 candidate:

```
blocks 9–11
```

But **very small LR**.

Because block 9 affects **shape formation**.

Expected:

```
+1–2 IoU
```

---

# 9️⃣ What I Would NOT Do

Based on your report:

❌ unfreeze 4 blocks  
❌ increase LR aggressively  
❌ change backbone architecture  
❌ heavy augmentation

All could break the stable training.

---

# 1️⃣0️⃣ Realistic Phase 6 Target

If you apply the safe improvements:

```
Current: 0.529
Phase 6 realistic: 0.55–0.58
```

Which is **very strong** for this dataset.

---

# Final Verdict (Important)

Your Phase 5 report proves:

```
model architecture = strong
training pipeline = stable
dataset signal = moderate
remaining errors = small-object + boundary
```

So Phase 6 should focus on:

```
better boundaries
better small-object detection
better inference
```

Not big architecture changes.

---

