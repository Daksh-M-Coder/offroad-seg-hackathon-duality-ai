# Project Management — Offroad Semantic Scene Segmentation

### How This Project Was Organised, Protected, and Kept Consistent Across 6 Phases

---

> **Who this document is for**: Anyone reading this project — judges, collaborators, or future-you.  
> **What it covers**: Not code. Not maths. Just: _how we managed the project as a living system._

---

## 📌 The Core Philosophy

> **"Every phase is a permanent record. Nothing is ever deleted, overwritten, or lost."**

From day one, the project was treated like a professional ML pipeline where:

1. **Every training run** produces a complete, self-contained output folder.
2. **Every model checkpoint** is named with its exact IoU score.
3. **Every decision** — even wrong ones — is documented.
4. **Backups exist** before any new phase touches the model.
5. **Folder structure is predictable** — anyone can navigate it without asking.

---

## 🗂️ Folder Structure — The Rules

### Top-Level Layout

The project root contains exactly these folders, each with a single, clear responsibility:

| Folder                   | Responsibility                                  |
| ------------------------ | ----------------------------------------------- |
| `BASIC_INTRO/`           | Background reading, theory, hackathon rules     |
| `TRAINING_SCRIPTS/`      | **All training code** — one script per phase    |
| `MODELS/`                | **Best model checkpoints only** — one per phase |
| `DATASET/`               | Raw data — never modified                       |
| `DATA_INSIGHTS/`         | Dataset analysis reports                        |
| `TRAINING AND PROGRESS/` | Training outputs — one subfolder per phase      |
| `SYSTEM_CHECK/`          | Hardware verification scripts                   |
| `ENV_SETUP/`             | Environment setup scripts and project docs      |
| `TESTING_INTERFACE/`     | Gradio visual testing app                       |
| `BASIC_INTRO/`           | Phase planning documents                        |

### The Rule: **One Folder Per Phase**

Each training phase gets its own folder under `TRAINING AND PROGRESS/`:

```
TRAINING AND PROGRESS/
├── PHASE_1_BASELINE/
├── PHASE_2_IMPROVED/
├── PHASE_3_ADVANCED/
├── PHASE_4_MASTERY/
├── PHASE_5_CONTROLLED/
└── PHASE_6_BOUNDARY/      ← created before training starts
```

**Why**: If all phases dumped output into the same folder, a single overwrite would destroy months of training history. Separate folders mean every phase is hermetically sealed.

### The Rule: **Create Output Folders Before Running**

Every phase folder is created **before** training starts — not during. This way:

- There is never confusion about where outputs go.
- The folder exists as a placeholder showing the phase was planned.
- If training crashes halfway, partial results are still organised.

---

## 🏆 Model Checkpoint Integrity — The Rules

### Rule 1: **IoU in the Filename — Always**

Every model saved to `MODELS/` has its validation IoU embedded in the filename:

```
Models/
├── phase2_best_model_iou0.4036.pth
├── phase3_best_model_iou0.5161.pth
├── phase4_best_model_iou0.5150.pth
├── phase5_best_model_iou0.5294.pth   ← current best
└── (phase6 will appear here after training)
```

**Why**: You can glance at the `MODELS/` folder and instantly know the performance of every saved model without opening a single file.

### Rule 2: **Two Copies — One Local, One Global**

Every best model is saved to **two locations**:

| Location                                         | Purpose                                        |
| ------------------------------------------------ | ---------------------------------------------- |
| `TRAINING AND PROGRESS/PHASE_X_Y/best_model.pth` | Phase-specific checkpoint — no name confusion  |
| `MODELS/phaseX_best_model_iouX.XXXX.pth`         | Global collection — all phases visible at once |

This means even if an entire phase folder is accidentally deleted, the model survives in `MODELS/`.

### Rule 3: **Make a Backup Before Starting a New Phase**

Before Phase 6 training began, the Phase 5 best model was manually backed up:

```
phase5_best_model_iou0.5294 - Backup.pth
```

**Why**: Phase 6 resumes FROM Phase 5's weights. If Phase 6 destabilises the model (overfit, bad LR, etc.), we reload the backup and start fresh — losing nothing.

### Rule 4: **Checkpoint Contains Everything Needed to Resume**

Every saved checkpoint file contains:

```
epoch            ← which epoch produced this
model_state_dict ← UPerNet head weights
backbone_state_dict ← unfrozen backbone blocks only
optimizer_state_dict ← optimizer momentum state
val_iou          ← the metric that made this the best
config           ← the exact hyperparameters used
```

**Why**: A checkpoint file should be completely self-describing. Six months from now, loading it should tell you everything about the training run that created it — with zero need to check old logs.

---

## 📝 File Naming Conventions — The Rules

### Training Scripts

```
train_phase1_baseline.py
train_phase2_improved.py
train_phase3_advanced.py
train_phase4_mastery.py
train_phase5_controlled.py
train_phase6_boundary.py
```

**Pattern**: `train_phase{N}_{descriptive_keyword}.py`

The keyword describes the _key innovation_ of that phase:

- `baseline` — original, unmodified
- `improved` — basic optimisation improvements
- `advanced` — architecture upgrade
- `mastery` — pushing limits within the same arch
- `controlled` — careful backbone fine-tuning
- `boundary` — boundary-aware loss + multi-scale TTA

### Phase Log Reports

Every phase produces exactly one human-readable report:

```
TRAINING AND PROGRESS/PHASE_X_Y/00_phase{N}_log.md
```

The `00_` prefix keeps the log at the top of any alphabetically-sorted file browser.

### Training Output Files

Every phase folder contains the same set of output files — no exceptions:

| File                     | Contents                                   |
| ------------------------ | ------------------------------------------ |
| `00_phase{N}_log.md`     | Full training report with analysis         |
| `all_metrics_curves.png` | Loss, IoU, Dice, Accuracy over all epochs  |
| `per_class_iou.png`      | Bar chart of final per-class IoU           |
| `lr_schedule.png`        | Learning rate curve(s)                     |
| `overfit_gap.png`        | Train-Val IoU gap across epochs (Phase 5+) |
| `evaluation_metrics.txt` | Per-epoch numbers in plain text            |
| `history.json`           | Machine-readable metrics for all epochs    |
| `best_model.pth`         | Best checkpoint by Val IoU                 |
| `final_model.pth`        | Last-epoch weights                         |

**Why the same set every time**: Anyone auditing the project can check any phase folder and know exactly what to expect. No hunting for results.

### Testing Interface Results

Every inference run through the Gradio interface saves a matched set of files:

```
TESTING_INTERFACE/
├── IMGS/
│   ├── 0007_20260310_104626_raw.png
│   ├── 0007_20260310_104626_overlay.png
│   └── 0007_20260310_104626_mask.png
└── RESULTS/
    └── 0007_20260310_104626.md
```

**Pattern**: `{NNNN}_{YYYYMMDD_HHMMSS}_{type}`

- `NNNN` — global sequence number (persists across restarts, never reused)
- `YYYYMMDD_HHMMSS` — timestamp (unique per run)
- `type` — `raw` / `overlay` / `mask`

The sequence number is initialised from existing files at startup — so restarting the app never produces collisions or gaps.

---

## 📋 Phase Discipline — The Rules

### Rule: Every Phase Has a Written Plan Before Training Starts

Before any training begins, a planning document is created in `BASIC_INTRO/`:

```
BASIC_INTRO/
├── 06_phase 6 work to be reviewd.md    ← Phase 6 plan (expert reviewed)
```

This plan defines:

- The bottleneck the phase is targeting.
- Exact hyperparameter changes and why.
- Expected IoU improvement range.
- What to do if it fails.

**Why**: Without a written plan, training becomes random experimentation. With a plan, every epoch either confirms or disproves a specific hypothesis.

### Rule: Expert Review Before Execution

Each phase script was reviewed by a domain expert before running. The review covered:

- Architecture correctness.
- Loss design and weight balance.
- Learning rate safety.
- Checkpoint loading integrity.
- Code cleanliness.

Only after review and all flagged issues resolved does training begin.

### Rule: Never Run the Same Phase Twice Without a New Name

If a phase needs to be re-run with different settings, it gets a new name or suffix. Phase 5 was never re-run as "Phase 5 v2" — instead, improvements accumulated into Phase 6.

---

## 📚 Documentation Consistency — The Rules

### README.md is the Single Source of Truth

`README.md` at the project root is always kept up to date with:

- Current best result prominently in the Results Summary table.
- Project tree reflecting actual folder state.
- One full section per phase (Configuration → Results → Curves → Analysis).
- Consistent section format across all phases.

Every time a new phase completes, three things happen:

1. The phase report (`00_phase{N}_log.md`) is written.
2. `README.md` gets a new phase section matching the format of all previous phases.
3. `TRAINING_SCRIPTS/SCRIPTS_EXPLAINED.md` gets the Phase N script explanation added.

### SCRIPTS_EXPLAINED.md is the Technical Deep-Dive

While `README.md` serves as the project overview, `SCRIPTS_EXPLAINED.md` is the place for:

- Exact code decisions and why they were made.
- A Parameter Evolution Table (one row per parameter, one column per phase).
- Lessons Learned entries added after each phase.

Both documents grow together — they are never allowed to fall out of sync.

---

## 🔒 Data Integrity — The Rules

### The Dataset is Immutable

The `DATASET/` folder is **never modified**. Training scripts read from it; they never write to it. All outputs go to `TRAINING AND PROGRESS/`.

### Nothing is Hard-Deleted Without a Backup

Before the MODELS/ folder cleanup between phases, a complete inventory was taken and only duplicate checkpoints from the same phase were removed — the single best checkpoint for each phase was preserved.

---

## 🔁 Phase Lifecycle Summary

Every phase follows this exact lifecycle:

```
1. PLAN     → Write phase plan in BASIC_INTRO/
2. REVIEW   → Expert reviews the plan and script
3. PREPARE  → Create output folder, write training script
4. BACKUP   → Copy previous best model as backup before starting
5. TRAIN    → Run the script, monitor live output
6. DOCUMENT → Write phase report, update README and SCRIPTS_EXPLAINED
7. ARCHIVE  → Best model copied to MODELS/ with IoU in filename
8. STANDBY  → Wait for next phase decision
```

No phase skips steps. No shortcuts.

---

## 📊 Version Control Strategy

Each phase checkpoint contains a `config` dictionary that records the **exact hyperparameters** used. This means the checkpoint itself is version-controlled metadata — even without Git, you can reconstruct what was done from the checkpoint file alone.

---

_This document was written to capture the management discipline applied throughout the Offroad Semantic Scene Segmentation project. The goal was never just to train a good model — it was to train a good model in a way that is transparent, reproducible, and auditable._

---

> _"A well-managed ML project is one where anyone can walk in at any phase, understand what was done, reproduce it, and continue from where it left off — without asking the author a single question."_
