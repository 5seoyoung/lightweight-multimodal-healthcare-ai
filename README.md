# Operational Performance Evaluation of Scheduled Knowledge Distillation for Lightweight Medical Image Classification: Threshold Optimization and Calibration Analysis

**We systematically evaluate the effect of Scheduled Knowledge Distillation (SKD) for a MobileNetV3-Small student on ChestMNIST multilabel classification from the perspective of operational, threshold-based F1-score and Expected Calibration Error (ECE).**

---

## Table of Contents

* [Motivation](#motivation)
* [Key Contributions](#key-contributions)
* [Problem Definition](#problem-definition)
* [Methodology](#methodology)
* [Results](#results)
* [Analysis & Insights](#analysis--insights)
* [Reproducibility](#reproducibility)
* [Future Work](#future-work)
* [Limitations](#limitations)

---

## Motivation

When deploying medical image classifiers to mobile/edge environments, simple accuracy or AUROC alone cannot guarantee real-world performance. In practice, the following are essential:

1. **Operational thresholds** — class-wise decision boundaries that balance precision/recall.
2. **Calibration of predictive confidence** — whether the predicted probabilities reflect actual correctness.
3. **Computational efficiency** — real-time inference under tight resource budgets.

Knowledge Distillation (KD) is a promising technique to boost lightweight models, but prior work has largely focused on top-1 accuracy or AUROC. **Systematic evaluation on operational F1 and calibration has been scarce.**

---

## Key Contributions

### 1) An operational, deployment-oriented evaluation protocol

* Search class-wise F1-optimal thresholds on the validation set.
* **Fix** those thresholds on the test set to mimic real deployment.
* Quantify calibration with **ECE** (Expected Calibration Error) and reliability diagrams.

### 2) Calibration impact analysis of scheduled KD

* **Main finding:** Distillation with a CE→KD schedule plus inverse frequency weighting maintains AUROC but induces **severe over-confidence** (ECE: 0.0066 → 0.3323).
* A pretrained student already achieves strong operational performance **without** KD.

### 3) Fully reproducible pipeline

* Systematic experiments over 3 random seeds.
* Automatic generation of checkpoints, logs, thresholds, and visualizations.
* Command-driven end-to-end reproduction.

---

## Problem Definition

### Research Question

> **Under the same compute budget, does Scheduled KD (CE→KD) with inverse class weighting improve the operational metrics (F1, ECE) of a student model over supervised training?**

### Hypotheses

Given a ResNet-18 teacher distilled via a schedule:

* **H1:** Improve per-class F1 in multilabel classification.
* **H2:** Improve calibration of predicted probabilities.
* **H3:** Maintain efficiency (MobileNetV3-Small student).

---

## Methodology

### Dataset

**ChestMNIST** (MedMNIST benchmark)

* Multilabel chest X-ray classification (14 disease classes).
* Train: 78,468 / Val: 11,219 / Test: 22,433.
* Class imbalance: prevalence from 0.2% to 26.5%.
* Official split for fair comparison.

* Our training scripts call the **MedMNIST Python API** with `download=True`.
* First run will auto-download ChestMNIST into `~/.medmnist/` and cache it.
* No manual file prep needed.

**Manual (explicit) usage example:**

```python
# pip install medmnist
from medmnist import ChestMNIST
ds_train = ChestMNIST(split='train', download=True, root='~/.medmnist')  # change root as you like
ds_val   = ChestMNIST(split='val',   download=True, root='~/.medmnist')
ds_test  = ChestMNIST(split='test',  download=True, root='~/.medmnist')
```

**Where files live:**

* Default cache: `~/.medmnist/`
* Override by passing `root=...` to the MedMNIST dataset class (as above).

**Official resources:**

* MedMNIST **Docs / PyPI (usage, quick examples, dataset descriptions)**:
  [https://pypi.org/project/medmnist/](https://pypi.org/project/medmnist/)
* MedMNIST **GitHub (official repo: install, scripts, issues)**:
  [https://github.com/MedMNIST/MedMNIST](https://github.com/MedMNIST/MedMNIST)
* MedMNIST **Website – ChestMNIST page (dataset card/overview)**:
  [https://medmnist.com/](https://medmnist.com/) (see “Datasets” → ChestMNIST)


### Model Architecture

| Component   | Spec                            |
| ----------- | ------------------------------- |
| **Teacher** | ResNet-18                       |
| **Student** | MobileNetV3-Small (pretrained)  |
| **Head**    | Sigmoid multilabel (14 classes) |

**How to verify parameter counts:**

```python
import timm
def count_params(model): 
    return sum(p.numel() for p in model.parameters())

teacher = timm.create_model('resnet18', pretrained=False, num_classes=14)
student = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=14)

print(f"Teacher: {count_params(teacher):,}")
print(f"Student: {count_params(student):,}")
```

### Training Setup

```
Optimizer: AdamW
Learning rate: 3e-4
Weight decay: 1e-4
Batch size: 64
Epochs: 12
Early stopping: patience=5 (per default implementation)
Input size: 128×128
Augmentation: light (horizontal flip, small shift/crop)
Seeds: {0, 1, 2}
```

### Knowledge Distillation Framework

#### Objective

$$L_{\text{total}} = (1-\alpha) \cdot L_{\text{CE}} + \alpha \cdot L_{\text{KD}}$$

* $L_{\text{CE}}$: Binary Cross-Entropy (hard labels).
* $L_{\text{KD}}$: KL divergence between softened predictions:
  $$L_{\text{KD}} = \text{KL}(\sigma(z_T/\tau) \parallel \sigma(z_S/\tau))$$

#### Schedule: CE→KD

| Hyperparameter       | Start | End | Schedule |
| -------------------- | ----: | --: | -------- |
| $\alpha$ (KD weight) |   0.1 | 0.7 | linear ↑ |
| $\tau$ (temperature) |   2.0 | 5.0 | linear ↑ |

**Rationale:**
Early epochs: fit hard labels to stabilize boundaries.
Later: inject teacher’s soft distribution to improve generalization.

#### Class Imbalance Handling

**Inverse Frequency Weighting**:
$$w_c = \frac{1}{\text{freq}(c) + \epsilon}$$

Applied to the **KD term** (`--cw_kd inverse`). The supervised BCE keeps default weighting (no extra class weights). This aims to strengthen KD signals for rare diseases.

### Evaluation Protocol (core contribution)

#### Step 1: Threshold optimization on validation

For each class $c$:
$$\theta_c^* = \underset{\theta \in [0,1]}{\arg\max}, F1_c(\theta)$$

* Grid search in [0.0, 1.0], step=0.01.
* Optimized independently per class.

#### Step 2: Fixed thresholds on test

* Apply $\theta_c^*$ **as is** to the test set.
* Mimics deployment where thresholds are not adjusted on-the-fly.

#### Step 3: Calibration measurement

**Expected Calibration Error (ECE):**
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N}, |\text{acc}(B_m) - \text{conf}(B_m)|$$

* $M=10$ bins.
* $B_m$: samples in the $m$-th confidence bin.

**Reliability diagrams:**

* X: predicted confidence (binned).
* Y: empirical accuracy.
* Perfect calibration lies on the diagonal.

---

## Results

### Main comparison (test set, 3-seed average)

| Method                             |             AUROC ↑ |             AUPRC ↑ |      **F1-macro** ↑ |           **ECE** ↓ |
| ---------------------------------- | ------------------: | ------------------: | ------------------: | ------------------: |
| **Teacher** (ResNet-18)            |     0.7665 ± 0.0006 |     0.1580 ± 0.0010 |     0.2120 ± 0.0010 |                   – |
| **Student–Supervised** (MobileNet) | **0.7796 ± 0.0031** | **0.1711 ± 0.0038** | **0.2285 ± 0.0071** | **0.0066 ± 0.0027** |
| **Student–Distilled** (CE→KD)      |     0.7669 ± 0.0139 |     0.1640 ± 0.0011 |     0.2235 ± 0.0044 |     0.3323 ± 0.0039 |

### Per-seed details

| Method                 | Seed |  AUROC |  AUPRC |   F1-macro |        ECE |
| ---------------------- | ---: | -----: | -----: | ---------: | ---------: |
| **Student–Supervised** |    0 | 0.7761 | 0.1673 |     0.2230 |     0.0097 |
|                        |    1 | 0.7815 | 0.1750 | **0.2365** | **0.0048** |
|                        |    2 | 0.7811 | 0.1711 |     0.2260 |     0.0053 |
| **Student–Distilled**  |    0 | 0.7804 | 0.1645 |     0.2253 |     0.3298 |
|                        |    1 | 0.7677 | 0.1627 |     0.2184 |     0.3368 |
|                        |    2 | 0.7526 | 0.1648 |     0.2267 |     0.3303 |

### Key observations

1. **Supervised student superiority**

   * F1: +2.2% relative improvement.
   * ECE: **≈50× lower** (0.33 vs 0.0066).
   * AUROC/AUPRC consistently higher.

2. **Calibration collapse with distillation**

   * ECE ≈ 0.33 indicates severe over-confidence.
   * Reliability curves lie **below** the diagonal.
   * Predicted probabilities are systematically higher than empirical accuracy.

3. **Efficiency**

   * The student is far smaller in parameters/compute than the teacher and is favorable for mobile deployment.
   * **Inference time not measured** in this study (will report in future work).

---

## Analysis & Insights

### Why did distillation worsen calibration?

#### Hypothesis 1: Transfer of teacher miscalibration

```
Teacher is miscalibrated
    ↓
α, τ schedules increase reliance on soft targets
    ↓
Student learns the teacher’s over-confidence pattern
```

**Evidence:** Threshold search for the teacher exhibited anomalies (many classes with $\theta_c \approx 0$).

#### Hypothesis 2: Rare-class signal dilution

* Inverse weighting applied to the KD term.
* High temperatures (2.0→5.0) may over-soften distributions.
* Rare positive signals can be down-weighted in practice.

#### Hypothesis 3: Strong pretrained student

* ImageNet pretraining provides strong representations.
* Additional KD may interfere rather than help in this setting.
* Domain specifics of medical imaging may exacerbate the issue.

### Reliability analysis

The **supervised student** stays close to the diagonal across bins, indicating good alignment between confidence and accuracy. The **distilled student** sits **below** the diagonal in many bins (over-confidence), especially at high-confidence ranges where empirical accuracy drops notably — consistent with `figures/fig2_reliability.pdf`.

**Per-class behavior:** The extremely rare class (Hernia, prevalence 0.2%) shows the most pronounced degradation under KD, suggesting that the combination of inverse weighting and high temperature may dilute rare positive signals. More detailed per-class F1 can be produced by extending `scripts/calibration_and_thresholds.py`.

---

## Reproducibility

### Environment

```bash
Python 3.8+
PyTorch 2.0+
timm 0.9.0+
medmnist 2.0+
numpy, scikit-learn, matplotlib
```

### Install dependencies

```bash
pip install torch torchvision timm medmnist scikit-learn matplotlib pandas
```

### Full reproduction procedure

Run the following from the project root. You may also save it as `scripts/reproduce_all.sh` and execute.

```bash
#!/bin/bash
set -euo pipefail
export PYTHONPATH="$(pwd)"
mkdir -p runs figures

# Step 1: Train teacher
for seed in 0 1 2; do
  python src/train.py \
    --dataset chestmnist \
    --backbone resnet18 \
    --img_size 128 \
    --batch_size 64 \
    --epochs 12 \
    --lr 3e-4 \
    --aug light \
    --patience 5 \
    --seed $seed \
    --exp_name teacher_resnet18_s${seed} \
    --outdir runs | tee runs/teacher_resnet18_s${seed}/run.log
done

# Step 2: Train student (supervised)
for seed in 0 1 2; do
  python src/train.py \
    --dataset chestmnist \
    --backbone mobilenetv3_small_100 \
    --pretrained \
    --img_size 128 \
    --batch_size 64 \
    --epochs 12 \
    --lr 3e-4 \
    --aug light \
    --seed $seed \
    --exp_name student_mbv3_sup_s${seed} \
    --outdir runs | tee runs/student_mbv3_sup_s${seed}/run.log
done

# Step 3: Distillation (CE→KD, inverse KD weighting)
for seed in 0 1 2; do
  python src/distill_train.py \
    --dataset chestmnist \
    --teacher_backbone resnet18 \
    --teacher_ckpt runs/teacher_resnet18_s${seed}/best.pt \
    --student_backbone mobilenetv3_small_100 \
    --pretrained_student \
    --sched ce2kd \
    --alpha_min 0.1 --alpha_max 0.7 \
    --tau_min 2.0 --tau_max 5.0 \
    --cw_kd inverse \
    --img_size 128 \
    --batch_size 64 \
    --epochs 12 \
    --lr 3e-4 \
    --aug light \
    --seed $seed \
    --exp_name distill_ce2kd_inverse_s${seed} \
    --outdir runs | tee runs/distill_ce2kd_inverse_s${seed}/run.log
done

# Step 4: Thresholding & calibration (student + distilled)
for run_dir in runs/student_mbv3_sup_s* runs/distill_ce2kd_inverse_s*; do
  [[ -d "$run_dir" ]] || continue
  
  # Validate: search per-class F1-optimal thresholds
  python scripts/calibration_and_thresholds.py \
    --dataset chestmnist \
    --backbone mobilenetv3_small_100 \
    --ckpt ${run_dir}/best.pt \
    --img_size 128 \
    --split val \
    --bins 10 \
    --save_thresholds ${run_dir}/thresholds_val_f1.json
  
  # Test: evaluate with fixed thresholds
  python scripts/calibration_and_thresholds.py \
    --dataset chestmnist \
    --backbone mobilenetv3_small_100 \
    --ckpt ${run_dir}/best.pt \
    --img_size 128 \
    --split test \
    --bins 10 \
    --load_thresholds ${run_dir}/thresholds_val_f1.json
done

# Step 5: Tables & figures
python scripts/tab1_class_stats.py
python scripts/tab2_main_results.py
python scripts/fig1_pipeline.py
python scripts/fig2_reliability.py
python scripts/fig3_training_dynamics.py

echo "[DONE] All artifacts saved under figures/ and runs/*/."
```

### Output layout

```
runs/
├── teacher_resnet18_s{0,1,2}/
│   ├── best.pt                    # checkpoint
│   └── run.log                    # training log
├── student_mbv3_sup_s{0,1,2}/
│   ├── best.pt
│   ├── run.log
│   ├── thresholds_val_f1.json     # per-class optimal thresholds
│   └── reliability_*.png          # calibration plots
└── distill_ce2kd_inverse_s{0,1,2}/
    ├── best.pt
    ├── run.log
    ├── thresholds_val_f1.json
    └── reliability_*.png

figures/
├── tab1_class_stats.csv           # dataset stats
├── tab2_main_results.csv          # aggregated results
├── fig1_pipeline.pdf              # method schematic
├── fig2_reliability.pdf           # 6-panel reliability
└── fig3_*.pdf                     # training dynamics
```

---

## Future Work

### Near-term improvements

#### 1) Teacher pre-calibration

```python
# Temperature Scaling on the teacher
teacher = apply_temperature_scaling(teacher, val_loader)

# Then distill with the calibrated teacher
distill(student, teacher, ...)
```

*Expected effect:* 30–50% reduction in ECE (based on prior work).

#### 2) Adaptive KD schedules

```python
# Per-class adaptive alpha
alpha_c = base_alpha * (1 + rarity_factor_c)

# Non-linear temperature annealing
tau(t) = tau_min + (tau_max - tau_min) * cosine_schedule(t)
```

#### 3) Feature-level distillation

* Attention Transfer (AT)
* Correlation-based Distillation (CRD)
* Intermediate hint losses

#### 4) Post-hoc calibration

```python
# Post-hoc calibration on validation
calibrator = TemperatureScaling()
calibrator.fit(student, val_loader)

# Re-optimize thresholds
thresholds_calibrated = optimize_f1(calibrated_student, val_loader)
```

### Longer-term directions

1. **Multi-teacher distillation** (ensembles with diverse architectures)
2. **Self-distillation** (use previous student snapshots as teacher)
3. **Task-specific KD** for particular disease groups
4. **Continual learning** with KD for new data/classes
5. **External validation** on CheXpert, NIH ChestX-ray, etc.

---

## Limitations

### Dataset

* **MedMNIST constraints:** 28×28 upsampled to 128×128; far from clinical resolutions.
* **Single source:** derived from ChestX-ray14; lacks multi-site generalization tests.
* **Label noise:** automatically derived labels have limited precision.

### Methodology

* **Teacher choice:** only ResNet-18 evaluated; stronger teachers (ResNet-50, ViT) not explored.
* **Schedule search:** only linear schedules; no non-linear/adaptive schedules yet.
* **Hyperparameters:** limited grid for $\alpha$, $\tau$.
* **Class weighting:** applied to **KD** term only; supervised BCE weighting not explored.

### Evaluation

* **Threshold generalization:** optimized on validation; may degrade under distribution shift.
* **Single calibration metric:** only ECE; MCE/Brier could complement.
* **Efficiency unmeasured:** inference time/memory/energy not reported.
* **Per-class reporting:** automated per-class F1 export not yet implemented.

### Practical deployment

**This study is for educational/benchmarking purposes.** Before clinical use:

* Multi-site prospective validation
* Regulatory approval (e.g., FDA/MFDS)
* IRB review
* Clinician-in-the-loop pilots
* Human-in-the-loop verification in production
* Continuous monitoring and rollback procedures

---
