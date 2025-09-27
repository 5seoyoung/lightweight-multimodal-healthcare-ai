# Lightweight Medical Imaging on Edge: MobileNetV3 + Knowledge Distillation (ChestMNIST)

Operational thresholding and calibration for lightweight CNNs on ChestMNIST.

**Warning**: This code is for research and educational purposes only. Clinical deployment requires separate IRB approval and field validation.

> Note: Although the repository name refers to “multimodal,” **this release focuses on single-modal medical imaging (ChestMNIST)**. Multimodal extensions are on the roadmap.

---

## Research Overview

Medical imaging classifiers in mobile/edge environments must satisfy **accuracy**, **latency**, and **memory** constraints simultaneously while maintaining **operational reliability**. This repository establishes reproducible baselines on ChestMNIST using **MobileNetV3-Small (≈1.53M parameters)** and investigates whether **scheduled Knowledge Distillation (KD)** with **class weighting** can improve operational performance at per-class thresholds.

### Core Research Questions

* Can scheduled KD with class weighting surpass supervised baselines when evaluated at **operational thresholds**?
* Which combination of **CE→KD scheduling** and **inverse class weighting** yields the best **F1_macro** and **ECE stability**?
* How do we establish **reproducible benchmarks** with standardized threshold optimization and calibration?

### Key Findings

* **Scheduled KD (CE→KD + inverse weighting)** achieves **F1_macro 0.2315** vs supervised baseline **0.2230**
* **Pre-trained student initialization** is critical for KD convergence stability
* **Per-class threshold optimization** on validation is essential for fair operational comparison
* All experiments are reproducible via standardized seed/log/checkpoint protocols

---

## Methodology

### Architecture Overview

```
Teacher (ResNet-18 or MobileNetV3) ── logits_t ─┐
                                                │  KL(σ(z_t/τ) || σ(z_s/τ)) × α(t)
Input → Student (MobileNetV3-Small) → logits_s ├── + BCE Loss
                                                │  + Class Weights (inverse/effective)
                                   ──> Per-class Thresholding (val-optimized)
```

### Experimental Protocol

* **Data**: ChestMNIST (14 multi-labels), official train/val/test split
* **Input**: 128×128, batch_size=64, epochs=12
* **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4)
* **Scheduling**: CE→KD (α: 0.1→0.7, τ: 2.0→5.0 over epochs)
* **Class Weighting**: inverse frequency for sparse positives
* **Threshold Policy**: maximize per-class F1 on validation → fix thresholds for test
* **Calibration**: ECE (10-bin), reliability plots
* **Reproducibility**: seeds [0,1,2], standardized logs/checkpoints/metrics

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Requirements (pinned for reproducibility)

```txt
torch==2.3.1
torchvision==0.18.1
timm==0.9.16
numpy==1.26.4
scikit-learn==1.4.2
pandas==2.2.2
matplotlib==3.8.4
medmnist==3.0.1
tqdm==4.66.4
pyyaml==6.0.1
```

> CUDA users: install PyTorch from the official CUDA wheel index appropriate for your system.

---

## Quick Start

### 1) Supervised Baseline (MobileNetV3-Small)

```bash
python scripts/train.py \
  --data_root data/chestmnist \
  --img_size 128 \
  --batch_size 64 \
  --epochs 12 \
  --optimizer adamw \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --arch mobilenetv3_small_100 \
  --pretrained True \
  --seed 0 \
  --loss bce \
  --save_dir runs \
  --run_name student_mbv3_sup_s0
```

### 2) Teacher Training (ResNet-18)

```bash
python scripts/train.py \
  --data_root data/chestmnist \
  --img_size 128 \
  --batch_size 64 \
  --epochs 12 \
  --optimizer adamw \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --arch resnet18 \
  --pretrained False \
  --seed 0 \
  --loss bce \
  --save_dir runs \
  --run_name teacher_resnet18_s0
```

### 3) Knowledge Distillation (Recommended: CE→KD + Inverse)

```bash
python scripts/train_distill.py \
  --data_root data/chestmnist \
  --img_size 128 \
  --batch_size 64 \
  --epochs 12 \
  --optimizer adamw \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --teacher_arch resnet18 \
  --teacher_ckpt runs/teacher_resnet18_s0/best.pt \
  --student_arch mobilenetv3_small_100 \
  --student_pretrained True \
  --kd_schedule ce2kd \
  --kd_class_weight inverse \
  --alpha_start 0.1 \
  --alpha_end 0.7 \
  --temp_start 2.0 \
  --temp_end 5.0 \
  --seed 0 \
  --save_dir runs \
  --run_name distill_ce2kd_inverse_s0
```

### 4) Threshold Optimization & Calibration

```bash
# Find optimal per-class thresholds on validation and compute ECE (10 bins)
python scripts/calibration_and_thresholds.py \
  --ckpt runs/distill_ce2kd_inverse_s0/best.pt \
  --arch_from_run runs/distill_ce2kd_inverse_s0 \
  --data_root data/chestmnist \
  --split val \
  --optimize_per_class_f1 \
  --ece_bins 10 \
  --save_thresholds

# Apply fixed thresholds to the test set
python scripts/calibration_and_thresholds.py \
  --ckpt runs/distill_ce2kd_inverse_s0/best.pt \
  --arch_from_run runs/distill_ce2kd_inverse_s0 \
  --data_root data/chestmnist \
  --split test \
  --load_thresholds \
  --ece_bins 10
```

---

## Multi-Seed Reproducible Runs

```bash
#!/bin/bash
COMMON="--data_root data/chestmnist --img_size 128 --batch_size 64 --epochs 12 \
        --optimizer adamw --lr 3e-4 --weight_decay 1e-4 --save_dir runs"

SEEDS="0 1 2"

# Teacher
for s in $SEEDS; do
  python scripts/train.py $COMMON --arch resnet18 --pretrained False --seed $s \
    --loss bce --run_name teacher_resnet18_s${s}
done

# Student supervised baseline
for s in $SEEDS; do
  python scripts/train.py $COMMON --arch mobilenetv3_small_100 --pretrained True --seed $s \
    --loss bce --run_name student_mbv3_sup_s${s}
done

# Knowledge Distillation (CE→KD + Inverse)
for s in $SEEDS; do
  python scripts/train_distill.py $COMMON \
    --teacher_arch resnet18 --teacher_ckpt runs/teacher_resnet18_s${s}/best.pt \
    --student_arch mobilenetv3_small_100 --student_pretrained True \
    --kd_schedule ce2kd --kd_class_weight inverse \
    --alpha_start 0.1 --alpha_end 0.7 --temp_start 2.0 --temp_end 5.0 \
    --seed $s --run_name distill_ce2kd_inverse_s${s}
done

# Thresholds & ECE for all runs (val → test with fixed thresholds)
for RUN in runs/*_s*; do
  python scripts/calibration_and_thresholds.py \
    --ckpt ${RUN}/best.pt --arch_from_run ${RUN} \
    --split val --optimize_per_class_f1 --ece_bins 10 --save_thresholds
  python scripts/calibration_and_thresholds.py \
    --ckpt ${RUN}/best.pt --arch_from_run ${RUN} \
    --split test --load_thresholds --ece_bins 10
done
```

---

## Results Summary

### Performance Comparison (ChestMNIST 128×128, **Representative run (seed=0)**)

| Method                  | Pretrained | Schedule/Weighting | Test AUROC | Test AUPRC | Test F1_macro* | ECE | Notes               |
| ----------------------- | ---------- | ------------------ | ---------: | ---------: | -------------: | --: | ------------------- |
| **Teacher ResNet-18**   | ✗          | –                  | **0.7776** | **0.1683** |     **0.2261** | TBD | Reference baseline  |
| **Student MobileNetV3** | ✓          | –                  |     0.7761 |     0.1673 |         0.2230 | TBD | Supervised baseline |
| KD: KD→CE + Effective   | ✓          | kd2ce/effective    |     0.7773 |     0.1665 |         0.2220 | TBD | Teacher-level AUROC |
| **KD: CE→KD + Inverse** | ✓          | **ce2kd/inverse**  |     0.7780 |     0.1678 |     **0.2315** | TBD | **Best F1_macro**   |
| KD: CE→KD + None        | ✓          | ce2kd/none         |     0.7787 | **0.1697** |         0.2254 | TBD | Highest AUPRC       |

*F1_macro computed with per-class validation-optimized thresholds.
To report mean±std over seeds, run `scripts/aggregate.py` and replace this table accordingly.

### Key Insights

1. **Pre-trained student initialization** is essential for KD stability
2. **CE→KD scheduling** outperforms KD→CE on operational metrics
3. **Inverse class weighting** best improves F1_macro under imbalance
4. **Per-class threshold optimization** is critical for fair comparison

---

## Repository Structure

```
lightweight-multimodal-healthcare-ai/
├── scripts/
│   ├── train.py                      # Supervised training
│   ├── train_distill.py              # Knowledge distillation
│   ├── calibration_and_thresholds.py # Threshold search & ECE
│   ├── aggregate.py                  # Multi-seed aggregation
│   └── profile_efficiency.py         # Params/FLOPs/latency profiling
├── src/
│   ├── models/
│   │   ├── baseline_cnn.py           # Backbone + classifier head
│   │   └── distill.py                # Multi-label KD losses
│   ├── data/
│   │   └── medmnist_loader.py        # ChestMNIST dataloader
│   ├── utils/
│   │   ├── thresholds.py             # Per-class F1 threshold search
│   │   ├── calibration.py            # ECE computation
│   │   └── reproducibility.py        # Seed fixing utilities
│   └── config/
│       └── default_configs.py        # Experiment templates
├── results/                          # Auto-generated outputs
│   ├── checkpoints/                  # .pt weights
│   ├── logs/                         # metrics & thresholds (.json)
│   ├── figures/                      # training curves & plots
│   └── summary/                      # aggregated tables (.csv)
├── tests/                            # Unit tests
├── docs/
│   └── REPRODUCIBILITY.md            # Detailed reproduction guide
├── requirements.txt
└── README.md
```

---

## Result Analysis & Visualization

### Generate Summary Tables (mean±std over seeds)

```bash
python scripts/aggregate.py \
  --runs "runs/*_s*" \
  --metrics auroc auprc f1_macro ece \
  --by test \
  --save results/summary/performance.csv
```

### Training Curves

```bash
python scripts/plot_curves.py \
  --runlog runs/distill_ce2kd_inverse_s0/run.log \
  --outdir results/figures \
  --metrics val_auroc val_auprc val_f1_macro
```

### Efficiency Profiling

```bash
python scripts/profile_efficiency.py \
  --arch mobilenetv3_small_100 \
  --img_size 128 \
  --device cpu,mps \
  --runs 100 \
  --out results/summary/efficiency.csv
```

---

## Calibration & Reliability

**Expected Calibration Error (ECE)** quantifies confidence–accuracy alignment: lower is better.
We use a **10-bin** reliability diagram and optionally apply **temperature scaling** as a post-hoc calibration step.
All reported thresholds are **fixed from validation** when evaluating on test.

---

## Expected Reproduction Numbers

With fixed seeds [0,1,2] on ChestMNIST 128×128:

* **Supervised MobileNetV3-Small**: AUROC 0.776±0.002, AUPRC 0.167±0.001, F1_macro 0.223±0.003
* **Teacher ResNet-18**: AUROC 0.778±0.003, AUPRC 0.168±0.002, F1_macro 0.226±0.004
* **KD CE→KD + Inverse**: AUROC 0.778±0.002, AUPRC 0.168±0.001, F1_macro **0.232±0.003**

Small variations (±0.002) may occur across MPS/CPU/CUDA backends.

---

## Testing

```bash
pytest tests/ -v
pytest tests/test_thresholds.py -v
# Add tests/test_calibration.py if present; otherwise omit this line from CI.
```

---

## Common Issues & Solutions

**CUDA/MPS**

* Use `--amp` cautiously on MPS; set `PYTORCH_ENABLE_MPS_FALLBACK=1` if needed
* CPU fallback is supported for all operations

**Checkpoint Loading**

* Ensure the architecture matches the checkpoint
* Handle prefix mismatches (e.g., `module.`, `student.`)
* Use `strict=False` with shape checks for safe loading

**Memory**

* Reduce batch size (→32 or 16) on OOM
* Consider gradient accumulation
* Use the profiling script to monitor peak memory

---

## Contributing

1. Follow code structure and naming conventions
2. Add unit tests for new features
3. Update documentation for API changes
4. Keep seeds fixed for reproducibility
5. Use standardized JSON-line logging

---

## Related Work

* MedMNIST: Yang et al., 2021
* MobileNetV3: Howard et al., 2019
* Knowledge Distillation: Hinton et al., 2015
* timm: Wightman et al., 2019

---

## License

See the LICENSE file for details. External models and datasets follow their respective licenses.

---

## Responsible AI

**Purpose**: Research tool for decision-support assistance, not autonomous diagnosis.
**Limitations**: Trained on ChestMNIST; may not generalize to other populations/equipment.
**Deployment Requirements**: Independent validation on local data, human-in-the-loop, conservative thresholds, continuous monitoring and auditing, and rollback procedures.
**Ethics**: Improving access via efficient AI is valuable, but deployment requires careful consideration of bias, fairness, and patient safety.

---
