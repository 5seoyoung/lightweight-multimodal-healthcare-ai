# Lightweight Multimodal Healthcare AI

Efficient multimodal transformers for clinical decision support with operational thresholding and calibration.

**Warning**: This code is for research and educational purposes only. Clinical deployment requires separate IRB approval and field validation.

---

## Research Overview

Medical imaging classifiers in mobile/edge environments must satisfy **accuracy**, **latency**, and **memory** constraints simultaneously while maintaining **operational reliability**. This repository establishes reproducible baselines on ChestMNIST using **MobileNetV3-Small (≈1.53M parameters)** and investigates whether **Scheduled Knowledge Distillation (KD)** with **class weighting** can improve operational performance at per-class thresholds.

### Core Research Questions

* Can scheduled KD with class weighting surpass supervised learning baselines when evaluated at **operational thresholds**?
* What combination of **CE→KD scheduling** and **inverse class weighting** achieves the best **F1_macro** and **ECE stability**?
* How do we establish **reproducible benchmarks** with standardized threshold optimization and calibration procedures?

### Key Findings

* **Scheduled KD (CE→KD + inverse weighting)** achieves **F1_macro 0.2315** vs supervised baseline **0.2230**
* **Pre-trained student initialization** critical for KD convergence stability 
* **Per-class threshold optimization** on validation set essential for fair operational comparison
* All experiments reproducible with standardized seed/log/checkpoint protocols

---

## Methodology

### Architecture Overview

```
Teacher (ResNet-18) ──── logits_t ─┐
                                   │ KL(σ(z_t/τ) || σ(z_s/τ)) × α(t)
Input → Student (MobileNetV3) ─── logits_s ├─ + BCE Loss
                                   │ + Class Weights (inverse/effective)
                         ────> Per-class Thresholding (val-optimized)
```

### Experimental Protocol

* **Data**: ChestMNIST (14 multi-labels), official train/val/test split
* **Input**: 128×128, batch_size=64, epochs=12
* **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4)
* **Scheduling**: CE→KD (α: 0.1→0.7, τ: 2.0→5.0 over epochs)
* **Class Weighting**: inverse frequency for sparse positive classes
* **Threshold Policy**: Per-class F1 maximization on validation → fixed test application
* **Calibration**: ECE (10-bin), reliability plots
* **Reproducibility**: Seeds [0,1,2], automated log/checkpoint/metric standardization

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
medmnist>=2.2.0
```

---

## Quick Start

### 1. Supervised Baseline (MobileNetV3-Small)

```bash
# Single run
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

### 2. Teacher Training (ResNet-18)

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

### 3. Knowledge Distillation (Recommended: CE→KD + Inverse)

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

### 4. Threshold Optimization & Calibration

```bash
# Find optimal per-class thresholds on validation
python scripts/calibration_and_thresholds.py \
  --ckpt runs/distill_ce2kd_inverse_s0/best.pt \
  --arch mobilenetv3_small_100 \
  --data_root data/chestmnist \
  --split val \
  --optimize_per_class_f1 \
  --save_thresholds

# Apply thresholds to test set
python scripts/calibration_and_thresholds.py \
  --ckpt runs/distill_ce2kd_inverse_s0/best.pt \
  --arch mobilenetv3_small_100 \
  --data_root data/chestmnist \
  --split test \
  --load_thresholds
```

---

## Multi-Seed Reproducible Runs

### Batch Training Script

```bash
#!/bin/bash
COMMON="--data_root data/chestmnist --img_size 128 --batch_size 64 --epochs 12 \
        --optimizer adamw --lr 3e-4 --weight_decay 1e-4 --save_dir runs"

SEEDS="0 1 2"

# Teacher training
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

# Threshold optimization for all runs
for RUN in runs/*_s*; do
  python scripts/calibration_and_thresholds.py \
    --ckpt ${RUN}/best.pt --arch_from_run ${RUN} \
    --split val --optimize_per_class_f1 --save_thresholds
  python scripts/calibration_and_thresholds.py \
    --ckpt ${RUN}/best.pt --arch_from_run ${RUN} \
    --split test --load_thresholds
done
```

---

## Results Summary

### Performance Comparison (ChestMNIST 128×128, Seeds 0-2 Average)

| Method | Pretrained | Schedule/Weighting | Test AUROC | Test AUPRC | Test F1_macro* | ECE | Notes |
|--------|------------|-------------------|------------|------------|----------------|-----|--------|
| **Teacher ResNet-18** | ✗ | – | **0.7776** | **0.1683** | **0.2261** | 0.0050 | Reference baseline |
| **Student MobileNetV3** | ✓ | – | 0.7761 | 0.1673 | 0.2230 | TBD | Supervised baseline |
| KD: KD→CE + Effective | ✓ | kd2ce/effective | 0.7773 | 0.1665 | 0.2220 | TBD | Teacher-level AUROC |
| **KD: CE→KD + Inverse** | ✓ | **ce2kd/inverse** | 0.7780 | 0.1678 | **0.2315** | TBD | **Best F1_macro** |
| KD: CE→KD + None | ✓ | ce2kd/none | 0.7787 | **0.1697** | 0.2254 | TBD | Highest AUPRC |

*F1_macro computed with per-class validation-optimized thresholds

### Key Insights

1. **Pre-trained student initialization** essential for KD stability
2. **CE→KD scheduling** outperforms KD→CE for operational metrics  
3. **Inverse class weighting** provides best F1_macro for imbalanced multi-label
4. **Threshold optimization** critical for fair operational comparison

---

## Repository Structure

```
lightweight-multimodal-healthcare-ai/
├── scripts/
│   ├── train.py                    # Supervised learning
│   ├── train_distill.py           # Knowledge distillation
│   ├── calibration_and_thresholds.py  # Threshold optimization & ECE
│   ├── aggregate.py               # Multi-seed result aggregation
│   └── profile_efficiency.py     # Params/FLOPs/latency profiling
├── src/
│   ├── models/
│   │   ├── baseline_cnn.py        # Backbone + classifier head
│   │   ├── distill.py             # Multi-label KD loss
│   │   └── medvae.py              # VAE for latent experiments
│   ├── data/
│   │   └── medmnist_loader.py     # ChestMNIST dataloader
│   ├── utils/
│   │   ├── thresholds.py          # Per-class F1 optimization
│   │   ├── calibration.py         # ECE computation
│   │   └── reproducibility.py    # Seed fixing utilities
│   └── config/
│       └── default_configs.py    # Experiment templates
├── results/                       # Auto-generated outputs
│   ├── checkpoints/              # Model weights (.pt)
│   ├── logs/                     # Metrics & thresholds (.json)
│   ├── figures/                  # Training curves & plots
│   └── summary/                  # Aggregated tables (.csv)
├── tests/                        # Unit tests
├── docs/
│   └── REPRODUCIBILITY.md       # Detailed reproduction guide
├── requirements.txt
└── README.md
```

---

## Result Analysis & Visualization

### Generate Summary Tables

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

### ECE Computation
Expected Calibration Error measures confidence-accuracy alignment:
- **Lower ECE** = better calibration
- Computed using 10-bin reliability diagram
- Temperature scaling available for post-hoc calibration

### Threshold Optimization
Per-class F1 maximization on validation set:
- Searches optimal threshold per class independently  
- Applies fixed thresholds to test set for fair comparison
- Essential for operational deployment scenarios

---

## Expected Reproduction Numbers

With fixed seeds [0,1,2] on ChestMNIST 128×128:

* **Supervised MobileNetV3-Small**: AUROC 0.776±0.002, AUPRC 0.167±0.001, F1_macro 0.223±0.003
* **Teacher ResNet-18**: AUROC 0.778±0.003, AUPRC 0.168±0.002, F1_macro 0.226±0.004  
* **KD CE→KD + Inverse**: AUROC 0.778±0.002, AUPRC 0.168±0.001, F1_macro **0.232±0.003**

Small variations (±0.002) possible due to hardware differences (MPS/CPU/CUDA).

---

## Advanced Features

### Scheduled Knowledge Distillation
- **CE→KD**: Early label learning → late teacher knowledge transfer
- **KD→CE**: Early soft targets → late hard label focus
- Dynamic α(t) and τ(t) scheduling over training epochs

### Class-Weighted KD  
- **Inverse**: Weight = 1 / class_frequency (emphasizes rare classes)
- **Effective**: Weight = (1-β)/(1-β^n) where n = samples per class
- **None**: Uniform weighting across all classes

### Multi-Label Calibration
- Per-class sigmoid calibration with ECE reporting
- Reliability plots for confidence-accuracy visualization
- Temperature scaling for post-hoc calibration improvement

---

## Testing

```bash
# Run unit tests
pytest tests/ -v

# Test specific components
pytest tests/test_thresholds.py -v
pytest tests/test_calibration.py -v
```

---

## Common Issues & Solutions

### CUDA/MPS Compatibility
- Use `--amp` flag carefully with MPS (may cause instability)
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for MPS compatibility
- CPU fallback available for all operations

### Checkpoint Loading
- Ensure architecture matches between training and evaluation
- Handle prefix mismatches (e.g., `student.`, `module.`)
- Use `strict=False` loading with shape verification

### Memory Management
- Reduce batch size if OOM (try 32 or 16)
- Use gradient accumulation for effective larger batches
- Monitor peak memory usage with profiling scripts

---

## Contributing

1. Follow existing code structure and naming conventions
2. Add unit tests for new features
3. Update documentation for API changes
4. Maintain reproducibility with seed fixing
5. Use standardized logging format (JSON lines)

---

**Related Work:**
- MedMNIST: Yang et al., 2021
- MobileNetV3: Howard et al., 2019  
- Knowledge Distillation: Hinton et al., 2015
- timm: Wightman et al., 2019

---

## License

See LICENSE file for details. External models and datasets follow their respective licenses.

---

## Responsible AI

**Purpose**: Research tool for decision support assistance, not autonomous diagnosis.

**Limitations**: 
- Trained on limited dataset (ChestMNIST) 
- May not generalize to different populations/equipment
- Requires validation on institutional data before deployment

**Clinical Deployment Requirements**:
- Independent validation on local data
- Human-in-the-loop verification
- Conservative threshold setting
- Continuous monitoring and auditing
- Rollback procedures for model failures

**Ethics**: This research aims to improve healthcare accessibility through efficient AI, but clinical deployment requires careful consideration of bias, fairness, and patient safety.
