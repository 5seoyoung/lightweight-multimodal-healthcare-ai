# lightweight-multimodal-healthcare-ai
Efficient multimodal transformers for clinical decision support (AAAI-UC Research)

# Lightweight Medical Imaging on Edge: Mobile Models + Knowledge Distillation (ChestMNIST)

An open-source experimental repository that combines lightweight mobile architectures with knowledge distillation to simultaneously achieve **accuracy–efficiency–reproducibility** on **ChestMNIST**.
All scripts follow **reproducible log/checkpoint/threshold formats**, and tables/figures are automatically generated from logs.

> **Warning: Not for clinical use**: This code is for research and educational purposes. Clinical deployment requires separate IRB approval and field validation.

---

## Research Overview (What & Why)

Medical imaging classifiers in mobile/edge environments must satisfy not only **accuracy** requirements but also **latency and memory** constraints simultaneously. This repository establishes a **MobileNetV3-Small (≈1.53M)** baseline on ChestMNIST (multi-label 14-class) and provides experimental validation of whether **Knowledge Distillation (KD)** can actually improve **rare-class-focused AUPRC** within the same resource budget.

**Core Questions**
* Can KD actually surpass the baseline established by lightweight students through supervised learning alone?
* If not, what are the bottlenecks? And what combinations (scheduling, class weighting, feature distillation, calibration) are needed?

## Methodology Summary (How)

* **Student**: Fixed to MobileNetV3-Small (maintaining efficiency budget)
* **Teacher**: ResNet-18 (cross-architecture) or MobileNetV3-Small (self-distillation)
* **Loss**: Supervised learning (BCE/Focal/ASL) + Logit KD (KL, temperature τ) + Optional **feature distillation (Attention Transfer / FitNets-hint)**
* **Scheduling**: Dynamic adjustment of **α (distillation weight) and τ (temperature)** by learning phase (KD→CE vs CE→KD)
* **Decision Policy**: **Per-class threshold** optimization on validation set (or fixed 0.5) + **temperature scaling** with ECE reporting
* **Reproducibility**: Same seed sets, standardized log/checkpoint/threshold files, automated table/figure generation scripts

## Architecture Overview (Text Diagram)

```
Teacher (ResNet18 / MBV3) ── logits_t ─┐
                                        │   KL(σ(z_t/τ) || σ(z_s/τ)) × α(t), τ(t)
Input → Student (MBV3-Small) → logits_s ├─ + Supervised Loss (BCE/Focal/ASL)
                                        │   + Feature Distillation (AT/Hint, λ_feat)
                              ──> Sigmoid → Thresholding (per-class, val-tuned)
```

## Experimental Design

* **Data**: ChestMNIST from MedMNIST official split; PneumoniaMNIST/OrganMNIST-A for external distribution check (**test-only**)
* **Protocol**: Input 128/160, batch 64, 10–12 epochs, medical-friendly light augmentation
* **Metrics**: 1) **macro-AUPRC** (primary), 2) macro-AUROC, 3) **macro-F1** (fixed/val-optimized thresholds), 4) **ECE** (15 bins)
* **Efficiency**: Parameter count, FLOPs (128/160), batch-1 **latency (MPS/CPU)** 100-run average, **peak memory**
* **Statistics**: N=3 (up to 5 if needed) seed mean±std (95% CI), Wilcoxon signed-rank + Cliff's δ

## Key Findings

* **Supervised Baseline (Student Only)** MobileNetV3-Small (128–160, 10–12ep, light aug): **AUPRC 0.173–0.175**, **AUROC 0.782–0.785**, **F1_macro 0.228–0.237** → Low inter-seed variation with high reproducibility.

* **Initial KD Experiments (Same Protocol)**
   1. ResNet-18 → MobileNetV3 (α=0.1, τ=5.0, 128): **AUPRC 0.157**, **F1_macro ≈0.181**
   2. MobileNetV3 → MobileNetV3 (α=0.4, τ=3.0, 160): **AUPRC 0.170**, **F1_macro ≈0.208** 
   → **Both failed to surpass baseline** (particularly evident in AUPRC).

* **Why Did KD Lose? (Diagnosis)**
   1. **Soft target dilution in imbalanced settings**: **Sparse positive precision signals weakened** by being pulled toward majority negatives
   2. **Teacher-student architectural mismatch**: Difficult to bridge **representation distribution differences** between ResNet↔MobileNet with logits alone
   3. **Static α/τ inadequacy**: **Imbalanced learning pressure** due to different supervision signal combinations needed in early/late phases

## Our Proposed Solutions (Roadmap)

* **α(t)/τ(t) Scheduling**: Compare KD→CE (early representation alignment, late boundary refinement) ↔ CE→KD (teacher priors injection after label fitting)
* **Class-weighted KD**: Apply KD weights w_c to sparse classes (inverse frequency/Effective Number) → **protect rare-class AUPRC**
* **Feature Distillation (AT/Hint)**: Intermediate representation matching to mitigate **architectural mismatch**
* **Imbalance-aware Supervised Loss**: Focus on **hard examples and positive sparsity** with Focal/ASL/CB
* **Threshold/Calibration**: Fix per-class thresholds on validation + temperature scaling (ECE reporting)

**Target Metrics**: macro-AUPRC **≥ 0.185–0.195**, macro-F1 **≥ 0.245**, **parameters ≤ 2M**, latency/memory **within ±10%**

## Our Contributions

1. **Rigorous lightweight baseline establishment**: Established **reproducible** AUPRC/AUROC/F1 standards with MobileNetV3-Small
2. **Valuable negative result reporting**: Demonstrated with data that **naive KD fails to beat AUPRC** within same budget
3. **Practical diagnosis and prescription**: Why it lost (dilution·mismatch·static hyper) → **How to fix** (schedule/weight/feature/calibration)
4. **Complete reproducibility package**: **Automated figure/table regeneration** with standard logs·checkpoints·thresholds·CLI·parsers·plotting scripts

---

## Key Features

* **Lightweight baselines**: Establish ChestMNIST multi-label classification baselines with MobileNetV3-Small (≈1.53M)
* **Knowledge Distillation (KD)**: Initial evaluation of ResNet-18→MobileNetV3, MobileNetV3→MobileNetV3 self-distillation
* **Imbalance handling**: Multi-label standard **macro-AUPRC** / **macro-F1** (fixed & validation-optimized thresholds)
* **Decision thresholds/calibration**: Validation-based per-class threshold search and ECE reporting (optional)
* **Reproducibility package**: Standardized directory/log/checkpoint + automated table/curve generation scripts
* **Efficiency measurement**: Parameter count/FLOPs/latency/peak memory profiling

---

## Repository Structure

```
lightweight-multimodal-healthcare-ai/
├─ src/
│  ├─ datasets/medmnist_loader.py      # MedMNIST loader (with augmentation)
│  ├─ models/
│  │  ├─ baseline_cnn.py               # timm backbone wrapper + linear head
│  │  └─ distill.py                    # DistillLoss (multi-label/multi-class support)
│  ├─ utils/
│  │  ├─ thresholds.py                 # per-class F1-opt threshold search
│  │  ├─ class_freq.py                 # pos_weight estimation
│  │  └─ seed.py                       # seed fixing utilities
│  ├─ train.py                         # supervised learning baseline
│  ├─ distill_train.py                 # knowledge distillation training (with EMA)
│  └─ cli.py                           # YAML-based batch executor
├─ configs/                            # experiment templates (YAML)
├─ scripts/
│  ├─ make_tables.py                   # log→table CSV
│  ├─ plot_curves.py                   # run.log → training curves
│  └─ profile_efficiency.py            # efficiency measurement
├─ results/                            # (auto-generated) checkpoints/logs/tables/figures
├─ tests/                              # unit tests
├─ docs/REPRODUCIBILITY.md             # reproduction guide
├─ requirements.txt
└─ README.md
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional: Environment freezing

```bash
pip freeze > docs/env_freeze.txt
```

---

## Dataset

* Default: **ChestMNIST** (MedMNIST) – automatically downloaded on first run
* Optional: PneumoniaMNIST, OrganMNIST A – for external validation (test-only)
* Change cache path (optional):
  `export MEDMNIST_CACHE=./data/medmnist_cache`

---

## Quick Start

### 1) Supervised Learning Baseline (MobileNetV3-Small)

```bash
# Resolution 128
python -m src.train \
  --dataset chestmnist \
  --img_size 128 \
  --batch_size 64 \
  --epochs 12 \
  --lr 3e-4 \
  --backbone mobilenetv3_small_100 \
  --aug light \
  --outdir results

# Resolution 160
python -m src.train \
  --dataset chestmnist \
  --img_size 160 \
  --batch_size 64 \
  --epochs 12 \
  --lr 3e-4 \
  --backbone mobilenetv3_small_100 \
  --aug light \
  --outdir results
```

Output

* Checkpoint: `results/checkpoints/chestmnist_mobilenetv3_small_100.pt`
* Test results: `results/logs/chestmnist_mobilenetv3_small_100_test.json`
* Console JSONL: Per-epoch `val_auprc`, `val_f1_macro(_opt)`, `thresholds`, `sec` records

### 2) Knowledge Distillation (Initial Setup Reproduction)

ResNet-18 → MobileNetV3-Small (input 128, α=0.1, τ=5.0)

```bash
python -m src.distill_train \
  --dataset chestmnist \
  --img_size 128 \
  --batch_size 64 \
  --epochs 12 \
  --lr 3e-4 \
  --teacher_backbone resnet18 \
  --student_backbone mobilenetv3_small_100 \
  --alpha 0.1 \
  --tau 5.0 \
  --selection_metric auprc \
  --outdir results
```

MobileNetV3-Small → MobileNetV3-Small (input 160, α=0.4, τ=3.0)

```bash
python -m src.distill_train \
  --dataset chestmnist \
  --img_size 160 \
  --batch_size 64 \
  --epochs 12 \
  --lr 3e-4 \
  --teacher_backbone mobilenetv3_small_100 \
  --student_backbone mobilenetv3_small_100 \
  --alpha 0.4 \
  --tau 3.0 \
  --selection_metric auprc \
  --outdir results
```

Output

* Checkpoint: `results/checkpoints/distill_{dataset}_{teacher}_to_{student}.pt`
* Validation thresholds: `results/logs/distill_*_val_thresholds.json`
* Test results: `results/logs/distill_*_test.json`

---

## Multi-seed Runs (Batch)

### CLI + YAML

```bash
# Run 3 times
python -m src.cli --config configs/baseline_chestmnist_128.yaml --runs 3
```

Example `configs/baseline_chestmnist_128.yaml`

```yaml
name: "baseline_chestmnist_128"
script: "src.train"
args:
  dataset: "chestmnist"
  img_size: 128
  batch_size: 64
  epochs: 12
  lr: 3e-4
  backbone: "mobilenetv3_small_100"
  aug: "light"
  outdir: "results"
```

---

## Results/Log/Checkpoint Format

* Standard console (JSONL) line example

```json
{"epoch": 10, "train_loss": 0.241, "val_loss": 0.226,
 "val_auroc": 0.784, "val_auprc": 0.174, "val_f1_macro": 0.233,
 "val_f1_macro_opt": 0.246, "thresholds": [0.12, 0.63, ...], "sec": 41.3}
```

* Test results (JSON)

```json
{"loss": 0.232, "auroc": 0.7842, "auprc": 0.1746, "f1_macro": 0.2351}
```

Directory conventions

```
results/
  checkpoints/                     # *.pt
  logs/                            # *_test.json, *_val_thresholds.json
  seed_runs/                       # per-seed run.log (recommended to save with tee)
  thresholds/                      # backup if needed
  summary/runs.csv                 # generated by table scripts
  figures/                         # curves/plots
```

---

## Table & Figure Generation

Table summary

```bash
python scripts/make_tables.py \
  --logs_glob "results/logs/*test.json" \
  --seeds_glob "results/seed_runs/**/seed_*.log" \
  --out_csv "results/summary/runs.csv"
```

Training curves

```bash
python scripts/plot_curves.py \
  --runlog "results/seed_runs/kd_r18_to_mbv3_128_a01_t5_e12/run.log" \
  --outdir "results/figures"
```

Efficiency (Params/FLOPs/Latency/PeakMem)

```bash
python scripts/profile_efficiency.py \
  --backbone mobilenetv3_small_100 \
  --img_size 128 \
  --device mps,cpu \
  --runs 100 \
  --out_csv results/summary/efficiency_mbv3_128.csv
```

Detailed procedures in `docs/REPRODUCIBILITY.md`.

---

## Expected Reproduction Numbers (ChestMNIST)

* **Supervised MobileNetV3-Small** (128–160, 10–12 ep, light aug)
  AUPRC **0.173–0.175**, AUROC **0.782–0.785**, F1_macro **0.228–0.237**
* **KD: ResNet-18 → MobileNetV3** (α=0.1, τ=5.0, 128)
  AUPRC ≈ **0.157**, F1_macro ≈ **0.181**
* **Self-distill: MobileNetV3 → MobileNetV3** (α=0.4, τ=3.0, 160)
  AUPRC ≈ **0.170**, F1_macro ≈ **0.208**

> Reported as mean±std and 95% CI, with ±0.002 variation possible due to hardware/backend (MPS/CPU/CUDA) differences.

---

## Reproduction Checkpoints (Quick Review)

* Run baseline (128/160) → Results: `results/logs/chestmnist_*_test.json`
* Run 2 initial KD settings → Compare results: baseline vs KD
* Generate table CSV with `scripts/make_tables.py`, regenerate training curves with `scripts/plot_curves.py`
* Measure Params/FLOPs/Latency/PeakMem with `scripts/profile_efficiency.py`

## Limitations and Future Plans

* MedMNIST is a **low-resolution benchmark** that doesn't replace the complexity of clinical source data
* External distribution evaluation is **test-only small-scale**, not covering institutional/equipment/demographic variations
* Next steps: **Scheduled KD + class weighting + feature distillation** combined ablation, **quantitative significance testing**, additional latency reduction with **quantization/sparsification**, **multimodal expansion** and **explainability** analysis

---

## Testing

```bash
pytest -q
# or
python -m pytest -q
```

Included tests

* `tests/test_thresholds.py`: per-class F1-opt threshold search validation
* `tests/test_class_freq.py`: pos_weight (neg/pos) estimation validation

---

## Tips & Common Issues

* There may be numerical differences between MPS/CPU/CUDA, so report average/CI with multiple seeds.
* First run may require downloading `timm` pretrained weights.
* If MedMNIST is blocked in proxy/firewall environments, set `MEDMNIST_CACHE` to local path.

---

## Citation

If you use this repository in research or reports, please cite:

```
@misc{lightweight-medai-2025,
  title  = {Lightweight Mobile Architectures and Knowledge Distillation for ChestMNIST},
  author = {Anonymous},
  year   = {2025},
  note   = {GitHub repository},
  howpublished = {\url{https://example.com/anon-repo}}  % Use anonymous link during review
}
```

Related data/libraries

* MedMNIST: Yang et al., 2021
* MobileNetV3: Howard et al., 2019
* Knowledge Distillation: Hinton et al., 2015
* FitNets: Romero et al., 2015
* Attention Transfer: Zagoruyko & Komodakis, 2017
* timm: Wightman, 2019–

---

## License

Check the `LICENSE` file for this repository's license. Some external models/data follow their respective licenses.

---

## Responsible Use

* Purpose: Decision-making **assistance**
* Risks: False positives/negatives, distribution shift, calibration errors
* Requirements: Independent validation, site/equipment diversity, human-in-the-loop, auditable logging

---
