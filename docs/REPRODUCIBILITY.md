# Reproducibility Guide

이 문서는 본 저장소의 실험을 동일하게 재현하기 위한 환경 세팅, 실행 명령, 로그/체크포인트 구조, 표·그림 생성 방법을 단계별로 정리한다. 모든 커맨드는 프로젝트 루트에서 실행한다고 가정한다.

## 1. 환경 요구사항

* OS: macOS 13+/Ubuntu 20.04+ (다른 OS도 가능하나 수치 오차가 약간 발생할 수 있음)
* Python: 3.10–3.11 권장
* 패키지: `requirements.txt` 사용
* 가속: Apple Silicon의 경우 MPS 사용 가능, 그 외 CPU 또는 CUDA

설치

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

환경 고정(선택)

```bash
pip freeze > docs/env_freeze.txt
```

## 2. 데이터셋 준비

MedMNIST 계열은 최초 실행 시 자동 다운로드된다. 기본 캐시 경로는 `~/.medmnist` 이며, 프록시/방화벽 환경이면 아래처럼 환경변수로 경로를 지정할 수 있다.

```bash
export MEDMNIST_CACHE=./data/medmnist_cache
mkdir -p "$MEDMNIST_CACHE"
```

본 논문 실험의 기본 데이터셋: `chestmnist`
도메인 일반화 점검(선택): `pneumoniamnist`, `organamnist` 또는 `organmnist_a`

## 3. 재현 가능 실행: 단일 러닝

### 3.1 지도학습 베이스라인 (MobileNetV3-Small)

입력 128/160, 약한 증강, 10–12 epoch에서 아래 커맨드를 각각 실행한다. 로그는 표준출력(JSONL)과 테스트 결과 JSON 파일로 기록된다.

```bash
# 해상도 128
python -m src.train \
  --dataset chestmnist \
  --img_size 128 \
  --batch_size 64 \
  --epochs 12 \
  --lr 3e-4 \
  --backbone mobilenetv3_small_100 \
  --aug light \
  --outdir results | tee results/seed_runs/baseline128/seed_0.log

# 해상도 160
python -m src.train \
  --dataset chestmnist \
  --img_size 160 \
  --batch_size 64 \
  --epochs 12 \
  --lr 3e-4 \
  --backbone mobilenetv3_small_100 \
  --aug light \
  --outdir results | tee results/seed_runs/baseline160/seed_0.log
```

출력 파일

* `results/logs/chestmnist_mobilenetv3_small_100_test.json`
  예시 키: `{"loss":..., "auroc":..., "auprc":..., "f1_macro":...}`
* 표준출력(JSONL): 각 epoch별 `val_auroc`, `val_auprc`, `val_f1_macro`, `val_f1_macro_opt`, `thresholds`, `sec` 포함

기대 범위(재현 확인용)

* AUPRC 0.173–0.175
* AUROC 0.782–0.785
* F1\_macro 0.228–0.237

### 3.2 지식증류 초기 실험

교사–학생은 동일 입력 파이프라인으로 학습하며, 선택 지표는 `--selection_metric auprc` 를 권장한다.

ResNet-18 → MobileNetV3-Small (입력 128, α=0.1, τ=5.0)

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
  --outdir results \
  | tee results/seed_runs/kd_r18_to_mbv3_128_a01_t5_e12/run.log
```

Self-distillation: MobileNetV3-Small → MobileNetV3-Small (입력 160, α=0.4, τ=3.0)

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
  --outdir results \
  | tee results/seed_runs/kd_mbv3_to_mbv3_160_a04_t3_e12/run.log
```

출력 파일

* 체크포인트: `results/checkpoints/distill_{dataset}_{teacher}_to_{student}.pt`
* 검증 임계값: `results/logs/distill_{dataset}_{teacher}_to_{student}_val_thresholds.json` (멀티라벨)
* 테스트 결과: `results/logs/distill_{dataset}_{teacher}_to_{student}_test.json`

기대 범위(재현 확인용)

* ResNet18→MobileNetV3: AUPRC ≈ 0.157, F1\_macro ≈ 0.181
* MobileNetV3 self: AUPRC ≈ 0.170, F1\_macro ≈ 0.208

## 4. 멀티 시드 실행

본 저장소는 seed 인자를 스크립트에서 직접 받지 않는다. 대신 동일 커맨드를 여러 번 실행해 seed별 로그를 분리 저장하는 방식을 권장한다(운영체제 RNG/라이브러리 비결정성 옵션은 일정한 편차 내로 통제되어 있음).

예: 3회 반복 실행 후 로그 파일을 seed\_0/1/2로 분리 저장

```bash
# 예시: baseline 128을 3회 반복
for i in 0 1 2; do
  python -m src.train \
    --dataset chestmnist \
    --img_size 128 \
    --batch_size 64 \
    --epochs 12 \
    --lr 3e-4 \
    --backbone mobilenetv3_small_100 \
    --aug light \
    --outdir results \
    | tee results/seed_runs/baseline128/seed_${i}.log
done
```

실험 수가 많을 경우 `configs/*.yaml` 과 `src/cli.py` 를 이용해 일괄 실행할 수 있다.

```bash
# YAML 기반 실행 (예: baseline_chestmnist_128.yaml)
python -m src.cli --config configs/baseline_chestmnist_128.yaml --runs 3
```

## 5. 로그/체크포인트/스키마

디렉터리 규칙

```
results/
  checkpoints/
    {dataset}_{backbone}.pt
    distill_{dataset}_{teacher}_to_{student}.pt
  logs/
    {dataset}_{backbone}_test.json
    distill_{dataset}_{teacher}_to_{student}_test.json
    distill_{dataset}_{teacher}_to_{student}_val_thresholds.json
  seed_runs/
    baseline128/seed_{k}.log
    baseline160/seed_{k}.log
    kd_r18_to_mbv3_128_a01_t5_e12/run.log
  thresholds/
    kd_r18_to_mbv3_128_a01_t5_e12.json    # 필요 시 수동 백업
  summary/
    runs.csv                               # 표 스크립트가 생성
```

에폭별 로그(JSONL, 표준출력)의 한 줄 예

```json
{"epoch": 8, "train_loss": 0.239, "val_loss": 0.225, "val_auroc": 0.783, "val_auprc": 0.174, "val_f1_macro": 0.232, "val_f1_macro_opt": 0.245, "thresholds": [0.12, 0.63, ...], "sec": 41.3}
```

테스트 결과 JSON 스키마

```json
{
  "loss": 0.232,
  "auroc": 0.7842,
  "auprc": 0.1746,
  "f1_macro": 0.2351
}
```

## 6. 표·그림 생성

### 6.1 결과 표 요약

`results/logs/*_test.json` 과 `results/seed_runs/*/*.log` 를 모아 CSV를 생성한다.

```bash
python scripts/make_tables.py \
  --logs_glob "results/logs/*test.json" \
  --seeds_glob "results/seed_runs/**/seed_*.log" \
  --out_csv "results/summary/runs.csv"
```

출력 `results/summary/runs.csv`에는 실험명, AUPRC/AUROC/F1\_macro 평균±표준편차(및 95% CI), 임계값 사용 여부 등이 정리된다.

### 6.2 학습 곡선/PR 곡선/신뢰도 다이어그램

학습 곡선(에폭별 val AUPRC 등)

```bash
python scripts/plot_curves.py \
  --runlog "results/seed_runs/kd_r18_to_mbv3_128_a01_t5_e12/run.log" \
  --outdir "results/figures"
```

PR 곡선과 신뢰도 다이어그램은 본 저장소 기본 스크립트에는 포함되어 있지 않지만, `results/logs/*_test.json` 과 검증 분포를 이용해 쉽게 확장 가능하다(예: sklearn/plotly).

### 6.3 효율성(Params/FLOPs/Latency/PeakMem)

```bash
# 예: MobileNetV3-Small, 128 해상도, MPS/CPU 모두 측정
python scripts/profile_efficiency.py \
  --backbone mobilenetv3_small_100 \
  --img_size 128 \
  --device mps,cpu \
  --runs 100 \
  --out_csv results/summary/efficiency_mbv3_128.csv
```

## 7. 기대 값 검증 체크포인트

아래 범위를 만족하면 재현 성공으로 간주한다(동일 하드웨어가 아니면 ±0.002 내외 편차 가능).

* supervised MobileNetV3-Small 128–160
  AUPRC 0.173–0.175, AUROC 0.782–0.785, F1\_macro 0.228–0.237
* KD: ResNet-18 → MobileNetV3 (α=0.1, τ=5.0, 128)
  AUPRC ≈ 0.157, F1\_macro ≈ 0.181
* KD: MobileNetV3 self (α=0.4, τ=3.0, 160)
  AUPRC ≈ 0.170, F1\_macro ≈ 0.208

## 8. 통계/보고 규약

* 모든 비교는 시드별 짝지은 값으로 Wilcoxon signed-rank 검정
* 효과크기: Cliff’s delta, AUPRC 차이에 대한 부트스트랩 CI 병기
* 리더보드 규약: 검증 최고 체크포인트 한 번으로 테스트 단일 평가

## 9. 결정 임계값과 교정

* 멀티라벨에서 검증 세트 기준 per-class 임계값을 탐색(`thresholds` 필드로 로그에 기록)
* baseline 학습 스크립트는 해당 임계값을 내부적으로 테스트에 적용하며, KD 스크립트는 `*_val_thresholds.json` 파일로 저장
* 확률 교정(ECE/온도 스케일링)은 기본 스크립트에 포함되어 있지 않다. 필요한 경우 검증 NLL 최소화 방식으로 per-class temperature를 적합해 테스트에 적용하고, ECE(15-bin)와 신뢰도 다이어그램을 추가 보고할 수 있다.

## 10. 재현성 팁 및 흔한 이슈

* MPS/CPU/CUDA 백엔드 차이로 미세한 수치 차이가 발생할 수 있다. 논문 표에는 평균±표준편차 및 95% CI를 함께 보고한다.
* Apple Silicon에서 MPS가 활성화되면 메모리 사용량이 줄고 지연시간이 짧아지지만, 드라이버 버전에 따라 편차가 있다.
* `medmnist` 다운로드 실패 시 재시도하면 대부분 해결되며, 프록시 환경이면 `MEDMNIST_CACHE` 를 로컬로 설정한다.
* 첫 실행에서 `timm` 프리트레인 가중치 다운로드가 발생할 수 있다. 방화벽 환경이면 수동으로 모델 가중치를 캐시에 배치한다.

## 11. 재사용 및 확장

* 다른 학생 백본으로 교체: `--backbone` 혹은 `--student_backbone` 변경
* 입력 해상도/증강 강도 변경: `--img_size`, `--aug` 로 제어
* 클래스 불균형 보정: `--use_pos_weight` 플래그(멀티라벨 전용)
* 설정 일괄화: `configs/*.yaml` 와 `src/cli.py`로 다중 러닝 자동화
* 효율성 최적화: 양자화/스파스화는 별도 브랜치로 제공 예정

---

문의나 재현 실패 사례는 실행 커맨드와 생성된 로그(JSONL/JSON), `docs/env_freeze.txt` 를 함께 공유하면 원인 분석에 도움이 된다.
