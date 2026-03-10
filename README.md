# grad-shake-align-multi-task

本仓库用于复现实验：Dual-Rank LoRA + Gradient Shake-to-Align，支持单任务和多任务的一键流程：

- `HPO -> FINAL -> PLOT`（通过 `scripts/pipeline_oneclick.py`）
- 单任务（single-task）
- 多任务（multi-task，按 `max_steps` 或 `epochs`）

## 环境

推荐使用你当前环境：

```bash
/home/lyclyq/miniconda3/envs/optimization/bin/python -V
```

依赖文件：

- 精简依赖：`requirements.txt`
- 锁定依赖（复现用）：`requirements.lock.txt`

## 一键运行示例

### 1) Multi-task MVP（当前常用）

```bash
/home/lyclyq/miniconda3/envs/optimization/bin/python scripts/pipeline_oneclick.py \
  --runs_group mvp_multi_glue3_20_30_50 \
  --dataset glue/rte \
  --model roberta-base \
  --multi_enabled true \
  --multi_datasets '["glue/cola","glue/mrpc","glue/rte"]' \
  --multi_steps_mode max_steps \
  --max_steps 50 \
  --trials 5 \
  --hpo_baseline_max_steps 20 \
  --hpo_grid_max_steps 20 \
  --hpo_rerank_max_steps 30 \
  --final_seeds '[2,3,5,7,11]'
```

### 2) Single-task 示例（RTE）

```bash
/home/lyclyq/miniconda3/envs/optimization/bin/python scripts/pipeline_oneclick.py \
  --runs_group single_glue_rte_roberta \
  --dataset glue/rte \
  --model roberta-base \
  --multi_enabled false \
  --trials 20 \
  --epochs 3 \
  --hpo_baseline_epochs 1 \
  --hpo_grid_epochs 1 \
  --rerank_epochs 2 \
  --final_epochs 7 \
  --final_seeds '[2,3,5,7,11]'
```

## 可改参数（本实验最常改）

- LoRA 容量
  - `--ours_r <int>`
  - `--ours_R <int>`
- 基座模型
  - `--model roberta-base` / `bert-base-uncased` / `distilbert-base-uncased` / ...
- HPO 总试验数
  - `--trials <int>`: 预算上限（会按预算截断 HPO 网格）
  - 不写 `--trials`: 按当前 `hpo.grid.knob_specs` 的离散化上限（`hpo.grid.max_m_*`）跑完整笛卡尔积（不会做预算截断，可能非常慢）
- 任务设置
  - 单任务：`--multi_enabled false --dataset glue/rte`
  - 多任务：`--multi_enabled true --multi_datasets '["glue/cola","glue/mrpc","glue/rte"]'`
- HPO 各阶段预算
  - `--hpo_baseline_max_steps <int>`
  - `--hpo_grid_max_steps <int>`
  - `--hpo_rerank_max_steps <int>`
  - 或 single-task 场景用 `--hpo_baseline_epochs / --hpo_grid_epochs / --rerank_epochs`
- FINAL 预算
  - multi-task 常用：`--multi_steps_mode max_steps --max_steps <int>`
  - single-task 常用：`--final_epochs <int>`
- FINAL seeds
  - `--final_seeds '[2,3,5,7,11]'`

## 附加覆盖参数

`pipeline_oneclick.py` 支持透传 `--set`（可重复），用于改实验细节：

```bash
--set train.batch_size=64
--set method.ours.voting.samples_per_vote=4
--set train.compile=true
--set train.tf32=true
--set train.fused_adamw=true
```

## 输出目录

运行后产物默认在：

- `runs/<runs_group>/hpo__...`
- `runs/<runs_group>/final__...`
- `runs/<runs_group>/final__.../trial_runs/_plots`

`pipeline_oneclick.py` 默认会跑完整链路：`HPO -> FINAL -> PLOT`。
