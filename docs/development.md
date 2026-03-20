# EDANE 开发文档

## 1. 项目目标

本项目提供一个可复现实验的 EDANE 研究原型，覆盖：

- 数据预处理与格式对齐
- 初始嵌入训练
- 动态增量更新
- 分类/链路预测/网络重构评估
- 实验结果导出

适用于毕业设计、课程实验和方法验证，不是工业级分布式实现。

**Python 依赖**：默认运行依赖为 NumPy ≥ 1.22、SciPy ≥ 1.7（`pip install numpy scipy`）；PyTorch 为可选增强后端，仅在 `backend="torch"` 时需要单独安装。

## 2. 目录结构

```
algorithms/
├── README.md                     # 项目总览
├── requirements.txt              # Python 依赖
├── run_all.bat                   # 一键实验脚本（Windows）
├── all_results.csv               # 多数据集汇总表
├── docs/                         # 文档
│   ├── algorithm.md              # 算法理论说明
│   ├── development.md            # 开发指南（本文件）
│   ├── evaluation_metrics_design_usage.md  # 评估指标说明
│   ├── getting_started.md        # 快速上手
│   ├── modules_1_4_integrated.md # 模块一至模块四整合文档
│   └── pipeline_usage.md         # 流水线详细用法
├── dataset/                      # 项目内原始数据目录（本地使用，prepare_datasets.py 优先读取）
├── src/                          # 源代码
│   ├── edane.py                  # 算法核心（模型定义、训练、增量更新、量化）
│   ├── edane_full_pipeline.py    # 端到端实验流水线
│   ├── prepare_datasets.py       # 原始数据集 → 统一 CSV 格式
│   ├── plot_metrics_svg.py       # 从 CSV 生成 SVG 曲线图
│   ├── run_edane_experiment.py   # 轻量快速验证脚本
│   └── run_stage23_experiments.py # 阶段2/3矩阵实验一键脚本
├── data/                         # 预处理后的数据集
│   ├── reddit_sample/
│   ├── amazon2m_sample/
│   ├── amazon3m_sample/
│   ├── mag_sample/
│   └── twitter_sample/
└── outputs/                      # 实验输出（按时间戳分目录，本地生成，默认不提交）
```

补充说明：

- `prepare_datasets.py` 当前按项目根目录下的 `dataset/` 作为默认原始数据目录。
- 当前仓库本地开发约定中，`dataset/` 与 `outputs/` 都按**本地数据/本地实验产物**对待，不建议直接纳入远程仓库版本管理。

## 3. 核心流程（代码视角）

### 3.1 算法层（`src/edane.py`）

1. `fit(adj, attrs)`
   - 邻接矩阵预处理（对称化、去自环，并转为 CSR 稀疏矩阵）
   - 属性标准化
   - 随机投影矩阵采样
   - 结构初始化：随机投影锚点 + 显式目标优化
   - 属性嵌入计算：`H_x = X W_x`
   - 可学习双曲门控融合
   - int8 / 可选 binary 压缩
   - `backend="torch"` 时，仅 dense 数值计算走 PyTorch；稀疏图操作仍走 SciPy

2. `apply_updates(...)`
   - 支持节点增删、边增删和属性变更
   - 自动扩展受影响节点到一跳邻居
   - 基于已有嵌入做局部近似结构更新（工程化实现，不是严格谱扰动闭式解）
   - 局部重算属性嵌入与融合向量

### 3.2 管线层（`src/edane_full_pipeline.py`）

1. 读取数据（文件模式）或构造合成图（synthetic 模式）
2. 切分快照并构造 `DynamicBatch`
3. 初始化训练并评估 snapshot 0
4. 按快照循环增量更新并评估
5. 导出指标、嵌入、映射、摘要
6. 自动导出 `metrics_curves.svg` 曲线图

## 4. 数据约定

### 4.0 原始数据与实验输入的关系

- `dataset/`：存放原始数据或原始压缩包，供 `prepare_datasets.py` 读取；
- `data/<preset>/`：存放转换后的统一 CSV 四件套，是 `edane_full_pipeline.py` 文件模式直接消费的标准输入。

也就是说，当前推荐流程是：

1. 把原始数据放在项目内 `dataset/`
2. 运行 `python src/prepare_datasets.py ...`
3. 让脚本生成 `data/reddit_sample/`、`data/amazon2m_sample/` 等标准化目录
4. 再用 `edane_full_pipeline.py --dataset-preset <preset>` 跑实验

### 4.1 输入文件（文件模式）

| 文件 | 列 | 是否必需 |
|------|----|---------|
| `edges.csv` | src, dst[, time] | 必需 |
| `features.csv` | node_id, f1, f2, ... | 可选 |
| `labels.csv` | node_id, label | 可选 |
| `attr_updates.csv` | time, node_id, f1, f2, ... | 可选 |

### 4.2 输出文件（每次运行）

在 `outputs/<数据集名>_<时间戳>/` 下生成（如 `synthetic_20260316_165834/`、`reddit_sample_20260316_170012/`）：

| 文件 | 说明 |
|------|------|
| `summary.json` | 整体实验摘要 |
| `metrics_per_snapshot.csv` | 每快照指标与时延（含 AP / reconstruction AUC） |
| `final_embedding.npy` | 最终浮点嵌入 |
| `final_embedding_int8.npy` | 量化嵌入（开启 `--quantize` 时） |
| `final_embedding_scale.npy` | 量化缩放系数 |
| `final_embedding_binary.npy` | Binary 副本（若开启 `--binary-quantize`） |
| `node_mapping.csv` | 节点 ID → 索引映射 |
| `metrics_curves.svg` | 指标曲线图 |

`summary.json` 现包含时延分解字段：

- `avg_update_latency_ms`（端到端，含节流等待）
- `avg_compute_update_latency_ms`（纯计算）
- `avg_pacing_wait_ms`（节流等待）

## 5. 关键参数建议

| 场景 | dim | snapshots | projection-density |
|------|-----|-----------|-------------------|
| 小规模调试 | 32 | 4–6 | 0.1 |
| 中等规模实验 | 64 | 6–10 | 0.08–0.15 |

- 增量更新更平滑：降低 `--learning-rate`（如 0.4–0.5）

## 6. 常见开发任务

### 6.1 新增数据集适配

1. 在 `src/prepare_datasets.py` 新增 `prepare_xxx_sample()`
2. 从项目内 `dataset/` 读取原始数据，并输出统一 CSV 四件套（至少 edges / features / labels）
3. 通过 `--dataset-preset` 接入管线

当前已内置预设：`reddit_sample`、`amazon2m_sample`、`amazon3m_sample`、`mag_sample`、`twitter_sample`

### 6.2 新增评估指标

1. 在 `src/edane_full_pipeline.py` 增加指标函数
2. 在 `evaluate_snapshot()` 中计算并返回
3. 同步更新 `metrics_per_snapshot.csv` 字段
4. 如需可视化，更新 `save_metrics_curves_svg()` 或 `src/plot_metrics_svg.py`

当前已内置：`macro_f1`、`micro_f1`、`link_auc`、`link_ap`、`reconstruction_auc`

### 6.4 调整模块级超参数

当前 `EDANE.__init__` 还支持：

- 初始化模块：`structure_weights`、`init_iterations`、`init_step_size`、`init_reg`、`init_tol`
- 融合模块：`fusion_train_steps`、`fusion_lr`、`fusion_weight_decay`
- 存储模块：`binary_quantize`
- 后端模块：`backend`（`numpy` / `torch`，其中 `torch` 仅切换 dense 计算路径）

主流水线默认只暴露常用参数；更细控制可直接通过 Python API 实验，或继续扩展 CLI。

### 6.5 阶段2/3实验矩阵复现

使用 `src/run_stage23_experiments.py` 可一键得到：

- 阶段2速率矩阵（默认 `10/100/1000`）
- 阶段3消融矩阵（`full / w/o-Attr / w/o-Hyperbolic / w/o-Inc`）

示例：

```bash
python src/run_stage23_experiments.py --mode file --dataset-preset reddit_sample --snapshots 3 --max-nodes 3000
```

输出：`stage2_rate_results.csv`、`stage3_ablation_results.csv`、`stage23_combined_results.csv`。

### 6.3 调整融合策略

在 `src/edane.py` 修改 `_fuse_embeddings()`：

- 可替换门控函数
- 可调整双曲映射缩放因子
- 可改为欧氏融合作对照实验

## 7. 调试与排错

- **报错：找不到数据预设目录**
  - 检查 `data/<preset>/` 是否存在
  - 先确认项目根目录下 `dataset/` 是否存在原始数据
  - 再运行 `python src/prepare_datasets.py`

- **报错：原始数据路径不对**
  - 当前 `prepare_datasets.py` 默认读取 `<project_root>/dataset`
  - 若你调整了目录结构，优先检查项目内 `dataset/README.md` 和 `prepare_datasets.py` 中对应的数据读取路径

- **报错：属性更新维度不一致**
  - `attr_updates.csv` 的特征列数必须与 `features.csv` 一致

- **内存压力较大**
  - file 模式优先使用 `--max-nodes` 控制节点规模（默认 10000）
  - 降低样本节点数
  - 降低 `--dim`
  - 减少 `--snapshots`
  - 重新生成更小样本（如 `--mag-max-nodes`、`--twitter-max-nodes`）

## 8. 后续工程化建议

- 继续优化稀疏算子实现（如更高效的增量边更新）
- 将批处理改为 mini-batch / 分块 out-of-core
- 增加多进程或分布式执行
- 增加实验配置文件（YAML）与统一日志系统

## 9. 当前实现边界（开发时应保持一致）

为避免文档与代码再次偏离，建议开发时统一使用以下表述口径：

- 模块一：当前实现是**稀疏随机投影锚点 + 显式目标优化初始化**；
- 模块二：当前实现是**局部一跳增量更新的工程化近似版本**，不是严格矩阵谱扰动闭式解；
- 模块三：当前实现是**轻量双曲门控融合**，不是重型多头注意力网络；
- 模块四：当前实现以**逐维对称 int8 量化**为主，binary 副本偏实验性质；
- 整个项目当前是**研究原型 / 可运行实验系统**，不是工业级分布式生产实现。
