# EDANE 开发文档

## 1. 项目目标

本项目提供一个可复现实验的 EDANE 研究原型，覆盖：

- 数据预处理与格式对齐
- 初始嵌入训练
- 动态增量更新
- 分类/链路预测评估
- 实验结果导出

适用于毕业设计、课程实验和方法验证，不是工业级分布式实现。

**Python 依赖**：NumPy ≥ 1.22、SciPy ≥ 1.7（`pip install numpy scipy`）。

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
│   ├── getting_started.md        # 快速上手
│   ├── pipeline_usage.md         # 流水线详细用法
│   └── thesis_evaluation_conclusion_auto.md  # 论文结论（自动生成）
├── src/                          # 源代码
│   ├── edane.py                  # 算法核心（模型定义、训练、增量更新、量化）
│   ├── edane_full_pipeline.py    # 端到端实验流水线
│   ├── prepare_datasets.py       # 原始数据集 → 统一 CSV 格式
│   ├── plot_metrics_svg.py       # 从 CSV 生成 SVG 曲线图
│   ├── generate_thesis_conclusion.py  # 论文结论自动生成
│   └── run_edane_experiment.py   # 轻量快速验证脚本
├── data/                         # 预处理后的数据集
│   ├── reddit_sample/
│   ├── amazon2m_sample/
│   ├── amazon3m_sample/
│   ├── mag_sample/
│   └── twitter_sample/
└── outputs/                      # 实验输出（按时间戳分目录）
```

## 3. 核心流程（代码视角）

### 3.1 算法层（`src/edane.py`）

1. `fit(adj, attrs)`
   - 邻接矩阵预处理（对称化、去自环，并转为 CSR 稀疏矩阵）
   - 属性标准化
   - 随机投影矩阵采样
   - 结构嵌入计算：`H_s = Σ α_k S^k R`
   - 属性嵌入计算：`H_x = X W_x`
   - 双曲空间门控融合
   - 可选量化

2. `apply_updates(...)`
   - 应用边增删和属性变更
   - 自动扩展受影响节点到一跳邻居
   - 按局部近似规则更新结构嵌入
   - 局部重算属性嵌入与融合向量

### 3.2 管线层（`src/edane_full_pipeline.py`）

1. 读取数据（文件模式）或构造合成图（synthetic 模式）
2. 切分快照并构造 `DynamicBatch`
3. 初始化训练并评估 snapshot 0
4. 按快照循环增量更新并评估
5. 导出指标、嵌入、映射、摘要
6. 自动导出 `metrics_curves.svg` 曲线图

## 4. 数据约定

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
| `metrics_per_snapshot.csv` | 每快照指标与时延 |
| `final_embedding.npy` | 最终浮点嵌入 |
| `final_embedding_int8.npy` | 量化嵌入（开启 `--quantize` 时） |
| `final_embedding_scale.npy` | 量化缩放系数 |
| `node_mapping.csv` | 节点 ID → 索引映射 |
| `metrics_curves.svg` | 指标曲线图 |

## 5. 关键参数建议

| 场景 | dim | snapshots | projection-density |
|------|-----|-----------|-------------------|
| 小规模调试 | 32 | 4–6 | 0.1 |
| 中等规模实验 | 64 | 6–10 | 0.08–0.15 |

- 增量更新更平滑：降低 `--learning-rate`（如 0.4–0.5）

## 6. 常见开发任务

### 6.1 新增数据集适配

1. 在 `src/prepare_datasets.py` 新增 `prepare_xxx_sample()`
2. 输出统一 CSV 四件套（至少 edges / features / labels）
3. 通过 `--dataset-preset` 接入管线

当前已内置预设：`reddit_sample`、`amazon2m_sample`、`amazon3m_sample`、`mag_sample`、`twitter_sample`

### 6.2 新增评估指标

1. 在 `src/edane_full_pipeline.py` 增加指标函数
2. 在 `evaluate_snapshot()` 中计算并返回
3. 同步更新 `metrics_per_snapshot.csv` 字段
4. 如需可视化，更新 `save_metrics_curves_svg()` 或 `src/plot_metrics_svg.py`

### 6.3 调整融合策略

在 `src/edane.py` 修改 `_fuse_embeddings()`：

- 可替换门控函数
- 可调整双曲映射缩放因子
- 可改为欧氏融合作对照实验

## 7. 调试与排错

- **报错：找不到数据预设目录**
  - 检查 `data/<preset>/` 是否存在
  - 先运行 `python src/prepare_datasets.py`

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
