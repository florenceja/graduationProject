# OAG 数据规模与“全量可行方案”说明

本项目当前真实数据默认使用 **OAG-derived** 基准：

- 原始 zip：`dataset/OAG/v5_oag_publication_*.zip`
- 流水线输入 CSV：`data/OAG/{edges,features,labels}.csv`

> 重要：本项目的 file 模式会整读 CSV，因此“全量 OAG 可行”分为两件事：
> 1) 能否完成全量转换；
> 2) 转换后能否直接全量训练/评估。
> 当前代码更倾向于 **(1) 可在大机器上完成**，但 **(2) 需要采样或工程改造**。

## 1. 子图采样（推荐默认路线）

### 1.1 为什么需要采样

- 全量 OAG 节点/边数量远超普通开发机可承载范围
- 当前 `build_graph_from_files()` 会先将 CSV 整体读入内存后再裁剪

### 1.2 推荐子图构造命令

本机可跑子集：

```bash
python src/prepare_datasets.py --convert-oag --subset-profile test --overwrite
```

更接近正式实验（示例：节点>=15000，优先保边）：

```bash
python src/prepare_datasets.py --convert-oag --overwrite \
  --selection-strategy dense --max-papers 15000 --candidate-multiplier 3 \
  --min-venue-support 5 --keep-unlabeled --max-record-bytes 2000000
```

## 2. 全量转换（换硬件设备）

### 2.1 目标

在服务器/工作站上完成：

```bash
python src/prepare_datasets.py --convert-oag --subset-profile full --overwrite
```

### 2.2 建议硬件（经验下限）

> 这里给的是“更可能跑得完”的工程建议，不是理论最小值。

- 内存：**128GB+**（更理想：**256GB+**）
- 磁盘：NVMe SSD（至少数百 GB 可用空间，用于临时 SQLite、CSV 输出、缓存）
- CPU：16 核以上更理想（转换过程以 IO + JSON 解码为主）
- 运行环境：建议 Linux 服务器/工作站（长时任务更稳定）；Windows 也可但更容易遇到文件占用/路径/编码问题

### 2.3 现实可行方案（推荐写进论文复现说明）

方案 A（推荐）：**全量转换只做索引/统计 → 再采样子图用于训练**

1. 在大机器上完成 `--subset-profile full` 转换或更大规模的 index/统计
2. 再按实验规模抽取子图（例如 1e5~1e6 节点量级）
3. 将子图 CSV 拷贝回本项目 `data/OAG/`，在本机跑对比

方案 B（工程量大）：**改造为流式/采样式加载**

- 目标：让 `build_graph_from_files()` 在 `--max-nodes` 前就能预筛选
- 这样才能避免全量 CSV 先进入内存

方案 C（工程量大）：**改用更适合超大图的存储/框架**

- CSV → Parquet/Arrow
- 图采样器（按时间/引用扩展）→ 训练

## 3. 与当前实现口径相关的注意事项

- 引用边在流水线中会被规范化为无向图（方向丢失）
- 快照切分：有 time → 分位点切分；无 time → 按边数均分
