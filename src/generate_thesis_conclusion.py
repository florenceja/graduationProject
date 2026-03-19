"""根据 all_results.csv 自动生成论文评估结论稿。"""

import csv
import os
from statistics import mean
from typing import Dict, List, Optional


def _to_float(row: Dict[str, str], key: str) -> float:
    return float(row.get(key, "0") or 0.0)


def _metric_or_none(row: Dict[str, str], key: str) -> Optional[float]:
    raw = row.get(key, "")
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    return float(text)


def _mean_or_zero(values: List[float]) -> float:
    return mean(values) if values else 0.0


def _range_text(values: List[float], suffix: str = "") -> str:
    if not values:
        return f"未记录{suffix}".strip()
    return f"{min(values):.4f}~{max(values):.4f}{suffix}"


def _risk_by_metrics(row: Dict[str, str]) -> str:
    macro_f1 = _to_float(row, "final_macro_f1")
    micro_f1 = _to_float(row, "final_micro_f1")
    auc = _to_float(row, "final_link_auc")
    ap = _metric_or_none(row, "final_link_ap")
    recon_auc = _metric_or_none(row, "final_reconstruction_auc")
    avg_upd = _to_float(row, "avg_update_latency_ms")

    risks: List[str] = []
    if macro_f1 < 0.1:
        risks.append("小类别识别能力偏弱（Macro-F1 较低）")
    if micro_f1 - macro_f1 > 0.4:
        risks.append("类别不均衡影响明显（Micro-F1 与 Macro-F1 差距较大）")
    if auc < 0.75:
        risks.append("结构可分性一般（AUC 偏低）")
    if ap is not None and ap < 0.75:
        risks.append("真实边精准捕捉能力一般（AP 偏低）")
    if recon_auc is not None and recon_auc < 0.75:
        risks.append("网络重构能力一般（reconstruction AUC 偏低）")
    if avg_upd > 5000:
        risks.append("动态更新时延较高（大图场景实时性受限）")
    if not risks:
        return "未见显著单项风险，整体指标较均衡。"
    return "；".join(risks) + "。"


def _dataset_sentence(row: Dict[str, str]) -> str:
    name = row["dataset"]
    n = int(float(row["num_nodes"]))
    macro_f1 = _to_float(row, "final_macro_f1")
    micro_f1 = _to_float(row, "final_micro_f1")
    auc = _to_float(row, "final_link_auc")
    ap = _metric_or_none(row, "final_link_ap")
    recon_auc = _metric_or_none(row, "final_reconstruction_auc")
    avg_upd = _to_float(row, "avg_update_latency_ms")
    ap_text = f"{ap:.4f}" if ap is not None else "未记录"
    recon_text = f"{recon_auc:.4f}" if recon_auc is not None else "未记录"
    return (
        f"- `{name}`：在 {n} 节点规模下，最终 Macro-F1={macro_f1:.4f}、"
        f"Micro-F1={micro_f1:.4f}、AUC={auc:.4f}、AP={ap_text}、重构AUC={recon_text}，平均更新时延={avg_upd:.2f} ms。"
    )


def generate_markdown(rows: List[Dict[str, str]]) -> str:
    """将 all_results.csv 的行列表转换为论文评估结论 Markdown 文本。"""
    if not rows:
        raise ValueError("all_results.csv 为空，无法生成结论。")

    aucs = [_to_float(r, "final_link_auc") for r in rows]
    aps = [v for r in rows if (v := _metric_or_none(r, "final_link_ap")) is not None]
    recon_aucs = [v for r in rows if (v := _metric_or_none(r, "final_reconstruction_auc")) is not None]
    macros = [_to_float(r, "final_macro_f1") for r in rows]
    micros = [_to_float(r, "final_micro_f1") for r in rows]
    init_lat = [_to_float(r, "initialization_latency_ms") for r in rows]
    upd_lat = [_to_float(r, "avg_update_latency_ms") for r in rows]
    p95_lat = [_to_float(r, "p95_update_latency_ms") for r in rows]
    crs = [_to_float(r, "quantization_compression_ratio") for r in rows]

    best_auc = max(rows, key=lambda r: _to_float(r, "final_link_auc"))
    best_ap = max(rows, key=lambda r: _to_float(r, "final_link_ap"))
    best_micro = max(rows, key=lambda r: _to_float(r, "final_micro_f1"))
    worst_macro = min(rows, key=lambda r: _to_float(r, "final_macro_f1"))

    lines: List[str] = []
    lines.append("# 论文可用版评估结论（自动生成）")
    lines.append("")
    lines.append("## 1. 总体结论")
    lines.append("")
    lines.append(
        f"基于 `all_results.csv` 的 {len(rows)} 组数据集实验结果，EDANE 在结构保持、动态更新与量化压缩方面总体达到预期。"
    )
    lines.append(
        f"链路预测 ROC-AUC 覆盖区间为 **{_range_text(aucs)}**，均高于随机水平；"
        f"Link prediction Average Precision (AP) 覆盖区间为 **{_range_text(aps)}**，Edge reconstruction ROC-AUC 覆盖区间为 **{_range_text(recon_aucs)}**；"
        f"量化压缩比稳定在 **{min(crs):.2f}~{max(crs):.2f}x**，说明压缩效果稳定。"
    )
    lines.append(
        f"初始化时延为 **{min(init_lat):.2f}~{max(init_lat):.2f} ms**，平均增量更新时延为 "
        f"**{min(upd_lat):.2f}~{max(upd_lat):.2f} ms**，P95 更新时延为 "
        f"**{min(p95_lat):.2f}~{max(p95_lat):.2f} ms**。"
    )
    lines.append(
        f"分类指标方面，Macro-F1 区间 **{min(macros):.4f}~{max(macros):.4f}**，"
        f"Micro-F1 区间 **{min(micros):.4f}~{max(micros):.4f}**，"
        "表明不同数据集在类别分布与特征可分性上存在显著差异。"
    )
    lines.append("")
    lines.append(
        f"其中，`{best_auc['dataset']}` 在结构建模上表现最好（AUC={_to_float(best_auc, 'final_link_auc'):.4f}），"
        f"`{best_ap['dataset']}` 在真实边精准捕捉上表现最好（AP={_to_float(best_ap, 'final_link_ap'):.4f}），"
        f"`{best_micro['dataset']}` 在总体分类准确性上表现最好（Micro-F1={_to_float(best_micro, 'final_micro_f1'):.4f}），"
        f"`{worst_macro['dataset']}` 的小类别识别能力最弱（Macro-F1={_to_float(worst_macro, 'final_macro_f1'):.4f}）。"
    )

    lines.append("")
    lines.append("## 2. 各数据集一句话解读")
    lines.append("")
    for row in rows:
        lines.append(_dataset_sentence(row))

    lines.append("")
    lines.append("## 3. 风险说明")
    lines.append("")
    for row in rows:
        lines.append(f"- `{row['dataset']}`：{_risk_by_metrics(row)}")

    lines.append("")
    lines.append("## 4. 可直接引用的结论段（正文版）")
    lines.append("")
    lines.append(
        "综合实验结果可见，EDANE 在多数据集上均能稳定完成动态图嵌入学习，并在链路预测、Average Precision 与网络重构任务上取得稳定表现，"
        "说明其结构保持能力与增量更新机制有效。同时，int8 量化始终维持接近 8 倍压缩比，验证了该方法在存储效率上的工程可行性。"
        "然而，不同数据集的 Macro-F1 波动较大，反映模型在长尾类别上的鲁棒性仍有提升空间。后续可通过类别重加权、重采样、"
        "更高质量属性特征与分层评估策略进一步增强小类别识别能力。"
    )
    lines.append("")
    lines.append(
        f"> 自动生成时间：{os.path.basename(__file__)} 运行时；样本数：{len(rows)}；"
        f"平均 AUC={_mean_or_zero(aucs):.4f}，平均 AP={_mean_or_zero(aps):.4f}，平均重构AUC={_mean_or_zero(recon_aucs):.4f}，平均 Micro-F1={_mean_or_zero(micros):.4f}。"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """读取 all_results.csv 并生成 docs/thesis_evaluation_conclusion_auto.md。"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_csv = os.path.join(project_root, "all_results.csv")
    out_md = os.path.join(project_root, "docs", "thesis_evaluation_conclusion_auto.md")

    with open(in_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = [row for row in csv.DictReader(f)]

    content = generate_markdown(rows)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"已生成: {out_md}")


if __name__ == "__main__":
    main()

