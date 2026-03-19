"""阶段2/3实验矩阵一键脚本。

功能：
1) 阶段2：按更新频率（如 10/100/1000 次每秒）批量运行；
2) 阶段3：按 full / w/o-Attr / w/o-Hyperbolic / w/o-Inc 批量运行；
3) 自动汇总每次运行的 summary.json 到 CSV，便于论文表格直接使用。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Sequence


SUMMARY_FIELDS: Sequence[str] = (
    "mode",
    "dataset_preset",
    "snapshot_mode",
    "seed",
    "classifier",
    "num_nodes",
    "feature_dim",
    "embedding_dim",
    "num_snapshots",
    "initialization_latency_ms",
    "avg_update_latency_ms",
    "avg_compute_update_latency_ms",
    "avg_pacing_wait_ms",
    "p95_update_latency_ms",
    "final_macro_f1",
    "final_micro_f1",
    "final_link_auc",
    "final_link_ap",
    "final_reconstruction_auc",
    "ablation_tag",
    "update_rate",
    "effective_update_rate",
    "quantization_compression_ratio",
    "quantization_error",
    "binary_compression_ratio",
    "binary_error",
    "output_dir",
)


def _parse_rates(text: str) -> List[int]:
    vals: List[int] = []
    for part in text.split(","):
        p = part.strip()
        if p == "":
            continue
        vals.append(int(p))
    if not vals:
        raise ValueError("--stage2-rates 至少需要一个整数，例如 10,100,1000")
    for v in vals:
        if v <= 0:
            raise ValueError("--stage2-rates 中的速率必须 > 0")
    return vals


def _read_summary(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    row: Dict[str, object] = {}
    for key in SUMMARY_FIELDS:
        row[key] = data.get(key, "")
    return row


def _run_one(command: List[str]) -> None:
    print("[RUN]", " ".join(command))
    subprocess.run(command, check=True)


def _write_rows(path: str, rows: List[Dict[str, object]], extra_fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(extra_fields) + list(SUMMARY_FIELDS)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _build_common_args(args: argparse.Namespace) -> List[str]:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pipeline_path = os.path.join(project_root, "src", "edane_full_pipeline.py")
    common = [
        sys.executable,
        pipeline_path,
        "--mode",
        args.mode,
        "--snapshots",
        str(args.snapshots),
        "--dim",
        str(args.dim),
        "--order",
        str(args.order),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
        "--classifier",
        args.classifier,
    ]
    if args.quantize:
        common.append("--quantize")
    if args.binary_quantize:
        common.append("--binary-quantize")

    if args.mode == "file":
        if not args.dataset_preset:
            raise ValueError("file 模式需要提供 --dataset-preset")
        common.extend(["--dataset-preset", args.dataset_preset])
        common.extend(["--max-nodes", str(args.max_nodes)])
    else:
        common.extend(["--synthetic-nodes", str(args.synthetic_nodes)])
        common.extend(["--synthetic-classes", str(args.synthetic_classes)])
        common.extend(["--synthetic-feat-dim", str(args.synthetic_feat_dim)])
        common.extend(["--synthetic-rounds", str(args.synthetic_rounds)])
    return common


def run_stage23_matrix(args: argparse.Namespace) -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.dataset_preset if args.mode == "file" else "synthetic"
    out_root = args.output_root or os.path.join(project_root, "outputs", f"stage23_matrix_{tag}_{timestamp}")
    os.makedirs(out_root, exist_ok=True)

    common = _build_common_args(args)
    stage2_rates = _parse_rates(args.stage2_rates)

    stage2_rows: List[Dict[str, object]] = []
    for rate in stage2_rates:
        run_dir = os.path.join(out_root, f"stage2_rate_{rate}_full")
        cmd = common + ["--update-rate", str(rate), "--output-dir", run_dir]
        _run_one(cmd)
        row = _read_summary(os.path.join(run_dir, "summary.json"))
        row["phase"] = "stage2"
        row["variant"] = "full"
        row["target_update_rate"] = rate
        stage2_rows.append(row)

    if args.include_no_inc_stage2:
        for rate in stage2_rates:
            run_dir = os.path.join(out_root, f"stage2_rate_{rate}_no_inc")
            cmd = common + ["--update-rate", str(rate), "--no-inc", "--output-dir", run_dir]
            _run_one(cmd)
            row = _read_summary(os.path.join(run_dir, "summary.json"))
            row["phase"] = "stage2"
            row["variant"] = "w/o-Inc"
            row["target_update_rate"] = rate
            stage2_rows.append(row)

    _write_rows(
        os.path.join(out_root, "stage2_rate_results.csv"),
        stage2_rows,
        extra_fields=("phase", "variant", "target_update_rate"),
    )

    stage3_variants = [
        ("full", []),
        ("w/o-Attr", ["--no-attr"]),
        ("w/o-Hyperbolic", ["--no-hyperbolic"]),
        ("w/o-Inc", ["--no-inc"]),
    ]
    stage3_rows: List[Dict[str, object]] = []
    for variant, flags in stage3_variants:
        run_dir = os.path.join(out_root, f"stage3_{variant.replace('/', '_')}")
        cmd = common + flags + ["--update-rate", str(args.stage3_update_rate), "--output-dir", run_dir]
        _run_one(cmd)
        row = _read_summary(os.path.join(run_dir, "summary.json"))
        row["phase"] = "stage3"
        row["variant"] = variant
        stage3_rows.append(row)

    _write_rows(
        os.path.join(out_root, "stage3_ablation_results.csv"),
        stage3_rows,
        extra_fields=("phase", "variant"),
    )

    merged_rows = stage2_rows + stage3_rows
    _write_rows(
        os.path.join(out_root, "stage23_combined_results.csv"),
        merged_rows,
        extra_fields=("phase", "variant", "target_update_rate"),
    )

    print("阶段2/3矩阵实验完成")
    print("输出目录:", out_root)
    print("- stage2_rate_results.csv")
    print("- stage3_ablation_results.csv")
    print("- stage23_combined_results.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段2/3实验矩阵一键脚本")
    parser.add_argument("--mode", choices=["file", "synthetic"], default="file")
    parser.add_argument("--dataset-preset", type=str, default="reddit_sample")
    parser.add_argument("--output-root", type=str, default="")

    parser.add_argument("--snapshots", type=int, default=3)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier", choices=["centroid", "logreg"], default="logreg")
    parser.add_argument("--max-nodes", type=int, default=10000)
    parser.add_argument("--quantize", dest="quantize", action="store_true")
    parser.add_argument("--no-quantize", dest="quantize", action="store_false")
    parser.add_argument("--binary-quantize", action="store_true")

    parser.add_argument("--stage2-rates", type=str, default="10,100,1000")
    parser.add_argument("--include-no-inc-stage2", action="store_true")
    parser.add_argument("--stage3-update-rate", type=int, default=0)

    parser.add_argument("--synthetic-nodes", type=int, default=600)
    parser.add_argument("--synthetic-classes", type=int, default=6)
    parser.add_argument("--synthetic-feat-dim", type=int, default=24)
    parser.add_argument("--synthetic-rounds", type=int, default=30)
    parser.set_defaults(quantize=True)
    return parser


if __name__ == "__main__":
    run_stage23_matrix(build_parser().parse_args())
