"""从 metrics_per_snapshot.csv 生成 SVG 曲线图。"""

import argparse
import csv
import os
from typing import Dict, List, Tuple


def read_metrics(csv_path: str) -> List[Dict[str, float]]:
    """读取 metrics_per_snapshot.csv 并返回字典列表。"""
    rows: List[Dict[str, float]] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "snapshot": float(row["snapshot"]),
                    "update_latency_ms": float(row["update_latency_ms"]),
                    "macro_f1": float(row["macro_f1"]),
                    "micro_f1": float(row["micro_f1"]),
                    "link_auc": float(row["link_auc"]),
                }
            )
    return rows


def save_metrics_curves_svg(metrics_rows: List[Dict[str, float]], output_path: str, title: str) -> None:
    xs = [int(r["snapshot"]) for r in metrics_rows]
    if len(xs) == 0:
        raise ValueError("metrics_rows 为空，无法绘图。")

    top_series = {
        "macro_f1": [float(r["macro_f1"]) for r in metrics_rows],
        "micro_f1": [float(r["micro_f1"]) for r in metrics_rows],
        "link_auc": [float(r["link_auc"]) for r in metrics_rows],
    }
    latency = [float(r["update_latency_ms"]) for r in metrics_rows]

    width, height = 1100, 680
    margin_l, margin_r = 70, 30
    margin_t, margin_b = 45, 40
    plot_w = width - margin_l - margin_r

    top_y0, top_h = margin_t, 260
    bot_y0, bot_h = top_y0 + top_h + 70, 220
    x_min, x_max = min(xs), max(xs)
    if x_max == x_min:
        x_max = x_min + 1

    def sx(x: float) -> float:
        return margin_l + (x - x_min) / (x_max - x_min) * plot_w

    def sy_top(y: float) -> float:
        return top_y0 + top_h - y * top_h

    lat_min = 0.0
    lat_max = max(max(latency), 1.0) * 1.05

    def sy_bot(y: float) -> float:
        return bot_y0 + bot_h - (y - lat_min) / max(lat_max - lat_min, 1e-12) * bot_h

    def polyline(points: List[Tuple[float, float]], color: str) -> str:
        pts = " ".join([f"{x:.2f},{y:.2f}" for x, y in points])
        return f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{pts}" />'

    svg_lines: List[str] = []
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    svg_lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    svg_lines.append(f'<text x="70" y="25" font-size="18" font-family="Arial">{title}</text>')
    svg_lines.append(f'<line x1="{margin_l}" y1="{top_y0}" x2="{margin_l}" y2="{top_y0 + top_h}" stroke="#333"/>')
    svg_lines.append(f'<line x1="{margin_l}" y1="{top_y0 + top_h}" x2="{margin_l + plot_w}" y2="{top_y0 + top_h}" stroke="#333"/>')
    svg_lines.append(f'<line x1="{margin_l}" y1="{bot_y0}" x2="{margin_l}" y2="{bot_y0 + bot_h}" stroke="#333"/>')
    svg_lines.append(f'<line x1="{margin_l}" y1="{bot_y0 + bot_h}" x2="{margin_l + plot_w}" y2="{bot_y0 + bot_h}" stroke="#333"/>')

    for i in range(6):
        yv = i / 5
        y = sy_top(yv)
        svg_lines.append(f'<line x1="{margin_l}" y1="{y:.2f}" x2="{margin_l + plot_w}" y2="{y:.2f}" stroke="#eee"/>')
        svg_lines.append(f'<text x="8" y="{y + 4:.2f}" font-size="11" font-family="Arial">{yv:.1f}</text>')

    for i in range(6):
        yv = lat_min + (lat_max - lat_min) * i / 5
        y = sy_bot(yv)
        svg_lines.append(f'<line x1="{margin_l}" y1="{y:.2f}" x2="{margin_l + plot_w}" y2="{y:.2f}" stroke="#eee"/>')
        svg_lines.append(f'<text x="8" y="{y + 4:.2f}" font-size="11" font-family="Arial">{yv:.0f}</text>')

    x_ticks = min(10, len(xs))
    for i in range(x_ticks):
        xv = x_min + (x_max - x_min) * i / max(x_ticks - 1, 1)
        xx = sx(xv)
        svg_lines.append(f'<line x1="{xx:.2f}" y1="{top_y0 + top_h}" x2="{xx:.2f}" y2="{bot_y0 + bot_h}" stroke="#f3f3f3"/>')
        svg_lines.append(f'<text x="{xx - 8:.2f}" y="{bot_y0 + bot_h + 18}" font-size="11" font-family="Arial">{int(round(xv))}</text>')

    colors = {"macro_f1": "#1f77b4", "micro_f1": "#2ca02c", "link_auc": "#d62728"}
    for name, values in top_series.items():
        points = [(sx(x), sy_top(y)) for x, y in zip(xs, values)]
        svg_lines.append(polyline(points, colors[name]))
    lat_points = [(sx(x), sy_bot(y)) for x, y in zip(xs, latency)]
    svg_lines.append(polyline(lat_points, "#9467bd"))

    svg_lines.append(f'<text x="{margin_l}" y="{top_y0 - 12}" font-size="13" font-family="Arial">F1/AUC over snapshots</text>')
    svg_lines.append(f'<text x="{margin_l}" y="{bot_y0 - 12}" font-size="13" font-family="Arial">Update latency (ms)</text>')
    svg_lines.append("</svg>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))


def main() -> None:
    """命令行入口：从 CSV 生成 SVG 曲线图。"""
    parser = argparse.ArgumentParser(description="从 metrics_per_snapshot.csv 生成 SVG 曲线图")
    parser.add_argument("--metrics-csv", type=str, required=True)
    parser.add_argument("--output-svg", type=str, default="")
    parser.add_argument("--title", type=str, default="EDANE Metrics Curves")
    args = parser.parse_args()

    rows = read_metrics(args.metrics_csv)
    output_svg = args.output_svg or os.path.splitext(args.metrics_csv)[0] + "_curves.svg"
    save_metrics_curves_svg(rows, output_svg, args.title)
    print(f"曲线图已生成: {output_svg}")


if __name__ == "__main__":
    main()
