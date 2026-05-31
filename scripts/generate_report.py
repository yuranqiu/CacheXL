#!/usr/bin/env python3
"""
生成统一报告: 同时生成 CoT, BoT, CacheXL 的报告，并输出对比表格
用法: python scripts/report_all.py [--methods cot,bot,cachexl]
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from methods.react.eval import (
    ks_statistic,
    expected_calibration_error,
    approx_tokens,
)

# Cost Constants
COST_EMBEDDING_INPUT = 0.10
COST_LLM_INPUT = 2.50
COST_LLM_OUTPUT = 10.00

METHODS = ["react", "bot", "cachexl"]


def load_jsonl(path: Path):
    items = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    items.append(json.loads(line))
                except:
                    pass
    return items


def calculate_metrics(items, method):
    if not items:
        return {}

    total = len(items)
    correct = 0
    total_latency = 0.0

    total_prompt = 0.0
    total_completion = 0.0
    total_embedding = 0.0

    strong_count = 0
    all_confs = []
    all_correct_flags = []

    for item in items:
        ground_truth = item.get("label") or item.get("gold") or item.get("answer")
        prediction = item.get("final_answer") or item.get("answer")

        is_correct = (
            str(ground_truth).strip().upper() == str(prediction).strip().upper()
        )
        if is_correct:
            correct += 1
            all_correct_flags.append(True)
        else:
            all_correct_flags.append(False)

        total_latency += float(item.get("latency", 0.0))

        # Tokens
        if method == "cachexl":
            total_embedding += float(item.get("embedding_tokens", 0.0))

            at = item.get("actor_tokens", {})
            total_prompt += float(at.get("prompt", 0))
            total_completion += float(at.get("completion", 0))

            rt = item.get("reflect_tokens", {})
            total_prompt += float(rt.get("prompt", 0))
            total_completion += float(rt.get("completion", 0))

            if item.get("used_escalator"):
                strong_count += 1
                st = item.get("escalator_tokens", {})
                total_prompt += float(st.get("prompt", 0))
                total_completion += float(st.get("completion", 0))
        else:
            at = item.get("actor_tokens", {})
            total_prompt += float(at.get("prompt", 0))
            total_completion += float(at.get("completion", 0))

        # Confidence
        conf = item.get("confidence") or item.get("reflect_confidence")
        if conf is not None:
            try:
                all_confs.append(float(conf))
            except:
                pass

    avg_latency = total_latency / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0

    cost_embedding = (total_embedding / 1_000_000) * COST_EMBEDDING_INPUT
    cost_input = (total_prompt / 1_000_000) * COST_LLM_INPUT
    cost_output = (total_completion / 1_000_000) * COST_LLM_OUTPUT
    total_cost = cost_embedding + cost_input + cost_output
    avg_cost = total_cost / total if total > 0 else 0

    # KS & ECE
    conf_correct = [all_confs[i] for i in range(len(all_confs)) if all_correct_flags[i]]
    conf_incorrect = [
        all_confs[i] for i in range(len(all_confs)) if not all_correct_flags[i]
    ]
    ks = ks_statistic(conf_correct, conf_incorrect)
    ece = expected_calibration_error(all_confs, all_correct_flags)

    return {
        "accuracy": accuracy,
        "avg_latency": avg_latency,
        "avg_cost": avg_cost,
        "avg_tokens": (total_prompt + total_completion + total_embedding) / total
        if total > 0
        else 0,
        "escalator_rate": strong_count / total if total > 0 else 0,
        "KS_Stat": ks,
        "ECE": ece,
        "total": total,
    }


def generate_method_report(method, results_dir):
    """生成单个方法的报告"""
    print(f"\n=== {method.upper()} Report ===")

    dataset_dirs = sorted([p for p in results_dir.iterdir() if p.is_dir()])
    rows = []

    for dataset_dir in dataset_dirs:
        files = sorted(
            list(dataset_dir.glob("batch*.jsonl"))
            + list(dataset_dir.glob("Batch*.jsonl"))
        )
        if not files:
            continue

        items = []
        for f in files:
            items.extend(load_jsonl(f))

        if not items:
            continue

        metrics = calculate_metrics(items, method)
        rows.append(
            {
                "Dataset": dataset_dir.name,
                "Accuracy": metrics.get("accuracy", 0),
                "AvgLatency": metrics.get("avg_latency", 0),
                "AvgCost": metrics.get("avg_cost", 0),
                "AvgTokens": metrics.get("avg_tokens", 0),
                "KS_Stat": metrics.get("KS_Stat", 0),
                "ECE": metrics.get("ECE", 0),
                "Samples": metrics.get("total", 0),
            }
        )

        print(
            f"  {dataset_dir.name}: Acc={metrics.get('accuracy', 0):.4f}, Latency={metrics.get('avg_latency', 0):.2f}s, Cost=${metrics.get('avg_cost', 0):.6f}"
        )

    return rows


def generate_comparison_report(experiments_dir, methods):
    """生成对比报告"""
    print("\n" + "=" * 80)
    print("COMPARISON REPORT")
    print("=" * 80)

    datasets = set()
    for m in methods:
        results_dir = experiments_dir / m / "results"
        if results_dir.exists():
            for d in results_dir.iterdir():
                if d.is_dir():
                    datasets.add(d.name)

    all_datasets = sorted(list(datasets))

    headers = [
        "Dataset",
        "ReAct_Acc",
        "BoT_Acc",
        "CacheXL_Acc",
        "ReAct_Cost",
        "BoT_Cost",
        "CacheXL_Cost",
        "ReAct_Latency",
        "BoT_Latency",
        "CacheXL_Latency",
        "ReAct_ECE",
        "BoT_ECE",
        "CacheXL_ECE",
        "ReAct_KS",
        "BoT_KS",
        "CacheXL_KS",
        "CacheXL_EscalatorRate",
        "ReAct_RelativeDiff",
        "BoT_RelativeDiff",
        "CacheXL_RelativeDiff",
        "ReAct_Ratio",
        "BoT_Ratio",
        "CacheXL_Ratio",
    ]

    csv_rows = [headers]

    print(
        f"\n{'Dataset':<12} | {'ReAct Acc':<12} | {'BoT Acc':<8} | {'Enh Acc':<8} | {'ReAct Cost':<13} | {'BoT Cost':<10} | {'Enh Cost':<10}"
    )
    print("-" * 100)

    for dataset in all_datasets:
        row = [dataset]
        method_metrics = {}

        for m in methods:
            results_dir = experiments_dir / m / "results" / dataset
            items = []
            if results_dir.exists():
                for f in results_dir.glob("*.jsonl"):
                    items.extend(load_jsonl(f))
            method_metrics[m] = calculate_metrics(items, m)

        # Accuracy
        row.append(f"{method_metrics.get('react', {}).get('accuracy', 0):.4f}")
        row.append(f"{method_metrics.get('bot', {}).get('accuracy', 0):.4f}")
        row.append(f"{method_metrics.get('cachexl', {}).get('accuracy', 0):.4f}")

        # Cost
        row.append(f"{method_metrics.get('react', {}).get('avg_cost', 0):.6f}")
        row.append(f"{method_metrics.get('bot', {}).get('avg_cost', 0):.6f}")
        row.append(f"{method_metrics.get('cachexl', {}).get('avg_cost', 0):.6f}")

        # Latency
        row.append(f"{method_metrics.get('react', {}).get('avg_latency', 0):.2f}")
        row.append(f"{method_metrics.get('bot', {}).get('avg_latency', 0):.2f}")
        row.append(f"{method_metrics.get('cachexl', {}).get('avg_latency', 0):.2f}")

        # ECE
        row.append(f"{method_metrics.get('react', {}).get('ECE') or 0:.4f}")
        row.append(f"{method_metrics.get('bot', {}).get('ECE') or 0:.4f}")
        row.append(f"{method_metrics.get('cachexl', {}).get('ECE') or 0:.4f}")

        # KS Stat
        row.append(f"{method_metrics.get('react', {}).get('KS_Stat') or 0:.4f}")
        row.append(f"{method_metrics.get('bot', {}).get('KS_Stat') or 0:.4f}")
        row.append(f"{method_metrics.get('cachexl', {}).get('KS_Stat') or 0:.4f}")

        # Strong Rate
        row.append(f"{method_metrics.get('cachexl', {}).get('escalator_rate', 0):.2f}")

        # Relative Diff and Ratio (vs BoT)
        bot_acc = method_metrics.get("bot", {}).get("accuracy", 0)
        for m in ["react", "bot", "cachexl"]:
            m_acc = method_metrics.get(m, {}).get("accuracy", 0)
            if bot_acc > 0:
                rel_diff = (m_acc - bot_acc) / bot_acc
                ratio = m_acc / bot_acc
            else:
                rel_diff = 0.0
                ratio = 0.0
            row.append(f"{rel_diff:.4f}")
            row.append(f"{ratio:.4f}")

        csv_rows.append(row)

        print(
            f"{dataset:<12} | {method_metrics.get('react', {}).get('accuracy', 0):.4f}        | {method_metrics.get('bot', {}).get('accuracy', 0):.4f}    | {method_metrics.get('cachexl', {}).get('accuracy', 0):.4f}    | {method_metrics.get('react', {}).get('avg_cost', 0):<13.6f} | {method_metrics.get('bot', {}).get('avg_cost', 0):<10.6f} | {method_metrics.get('cachexl', {}).get('avg_cost', 0):<10.6f}"
        )

    return csv_rows


def main():
    parser = argparse.ArgumentParser(description="生成统一实验报告")
    parser.add_argument(
        "--methods", default="react,bot,cachexl", help="要生成报告的方法，逗号分隔"
    )
    parser.add_argument(
        "--output-dir", default="experiments/reports", help="报告输出目录"
    )
    args = parser.parse_args()

    methods_to_report = [m.strip() for m in args.methods.split(",") if m.strip()]
    experiments_dir = PROJECT_ROOT / "experiments"
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成各方法报告
    all_csv_rows = None

    for method in methods_to_report:
        results_dir = experiments_dir / method / "results"
        if not results_dir.exists():
            print(f"[{method}] Results directory not found: {results_dir}")
            continue

        rows = generate_method_report(method, results_dir)

        # 保存单独报告
        if rows:
            csv_path = output_dir / f"{method}_report.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"[{method}] Report saved to {csv_path}")

    # 生成对比报告
    if len(methods_to_report) > 1:
        csv_rows = generate_comparison_report(experiments_dir, methods_to_report)

        comparison_path = output_dir / "comparison_report.csv"
        with open(comparison_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"\n[Comparison] Report saved to {comparison_path}")

    print(f"\nAll reports saved to: {output_dir}")


if __name__ == "__main__":
    main()
