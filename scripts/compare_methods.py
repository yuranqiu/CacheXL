import json
import csv
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Cost Constants (per 1M tokens)
COST_EMBEDDING_INPUT = 0.10
COST_LLM_INPUT = 2.50
COST_LLM_OUTPUT = 10.00

# 数据集格式映射
# label_to_letter: label 是选项字母(A/B/C/D)，直接比较字母
# single_choice_text: label 固定为 "A"，答案是 choices[0].text (gsm8k)
# direct_text: label 直接是答案文本 (pubmedqa, math500)

DATASET_FORMAT = {
    "mmlu": "label_to_letter",
    "ai2_arc": "label_to_letter",
    "csqa": "label_to_letter",
    "gpqa": "label_to_letter",
    "medqa": "label_to_letter",
    "medqa_usmle": "label_to_letter",
    "winogrande": "label_to_letter",
    "aicrypto": "label_to_letter",
    "gsm8k": "single_choice_text",
    "strategyqa": "label_to_letter",
    "pubmedqa": "direct_text",
    "math500": "direct_text",
}


def load_original_data(data_dir: Path, dataset: str) -> dict:
    """加载原始数据集，返回 id -> ground_truth_label 的映射"""
    data_file = data_dir / dataset / "data.jsonl"
    if not data_file.exists():
        return {}

    ground_truth_map = {}
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    item_id = item.get("id")
                    if not item_id:
                        continue

                    fmt = DATASET_FORMAT.get(dataset, "label_to_letter")
                    label = item.get("label")
                    choices = item.get("choices", [])

                    if fmt == "label_to_letter":
                        # 对于多选数据集，ground truth 是选项字母
                        ground_truth_map[item_id] = str(label) if label else ""

                    elif fmt == "single_choice_text":
                        # gsm8k: 答案是 choices[0].text
                        if choices and isinstance(choices[0], dict):
                            ground_truth_map[item_id] = str(
                                choices[0].get("text", "")
                            ).strip()
                        else:
                            ground_truth_map[item_id] = str(label) if label else ""

                    elif fmt == "direct_text":
                        # pubmedqa, math500: 直接是答案文本
                        ground_truth_map[item_id] = str(label) if label else ""

                    else:
                        ground_truth_map[item_id] = str(label) if label else ""

                except:
                    pass
    return ground_truth_map


def get_ground_truth(item: dict, dataset: str) -> str:
    """
    根据数据集格式获取标准答案
    """
    label = item.get("label") or item.get("gold") or item.get("answer")
    choices = item.get("choices", [])

    fmt = DATASET_FORMAT.get(dataset, "label_to_choice_text")

    if fmt == "label_to_choice_text":
        # label 是 A/B/C/D，需要找到对应的 text
        if choices:
            for c in choices:
                if isinstance(c, dict):
                    if c.get("label") == label:
                        return str(c.get("text", "")).strip()
                elif isinstance(c, str):
                    if c == label:
                        return c
        # 如果没找到，返回 label
        return str(label) if label else ""

    elif fmt == "single_choice_text":
        # gsm8k 格式: label 固定是 A，答案是 choices[0].text
        if choices and isinstance(choices[0], dict):
            return str(choices[0].get("text", "")).strip()
        return str(label) if label else ""

    elif fmt == "direct_text":
        # pubmedqa, math500: label 直接是答案
        return str(label) if label else ""

    return str(label) if label else ""


def get_prediction(item: dict, method: str) -> str:
    """
    根据方法和数据格式获取预测答案
    """
    # 优先使用 final_answer（大多数方法的最终答案）
    if "final_answer" in item:
        return item.get("final_answer")

    # BoT 和 react 可能只有 answer 字段
    if "answer" in item:
        return item.get("answer")

    return None


def normalize_answer(pred) -> str:
    """标准化答案用于比较"""
    if pred is None:
        return ""
    return str(pred).strip().upper()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    items.append(json.loads(line))
                except:
                    pass
    return items


def calculate_metrics(
    items: List[Dict[str, Any]],
    method: str,
    dataset: str = "",
    ground_truth_map: dict = None,
) -> Dict[str, Any]:
    if not items:
        return {}

    total = len(items)
    correct = 0
    total_latency = 0.0

    total_prompt = 0.0
    total_completion = 0.0
    total_embedding = 0.0

    strong_count = 0

    for item in items:
        # 从原始数据集获取 ground truth
        item_id = item.get("id", "")
        if ground_truth_map and item_id in ground_truth_map:
            ground_truth = ground_truth_map[item_id]
        else:
            # 回退到从 item 中获取
            ground_truth = get_ground_truth(item, dataset)

        # 获取预测答案
        prediction = get_prediction(item, method)

        # 标准化后比较
        gt_norm = normalize_answer(ground_truth)
        pred_norm = normalize_answer(prediction)

        if gt_norm == pred_norm and gt_norm:
            correct += 1

        # Latency
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

        elif method == "bot":
            at = item.get("actor_tokens", {})
            total_prompt += float(at.get("prompt", 0))
            total_completion += float(at.get("completion", 0))

            rt = item.get("reflect_tokens", {})
            total_prompt += float(rt.get("prompt", 0))
            total_completion += float(rt.get("completion", 0))

        else:  # react
            at = item.get("actor_tokens", {})
            total_prompt += float(at.get("prompt", 0))
            total_completion += float(at.get("completion", 0))

    avg_latency = total_latency / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0

    cost_embedding = (total_embedding / 1_000_000) * COST_EMBEDDING_INPUT
    cost_input = (total_prompt / 1_000_000) * COST_LLM_INPUT
    cost_output = (total_completion / 1_000_000) * COST_LLM_OUTPUT
    total_cost = cost_embedding + cost_input + cost_output
    avg_cost = total_cost / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_latency": avg_latency,
        "avg_cost": avg_cost,
        "avg_tokens": (total_prompt + total_completion + total_embedding) / total
        if total > 0
        else 0,
        "escalator_rate": strong_count / total if total > 0 else 0,
        "details": {
            "embedding_tokens": total_embedding,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
        },
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    experiments_dir = project_root / "experiments"
    data_dir = project_root / "data"

    methods = ["react", "bot", "cachexl"]

    # 获取所有数据集
    datasets = set()
    for m in methods:
        results_dir = experiments_dir / m / "results"
        if results_dir.exists():
            for d in results_dir.iterdir():
                if d.is_dir():
                    datasets.add(d.name)

    # 过滤掉不存在的数据集
    valid_datasets = []
    for ds in sorted(datasets):
        if (data_dir / ds / "data.jsonl").exists():
            valid_datasets.append(ds)
        else:
            print(f"[Warning] Skipping {ds}: data file not found")

    print(f"Found {len(valid_datasets)} datasets: {valid_datasets}")

    csv_rows = []
    headers = [
        "Dataset",
        "ReAct_Acc",
        "ReAct_Correct",
        "ReAct_Total",
        "BoT_Acc",
        "BoT_Correct",
        "BoT_Total",
        "CacheXL_Acc",
        "CacheXL_Correct",
        "CacheXL_Total",
        "ReAct_Cost",
        "BoT_Cost",
        "CacheXL_Cost",
        "ReAct_Latency",
        "BoT_Latency",
        "CacheXL_Latency",
        "CacheXL_EscalatorRate",
    ]
    csv_rows.append(headers)

    print(
        f"\n{'Dataset':<15} | {'ReAct':<12} | {'BoT':<12} | {'CacheXL':<12} | {'ReAct Cost':<12} | {'BoT Cost':<12} | {'Enh Cost':<12}"
    )
    print("-" * 110)

    for dataset in valid_datasets:
        method_metrics = {}

        # 加载原始数据集的 ground truth
        ground_truth_map = load_original_data(data_dir, dataset)

        for m in methods:
            results_dir = experiments_dir / m / "results" / dataset
            items = []
            if results_dir.exists():
                for f in results_dir.glob("*.jsonl"):
                    items.extend(load_jsonl(f))

            metrics = calculate_metrics(items, m, dataset, ground_truth_map)
            method_metrics[m] = metrics

        react = method_metrics.get("react", {})
        bot = method_metrics.get("bot", {})
        enh = method_metrics.get("cachexl", {})

        row = [
            dataset,
            f"{react.get('accuracy', 0):.4f}",
            react.get("correct", 0),
            react.get("total", 0),
            f"{bot.get('accuracy', 0):.4f}",
            bot.get("correct", 0),
            bot.get("total", 0),
            f"{enh.get('accuracy', 0):.4f}",
            enh.get("correct", 0),
            enh.get("total", 0),
            f"{react.get('avg_cost', 0):.6f}",
            f"{bot.get('avg_cost', 0):.6f}",
            f"{enh.get('avg_cost', 0):.6f}",
            f"{react.get('avg_latency', 0):.2f}",
            f"{bot.get('avg_latency', 0):.2f}",
            f"{enh.get('avg_latency', 0):.2f}",
            f"{enh.get('escalator_rate', 0):.2f}",
        ]
        csv_rows.append(row)

        print(
            f"{dataset:<15} | {react.get('accuracy', 0):.4f} ({react.get('correct', 0):>3}/{react.get('total', 0):>3}) | "
            f"{bot.get('accuracy', 0):.4f} ({bot.get('correct', 0):>3}/{bot.get('total', 0):>3}) | "
            f"{enh.get('accuracy', 0):.4f} ({enh.get('correct', 0):>3}/{enh.get('total', 0):>3}) | "
            f"{react.get('avg_cost', 0):.6f}   | {bot.get('avg_cost', 0):.6f}   | {enh.get('avg_cost', 0):.6f}"
        )

    # Write CSV
    output_csv_path = experiments_dir / "comparison_report_fixed.csv"
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"\nSaved to {output_csv_path}")


if __name__ == "__main__":
    main()
