import json
from core.utils import extract_json
from .prompts.loader import get_reflector_prompt


def build_reflector_messages(
    items: list[dict], dataset_name: str = "default", method: str = "react"
) -> list[dict]:
    # 构建Reflector的输入Prompt，包含批量题目和Actor答案
    batch = []
    for item in items:
        entry = {
            "id": item["id"],
            "question": item["question"],
            "choices": item.get("choices", []),
            "answer": item.get("answer", ""),
            "rationale": item.get("rationale", ""),
        }

        # 添加历史轨迹信息
        history = item.get("history", [])
        if history:
            history_str = ""
            for h in history:
                hist_round = h.get("round", "?")
                hist_ans = h.get("answer", "")
                hist_rat = h.get("rationale", "")[:200] if h.get("rationale") else ""
                hist_crit = h.get("critique", "")[:200] if h.get("critique") else ""
                history_str += f"- Round {hist_round}: answer={hist_ans}, rationale={hist_rat}, critique={hist_crit}\n"
            entry["history"] = history_str.strip()

        # 添加当前 critique（如果有）
        if item.get("critique"):
            entry["current_critique"] = item["critique"][:300]

        batch.append(entry)

    # 获取指定方法和数据集的Prompt模板
    template = get_reflector_prompt(method, dataset_name)

    # 填充模板
    content = template.format(
        batch_json=json.dumps(batch, ensure_ascii=False, indent=2)
    )

    prompt = {
        "role": "user",
        "content": content,
    }
    return [prompt]


def parse_reflector_response(text: str) -> list[dict]:
    # 解析Reflector的JSON响应，返回评估结果列表
    data = extract_json(text)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Invalid reflector response")
