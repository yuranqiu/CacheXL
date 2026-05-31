import json
import re
from core.utils import extract_json
from .prompts.loader import get_actor_prompt


def build_actor_messages(
    item: dict, dataset_name: str = "default", method: str = "react"
) -> list[dict]:
    """
    构建Actor的输入Prompt。

    Actor 是负责解决问题的代理。它接收问题和选项，并被要求提供逐步的推理过程（Rationale）和最终答案。

    Args:
        item (dict): 包含题目信息的字典，通常包括 "question", "choices" 等字段。
        dataset_name (str): 数据集名称，用于选择特定的 Prompt 模板。默认为 "default"。

    Returns:
        list[dict]: 包含系统或用户消息的列表，用于发送给 LLM。
    """
    # 获取问题文本
    question = item["question"]

    # 处理选项：将选项列表格式化为 "Label. Text" 的形式
    choices = item.get("choices", [])
    choices_text = "\n".join([f"{c['label']}. {c['text']}" for c in choices])

    # 处理之前的反思（如果有）：这用于多轮优化
    critique = item.get("critique")
    critique_text = f"\nPrevious reflection: {critique}" if critique else ""

    # 获取指定方法和数据集的Prompt模板
    template = get_actor_prompt(method, dataset_name)

    # 填充模板
    content = template.format(
        question=question, choices_text=choices_text, critique_text=critique_text
    )

    # 构建消息对象
    prompt = {
        "role": "user",
        "content": content,
    }
    return [prompt]


def parse_actor_response(text: str) -> dict:
    """
    解析Actor的响应文本。

    期望 LLM 返回一个 JSON 对象，包含 "rationale", "answer", "confidence" 等字段。
    如果直接解析失败，会尝试使用正则表达式提取 JSON 部分。

    Args:
        text (str): LLM 返回的原始文本响应。

    Returns:
        dict: 解析后的 JSON 对象。如果解析失败，返回 None。
    """
    # 尝试直接解析整个文本
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 如果直接解析失败，尝试提取代码块中的 JSON
    extracted = extract_json(text)
    if extracted:
        return extracted

    # 如果仍然失败，尝试使用正则进行容错解析
    return fallback_parse_actor(text)


def fallback_parse_actor(text: str) -> dict:
    # 使用正则进行容错解析，提取answer/rationale/confidence
    answer = None
    rationale = None
    confidence = None
    m = re.search(r'"answer"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if not m:
        m = re.search(
            r"\banswer\b\s*[:=]\s*([A-D]|yes|no|maybe)\b", text, re.IGNORECASE
        )
    if m:
        answer = m.group(1)
    m = re.search(r'"rationale"\s*:\s*"([^"]*)"', text, re.IGNORECASE | re.DOTALL)
    if m:
        rationale = m.group(1).strip()
    m = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
    if not m:
        m = re.search(r"\bconfidence\b\s*[:=]\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if m:
        confidence = float(m.group(1))
    if answer is None:
        answer = "A"
    if rationale is None:
        rationale = ""
    if confidence is None:
        confidence = 0.0
    return {"answer": answer, "rationale": rationale, "confidence": confidence}
