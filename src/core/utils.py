import json
import re
import random
from pathlib import Path
from transformers import AutoTokenizer

_TOKENIZER = None

def get_tokenizer():
    """从本地资产延迟加载分词器，如果不存在则返回 None。"""
    global _TOKENIZER
    if _TOKENIZER is None:
        assets_path = Path(__file__).resolve().parents[1] / "tokenizer_assets"
        if not assets_path.exists():
            return None
        try:
            # DeepSeek 分词器需要 trust_remote_code=True
            _TOKENIZER = AutoTokenizer.from_pretrained(
                str(assets_path), 
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load tokenizer from {assets_path}: {e}")
            return None
    return _TOKENIZER

def count_tokens(text: str) -> int:
    """计算 Token 数量。如果分词器可用则使用精确计算，否则使用字符长度估算。"""
    if not text:
        return 0
    tokenizer = get_tokenizer()
    if tokenizer:
        return len(tokenizer.encode(text, add_special_tokens=False))
    else:
        # 简单的字符估算 (约4字符/token)
        return len(text) // 4


def load_items(dataset_path: str, datasets_root: Path, limit: int | None = None, shuffle: bool = False, seed: int = 42) -> list[dict]:
    # 读取数据集样本列表，支持数量限制
    items = []
    p = Path(dataset_path)
    if not p.is_absolute():
        p = (datasets_root / p).resolve()
    src = p / "data.jsonl"
    if src.exists():
        lines = src.read_text(encoding="utf-8").splitlines()
        if shuffle:
            random.seed(seed)
            random.shuffle(lines)
            
        for i, line in enumerate(lines):
            if limit and i >= limit:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            obj["id"] = obj.get("id", f"{i}")
            items.append(obj)
    return items


def load_existing_results(path: Path) -> dict[str, dict]:
    # 断点续跑读取已有批次结果，用于跳过已完成任务
    if not path.exists():
        return {}
    items = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        item_id = obj.get("id")
        if item_id is None:
            continue
        items[str(item_id)] = obj
    return items


def write_results(base_results_dir: Path, dataset_name: str, batch_idx: int, items: list[dict]):
    # 按批次落盘
    out_dir = base_results_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"batch{batch_idx}.jsonl"
    with fp.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def extract_json(text: str):
    start = None
    for i, ch in enumerate(text):
        if ch in "[{":
            start = i
            break
    if start is None:
        raise ValueError("No JSON found")
    end = None
    for j in range(len(text) - 1, -1, -1):
        if text[j] in "]}":
            end = j + 1
            break
    if end is None:
        raise ValueError("No JSON end found")
    payload = text[start:end]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # 修复无效的转义字符：将所有反斜杠替换为双反斜杠（除非它是有效的转义序列）
        # 简单粗暴的修复：如果直接加载失败，尝试转义所有反斜杠，然后再次尝试
        fixed = payload.replace("\\", "\\\\")
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
    except Exception as e:
        print(f"Error parsing JSON payload: {e}")
        raise e

        # 如果还是不行，尝试更细致的正则修复
        fixed = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", payload)
        fixed = re.sub(r"\\u(?![0-9a-fA-F]{4})", r"\\\\u", fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            repaired = repair_json(fixed)
            
            # 1. 将 Python 字面量替换为 JSON 字面量
            repaired = repaired.replace("None", "null").replace("True", "true").replace("False", "false")
            
            # 2. 移除多余的逗号: ,} -> } 和 ,] -> ]
            repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
            
            # 3. 补充缺失的逗号
            # 情况 A: 字符串之间 (涵盖 "val" "key" 和 "val" "val")
            repaired = re.sub(r"(\")\s*(\")", r"\1,\2", repaired)
            # 情况 B: 数字和字符串/键之间 (例如 123 "key")
            repaired = re.sub(r"(\d)\s*(\")", r"\1,\2", repaired)
            # 情况 C: 布尔值/null 和字符串/键之间 (例如 true "key")
            repaired = re.sub(r"(true|false|null)\s*(\")", r"\1,\2", repaired)
            # 情况 D: 右括号/花括号和字符串/键之间 (例如 } "key")
            repaired = re.sub(r"([}\]])\s*(\")", r"\1,\2", repaired)
            # 情况 E: 右括号/花括号和左括号/花括号之间 (例如 } { 或 ] [)
            repaired = re.sub(r"([}\]])\s*([{\[])", r"\1,\2", repaired)
            
            return json.loads(repaired)


def repair_json(payload: str) -> str:
    # 修复JSON字符串中的常见转义字符问题
    in_string = False
    escape = False
    out = []
    for ch in payload:
        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                in_string = False
                out.append(ch)
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            out.append(ch)
            continue
        if ch == '"':
            in_string = True
        out.append(ch)
    return "".join(out)
