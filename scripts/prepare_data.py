import os
import argparse
import io
import json
import zipfile
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

import requests
import pandas as pd
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(p: Path):
    # 确保目录存在，不存在则创建
    p.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict]):
    # 将数据列表写入JSONL文件
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def download_file(url: str) -> bytes:
    # 下载文件内容，支持超时
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


def build_choices_from_columns(row: dict, labels: list[str], answer_label: str):
    choices = [{"label": l, "text": row[l]} for l in labels]
    return choices, answer_label


def dataset_revision(name: str, default: str | None = None) -> str | None:
    return os.getenv(f"{name.upper()}_REVISION") or os.getenv("HF_DATASET_REVISION") or default


def dataset_split(name: str, default: str) -> str:
    return os.getenv(f"{name.upper()}_SPLIT") or default


def write_meta(dest: Path, meta: dict):
    ensure_dir(dest)
    (dest / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_dataset_name(name: str) -> str:
    key = name.lower()
    if "winogrande" in key:
        return "winogrande"
    if "pubmedqa" in key:
        return "pubmedqa"
    if "medqa" in key:
        return "medqa"
    if "mmlu" in key:
        return "mmlu"
    if "sms" in key or "spam" in key:
        return "sms_spam"
    if "gpqa" in key:
        return "gpqa"
    if "math" in key and "500" in key:
        return "math500"
    return key


def normalize_gpqa(dest: Path, split_override: str | None = None, revision_override: str | None = None):
    # 处理GPQA数据集：支持从本地Zip或HuggingFace加载，并标准化格式
    local_zip = PROJECT_ROOT / "data" / "_raw" / "gpqa" / "dataset.zip"
    split = split_override or dataset_split("gpqa", "train")
    revision = revision_override or dataset_revision("gpqa")
    if local_zip.exists():
        zf = zipfile.ZipFile(local_zip)
        password = b"deserted-untie-orchid"
        members = zf.namelist()
        target = None
        for name in members:
            if name.endswith("gpqa_main.csv"):
                target = name
                break
        if target is None:
            raise RuntimeError("gpqa_main.csv not found in local zip")
        with zf.open(target, pwd=password) as f:
            df = pd.read_csv(f)
        write_meta(dest, {"source": "local_zip", "path": str(local_zip), "split": split, "revision": revision})
    else:
        kwargs = {"split": split}
        if revision:
            kwargs["revision"] = revision
        ds = load_dataset("Idavidrein/gpqa", **kwargs)
        df = ds.to_pandas()
        write_meta(
            dest,
            {
                "source": "hf",
                "repo": "Idavidrein/gpqa",
                "config": None,
                "split": split,
                "revision": revision,
            },
        )
    cols = [c.lower() for c in df.columns]
    out = []
    for i, row in df.iterrows():
        rowd = {c: row[c] for c in df.columns}
        if set(["a", "b", "c", "d", "answer"]).issubset(cols):
            labels = ["A", "B", "C", "D"]
            ans_key = next((c for c in df.columns if c.lower() == "answer"), None)
            ans_val = rowd.get(ans_key) if ans_key else None
            choices, label = build_choices_from_columns(rowd, labels, ans_val)
        else:
            qcol = next((c for c in df.columns if "question" in c.lower()), None)
            ccol = next((c for c in df.columns if "correct" in c.lower()), None)
            icols = [c for c in df.columns if "incorrect" in c.lower()]
            if qcol is None or ccol is None or len(icols) < 3:
                continue
            labels = ["A", "B", "C", "D"]
            options = [rowd[ccol]] + [rowd[ic] for ic in icols[:3]]
            choices = [{"label": l, "text": t} for l, t in zip(labels, options)]
            label = "A"
        qtext = rowd.get("Question") or rowd.get("question") or rowd.get(qcol)
        out.append({"id": f"gpqa_{i}", "question": qtext, "choices": choices, "label": label})
    ensure_dir(dest)
    write_jsonl(dest / "data.jsonl", out)


def normalize_math500(dest: Path, split_override: str | None = None, revision_override: str | None = None):
    # MATH-500 (HuggingFaceH4/MATH-500)
    # Expected columns: problem, solution, answer, subject, level
    split = split_override or dataset_split("math500", "test") # Default to test as it is a test set
    revision = revision_override or dataset_revision("math500")
    kwargs = {"split": split}
    if revision:
        kwargs["revision"] = revision
    ds = load_dataset("HuggingFaceH4/MATH-500", **kwargs)
    df = ds.to_pandas()
    
    rename_map = {}
    if "problem" in df.columns:
        rename_map["problem"] = "question"
    if "answer" in df.columns:
        rename_map["answer"] = "label"
    
    df = df.rename(columns=rename_map)
    
    if "id" not in df.columns:
        df["id"] = range(len(df))
        
    write_meta(
        dest,
        {
            "source": "huggingface",
            "dataset": "HuggingFaceH4/MATH-500",
            "split": split,
            "revision": kwargs.get("revision", "main"),
        },
    )
    write_jsonl(dest / "data.jsonl", df.to_dict(orient="records"))


def normalize_winogrande(dest: Path, split_override: str | None = None, revision_override: str | None = None):
    split = split_override or dataset_split("winogrande", "validation")
    revision = revision_override or dataset_revision("winogrande")
    kwargs = {"split": split}
    if revision:
        kwargs["revision"] = revision
    ds = load_dataset("allenai/winogrande", "winogrande_debiased", **kwargs)
    out = []
    for i, row in enumerate(ds):
        choices = [{"label": "A", "text": row["option1"]}, {"label": "B", "text": row["option2"]}]
        label = "A" if row["answer"] == "1" else "B"
        out.append({"id": f"winogrande_{i}", "question": row["sentence"], "choices": choices, "label": label})
    ensure_dir(dest)
    write_jsonl(dest / "data.jsonl", out)
    write_meta(
        dest,
        {
            "source": "hf",
            "repo": "allenai/winogrande",
            "config": "winogrande_debiased",
            "split": split,
            "revision": revision,
        },
    )


def normalize_pubmedqa(dest: Path, split_override: str | None = None, revision_override: str | None = None):
    split = split_override or dataset_split("pubmedqa", "train")
    revision = revision_override or dataset_revision("pubmedqa")
    kwargs = {"split": split}
    if revision:
        kwargs["revision"] = revision
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", **kwargs)
    out = []
    for i, row in enumerate(ds):
        choices = [
            {"label": "yes", "text": "yes"},
            {"label": "no", "text": "no"},
            {"label": "maybe", "text": "maybe"},
        ]
        out.append({"id": f"pubmedqa_{i}", "question": row["question"], "choices": choices, "label": row["final_decision"]})
    ensure_dir(dest)
    write_jsonl(dest / "data.jsonl", out)
    write_meta(
        dest,
        {
            "source": "hf",
            "repo": "qiaojin/PubMedQA",
            "config": "pqa_labeled",
            "split": split,
            "revision": revision,
        },
    )


def normalize_mmlu(dest: Path, split_override: str | None = None, revision_override: str | None = None):
    split = split_override or dataset_split("mmlu", "test")
    revision = revision_override or dataset_revision("mmlu")
    kwargs = {"split": split}
    if revision:
        kwargs["revision"] = revision
    ds = load_dataset("cais/mmlu", "all", **kwargs)
    out = []
    for i, row in enumerate(ds):
        labels = ["A", "B", "C", "D"]
        choice_texts = row.get("choices") or []
        if len(choice_texts) != 4:
            continue
        choices = [{"label": l, "text": t} for l, t in zip(labels, choice_texts)]
        answer_idx = row["answer"]
        label = labels[answer_idx] if isinstance(answer_idx, int) else answer_idx
        out.append({"id": f"mmlu_{i}", "question": row["question"], "choices": choices, "label": label})
    ensure_dir(dest)
    write_jsonl(dest / "data.jsonl", out)
    write_meta(
        dest,
        {
            "source": "hf",
            "repo": "cais/mmlu",
            "config": "all",
            "split": split,
            "revision": revision,
        },
    )


def normalize_sms_spam(dest: Path):
    import random
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    content = download_file(url)
    zf = zipfile.ZipFile(io.BytesIO(content))
    name = "SMSSpamCollection"
    with zf.open(name) as f:
        lines = [line.decode("utf-8").strip() for line in f.readlines()]
    out = []
    for i, line in enumerate(lines):
        if not line:
            continue
        try:
            label, text = line.split("\t", 1)
        except ValueError:
            continue
        choices = [{"label": "ham", "text": "ham"}, {"label": "spam", "text": "spam"}]
        out.append({"id": f"sms_{i}", "question": text, "choices": choices, "label": label})
    
    # Paper uses 1510 samples. Shuffle with fixed seed 42 and take first 1510.
    random.seed(42)
    random.shuffle(out)
    out = out[:1510]
    
    ensure_dir(dest)
    write_jsonl(dest / "data.jsonl", out)
    write_meta(dest, {"source": "uci", "url": url, "split": "subset_1510_seed42"})


def normalize_medqa(dest: Path, split_override: str | None = None, revision_override: str | None = None):
    base_dir = PROJECT_ROOT / "data" / "_raw" / "MedQA"
    local_file = None
    if base_dir.exists():
        matches = list(base_dir.rglob("data_clean/questions/US/4_options/test.jsonl"))
        if matches:
            local_file = matches[0]
    if local_file and local_file.exists():
        lines = local_file.read_text(encoding="utf-8").splitlines()
        out = []
        for i, line in enumerate(lines):
            row = json.loads(line)
            opts = row.get("options") or row.get("choices") or []
            choices = [{"label": chr(65 + idx), "text": t} for idx, t in enumerate(opts)]
            out.append({"id": f"medqa_{i}", "question": row["question"], "choices": choices, "label": row.get("answer")})
        ensure_dir(dest)
        write_jsonl(dest / "data.jsonl", out)
        write_meta(
            dest,
            {
                "source": "local",
                "path": str(local_file),
                "split": "test",
                "revision": None,
            },
        )
        return
    else:
        url = "https://github.com/jind11/MedQA/archive/refs/heads/master.zip"
        content = download_file(url)
        zf = zipfile.ZipFile(io.BytesIO(content))
        target = "MedQA-master/data_clean/questions/US/4_options/test.jsonl"
        if target in zf.namelist():
            with zf.open(target) as f:
                lines = [line.decode("utf-8").strip() for line in f.readlines()]
            out = []
            for i, line in enumerate(lines):
                row = json.loads(line)
                opts = row.get("options") or row.get("choices") or []
                choices = [{"label": chr(65 + idx), "text": t} for idx, t in enumerate(opts)]
                out.append({"id": f"medqa_{i}", "question": row["question"], "choices": choices, "label": row.get("answer")})
            ensure_dir(dest)
            write_jsonl(dest / "data.jsonl", out)
            write_meta(
                dest,
                {
                    "source": "github",
                    "url": url,
                    "path": target,
                    "split": "test",
                    "revision": None,
                },
            )
            return
        split = split_override or dataset_split("medqa", "test")
        revision = revision_override or dataset_revision("medqa")
        kwargs = {"split": split}
        if revision:
            kwargs["revision"] = revision
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", **kwargs)
        out = []
        for i, row in enumerate(ds):
            options = row.get("options", {})
            labels = ["A", "B", "C", "D"]
            choices = [{"label": l, "text": options.get(l, "")} for l in labels]
            ans_idx = row.get("answer_idx")
            ans_label = labels[ans_idx] if isinstance(ans_idx, int) and 0 <= ans_idx < len(labels) else ans_idx
            out.append({"id": f"medqa_{i}", "question": row["question"], "choices": choices, "label": ans_label})
        ensure_dir(dest)
        write_jsonl(dest / "data.jsonl", out)
        write_meta(
            dest,
            {
                "source": "hf",
                "repo": "GBaker/MedQA-USMLE-4-options",
                "config": None,
                "split": split,
                "revision": revision,
            },
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--dataset", required=False)
    args = ap.parse_args()
    root = Path(args.root)

    def get_overrides(ds_name: str):
        return (None, None)
    if args.dataset:
        name = args.dataset.lower()
        s, r = get_overrides(name)
        name = normalize_dataset_name(name)
        if name == "gpqa":
            normalize_gpqa(root / "gpqa", s, r)
        elif name == "winogrande":
            normalize_winogrande(root / "winogrande_debiased", s, r)
        elif name == "pubmedqa":
            normalize_pubmedqa(root / "pubmedqa", s, r)
        elif name == "medqa":
            normalize_medqa(root / "medqa_usmle", s, r)
        elif name == "mmlu":
            normalize_mmlu(root / "mmlu", s, r)
        elif name == "sms_spam":
            normalize_sms_spam(root / "sms_spam")
        elif name == "math500":
            normalize_math500(root / "math500", s, r)
        else:
            raise RuntimeError(f"Unknown dataset: {name}")
        return
    plan = [
        "gpqa",
        "winogrande",
        "pubmedqa",
        "medqa",
        "mmlu",
        "sms_spam",
        "math500",
    ]
    for p in plan:
        print(f"Processing {p}...")
        try:
            s, r = get_overrides(p)
            if p == "gpqa":
                normalize_gpqa(root / "gpqa", s, r)
            elif p == "winogrande":
                normalize_winogrande(root / "winogrande", s, r)
            elif p == "pubmedqa":
                normalize_pubmedqa(root / "pubmedqa", s, r)
            elif p == "medqa":
                normalize_medqa(root / "medqa", s, r)
            elif p == "mmlu":
                normalize_mmlu(root / "mmlu", s, r)
            elif p == "sms_spam":
                normalize_sms_spam(root / "sms_spam")
            elif p == "math500":
                normalize_math500(root / "math500", s, r)
        except Exception as e:
            print(f"Error processing {p}: {e}")


if __name__ == "__main__":
    main()
