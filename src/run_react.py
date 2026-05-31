import os
import argparse
import json
import time
import sys
import random
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.config import load_config
from core.llm import LLMClient
from methods.react.actor import build_actor_messages, parse_actor_response
from methods.react.reflector import (
    build_reflector_messages,
    parse_reflector_response,
)
from methods.react.batching import make_batches
from methods.react.eval import simple_accuracy, approx_tokens


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 定义项目根目录，用于路径解析


def parse_dataset_filter(value) -> set[str] | None:
    # 支持字符串或列表形式的过滤配置
    if value is None:
        return None
    if isinstance(value, str):
        parts = [v.strip() for v in value.split(",")]
    elif isinstance(value, list):
        parts = [str(v).strip() for v in value]
    else:
        return None
    selected = {p.lower() for p in parts if p}
    return selected or None


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


def load_items(
    dataset_path: str,
    datasets_root: Path,
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict]:
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


def write_results(
    base_results_dir: Path, dataset_name: str, batch_idx: int, items: list[dict]
):
    # 按批次落盘
    out_dir = base_results_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"batch{batch_idx}.jsonl"
    with fp.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    # 主程序入口，协调配置加载、模型初始化和任务调度
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit", type=int, default=0, help="限制每个数据集的样本数量（0表示全部）"
    )
    ap.add_argument("--dataset", default=None, help="指定运行的数据集，逗号分隔")
    ap.add_argument(
        "--datasets-dir", default=str(PROJECT_ROOT / "data"), help="数据集目录路径"
    )
    ap.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "experiments/react/results"),
        help="结果输出目录",
    )
    args = ap.parse_args()
    cfg = load_config()

    # 根据方法选择调整默认输出目录
    results_dir = Path(args.results_dir)

    # 使用命令行参数覆盖默认路径
    datasets_dir = Path(args.datasets_dir)
    http_cfg = {
        "ssl_verify": cfg.model.ssl_verify
        if cfg.model.ssl_verify is not None
        else True,
        "httpx_trust_env": cfg.model.httpx_trust_env
        if cfg.model.httpx_trust_env is not None
        else True,
        "http_timeout": cfg.model.http_timeout,
        "http_connect_timeout": cfg.model.http_connect_timeout,
        "http_read_timeout": cfg.model.http_read_timeout or cfg.model.http_timeout,
        "http_write_timeout": cfg.model.http_write_timeout,
        "http_pool_timeout": cfg.model.http_pool_timeout,
        "retry_attempts": cfg.model.retry_attempts,
        "retry_backoff": cfg.model.retry_backoff,
        "allow_fallback": cfg.model.allow_fallback
        if cfg.model.allow_fallback is not None
        else False,
        "verbose": cfg.model.verbose if cfg.model.verbose is not None else True,
    }
    client = LLMClient(cfg.model.base_url, cfg.model.api_key, config=http_cfg)

    model_params = {}
    if cfg.model.top_p is not None:
        model_params["top_p"] = cfg.model.top_p
    if cfg.model.max_tokens is not None:
        model_params["max_tokens"] = cfg.model.max_tokens
    if cfg.model.seed is not None:
        model_params["seed"] = cfg.model.seed
    if cfg.model.stop is not None:
        model_params["stop"] = cfg.model.stop
    if cfg.model.presence_penalty is not None:
        model_params["presence_penalty"] = cfg.model.presence_penalty
    if cfg.model.frequency_penalty is not None:
        model_params["frequency_penalty"] = cfg.model.frequency_penalty
    concurrency = cfg.experiment.concurrency or 1
    resume = cfg.experiment.resume if cfg.experiment.resume is not None else False
    # CLI 参数优先，其次读取配置
    limit = args.limit if args.limit > 0 else cfg.experiment.limit
    selected = parse_dataset_filter(args.dataset) or parse_dataset_filter(
        cfg.experiment.dataset_filter
    )
    datasets = [
        ds for ds in cfg.datasets if not selected or ds.name.lower() in selected
    ]
    total_datasets = len(datasets)
    for di, ds in enumerate(datasets, start=1):
        # 数据集级别统计
        items = load_items(
            ds.path,
            datasets_root=datasets_dir,
            limit=limit,
            shuffle=cfg.experiment.shuffle,
            seed=cfg.experiment.seed,
        )
        if not items:
            continue
        batches = make_batches(items, cfg.experiment.batch_size)
        total_batches = len(batches)
        dataset_total = len(items)
        dataset_done = 0
        print(
            f"{ds.name} dataset {di}/{total_datasets}: {len(items)} samples, {total_batches} batches"
        )

        def run_with_retry(base_messages, parse_fn, retry_hint: str):
            # 封装带有重试逻辑的LLM请求，处理解析错误
            for attempt in range(3):
                send_messages = base_messages
                if attempt > 0:
                    send_messages = [
                        {
                            "role": "user",
                            "content": base_messages[0]["content"] + f"\n{retry_hint}",
                        }
                    ]
                resp = client.chat(
                    cfg.model.backend,
                    send_messages,
                    cfg.model.temperature,
                    model_params,
                )
                text = resp["choices"][0]["message"]["content"]
                try:
                    parsed = parse_fn(text)
                except Exception:
                    if attempt == 2:
                        raise
                    continue
                return parsed, resp, text, send_messages
            raise ValueError("Retry attempts exhausted")

        def add_tokens(target: dict, key: str, tokens: dict):
            # 累加Token消耗统计
            current = target.get(key, {"prompt": 0.0, "completion": 0.0, "total": 0.0})
            current["prompt"] += tokens["prompt"]
            current["completion"] += tokens["completion"]
            current["total"] += tokens["total"]
            target[key] = current

        def run_actor(item: dict) -> dict:
            # 执行Actor阶段，生成初始答案
            base_messages = build_actor_messages(
                item, dataset_name=ds.name, method="react"
            )
            parsed, resp, text, send_messages = run_with_retry(
                base_messages, parse_actor_response, "Output only a valid JSON object."
            )
            usage = resp.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens") or sum(
                approx_tokens(m.get("content", "")) for m in send_messages
            )
            completion_tokens = usage.get("completion_tokens") or approx_tokens(text)
            total_tokens = usage.get("total_tokens") or (
                prompt_tokens + completion_tokens
            )
            return {
                "id": item["id"],
                "answer": parsed.get("answer"),
                "rationale": parsed.get("rationale"),
                "confidence": parsed.get("confidence"),
                "actor_tokens": {
                    "prompt": float(prompt_tokens),
                    "completion": float(completion_tokens),
                    "total": float(total_tokens),
                },
            }

        def run_reflector(
            messages: list[dict],
        ) -> tuple[list[dict], dict, str, list[dict]]:
            # 执行Reflector阶段，评估并优化答案
            parsed, resp, text, used_messages = run_with_retry(
                messages, parse_reflector_response, "Output only a valid JSON array."
            )
            return parsed, resp, text, used_messages

        for bi, batch in enumerate(batches):
            result_path = results_dir / ds.name / f"batch{bi}.jsonl"
            # 断点续跑：加载已有批次结果
            existing_map = load_existing_results(result_path) if resume else {}
            items_by_id = {it["id"]: it for it in batch}
            # 覆盖已有结果，保留断点续跑数据
            for item_id, existing in existing_map.items():
                if item_id in items_by_id:
                    items_by_id[item_id].update(existing)
            for it in items_by_id.values():
                it.setdefault(
                    "actor_tokens", {"prompt": 0.0, "completion": 0.0, "total": 0.0}
                )
                it.setdefault(
                    "reflect_tokens", {"prompt": 0.0, "completion": 0.0, "total": 0.0}
                )
            # 已完成样本直接跳过
            completed_ids = {
                iid
                for iid, it in items_by_id.items()
                if it.get("final_answer") is not None
            }
            active_ids = set(items_by_id.keys()) - completed_ids
            if resume and not active_ids and items_by_id:
                dataset_done += len(items_by_id)
                print(f"{ds.name} batch {bi + 1}/{total_batches} skipped (completed)")
                print(f"{ds.name} dataset progress {dataset_done}/{dataset_total}")
                continue
            if resume and existing_map:
                print(
                    f"{ds.name} batch {bi + 1}/{total_batches} resume {len(completed_ids)}/{len(items_by_id)} completed"
                )

            batch_start_time = time.time()

            # --- Baseline Method Execution ---
            max_rounds = max(1, cfg.experiment.max_rounds)
            for ri in range(max_rounds):
                if not active_ids:
                    break
                # print(f"{ds.name} batch {bi + 1}/{total_batches} round {ri + 1}/{max_rounds} active {len(active_ids)}/{len(items_by_id)}")
                # 进度条显示在 stdout，避免污染 error.log
                print(f"{ds.name} b{bi + 1}/{total_batches} actor running...")
                round_items = [
                    items_by_id[i] for i in items_by_id.keys() if i in active_ids
                ]
                if concurrency <= 1:
                    for it in tqdm(
                        round_items,
                        desc=f"{ds.name} b{bi + 1}/{total_batches} actor r{ri + 1}",
                        file=sys.stdout,
                    ):
                        result = run_actor(it)
                        target = items_by_id[result["id"]]
                        target["answer"] = result["answer"]
                        target["rationale"] = result["rationale"]
                        target["confidence"] = result["confidence"]
                        add_tokens(target, "actor_tokens", result["actor_tokens"])
                else:
                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        futures = [executor.submit(run_actor, it) for it in round_items]
                        for fut in tqdm(
                            as_completed(futures),
                            total=len(futures),
                            desc=f"{ds.name} b{bi + 1}/{total_batches} actor r{ri + 1}",
                            file=sys.stdout,
                        ):
                            result = fut.result()
                            target = items_by_id[result["id"]]
                            target["answer"] = result["answer"]
                            target["rationale"] = result["rationale"]
                            target["confidence"] = result["confidence"]
                            add_tokens(target, "actor_tokens", result["actor_tokens"])
                messages = build_reflector_messages(
                    list(items_by_id.values()), dataset_name=ds.name, method="react"
                )
                # print(f"{ds.name} batch {bi + 1}/{total_batches} reflector r{ri + 1} start")
                decisions, rresp, rtext, used_messages = run_reflector(messages)

                rusage = rresp.get("usage") or {}
                rprompt_tokens = rusage.get("prompt_tokens") or sum(
                    approx_tokens(m.get("content", "")) for m in used_messages
                )
                rcompletion_tokens = rusage.get("completion_tokens") or approx_tokens(
                    rtext
                )
                rtotal_tokens = rusage.get("total_tokens") or (
                    rprompt_tokens + rcompletion_tokens
                )
                denom = max(len(items_by_id), 1)
                per_item_prompt = rprompt_tokens / denom
                per_item_completion = rcompletion_tokens / denom
                per_item_total = rtotal_tokens / denom
                final_map = {d["id"]: d for d in decisions if d.get("id") is not None}
                next_active = set()
                for it in items_by_id.values():
                    did = it["id"]
                    if did in final_map:
                        decision = final_map[did]
                        it["final_answer"] = decision.get(
                            "final_answer", it.get("answer")
                        )
                        it["reflect_confidence"] = decision.get("confidence")
                        it["critique"] = decision.get("critique")
                        refine = decision.get("refine")
                        refine_flag = refine is True or str(refine).strip() in {
                            "1",
                            "true",
                            "True",
                        }
                        if refine_flag:
                            next_active.add(did)
                    else:
                        it["final_answer"] = it.get("answer")
                    add_tokens(
                        it,
                        "reflect_tokens",
                        {
                            "prompt": float(per_item_prompt),
                            "completion": float(per_item_completion),
                            "total": float(per_item_total),
                        },
                    )
                active_ids = next_active
                # print(f"{ds.name} batch {bi + 1}/{total_batches} reflector r{ri + 1} done")

            # 计算并分发延迟 (Latency)
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            if items_by_id:
                per_item_latency = batch_duration / len(items_by_id)
                for it in items_by_id.values():
                    it["latency"] = per_item_latency

            enriched = list(items_by_id.values())
            acc = simple_accuracy(enriched)  # 3. 结果落盘
            try:
                write_results(results_dir, ds.name, bi, enriched)
                # 验证文件是否存在
                saved_path = results_dir / ds.name / f"batch{bi}.jsonl"
                if not saved_path.exists():
                    print(f"Warning: Result file not found after writing: {saved_path}")
                else:
                    print(f"Saved batch {bi} to {saved_path}")
            except Exception as e:
                print(f"Error writing results for {ds.name} batch {bi}: {e}")
                import traceback

                traceback.print_exc()

            dataset_done += len(items_by_id)
            # if acc is not None:
            #    print(f"{ds.name} batch {bi + 1}/{total_batches} acc {acc:.4f}")
            print(
                f"{ds.name} progress: {dataset_done}/{dataset_total} | batch {bi + 1}/{total_batches} acc: {acc if acc is not None else 0:.4f}"
            )


if __name__ == "__main__":
    main()
