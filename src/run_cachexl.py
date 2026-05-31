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
from core.utils import load_items, load_existing_results, write_results
from methods.react.batching import make_batches
from methods.react.eval import simple_accuracy, approx_tokens
from methods.cachexl.workflow import run_cachexl_batch
from methods.cachexl.cache import EvidenceCache

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def main():
    # CacheXL 方法主程序入口
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
        default=str(PROJECT_ROOT / "experiments/cachexl/results"),
        help="结果输出目录",
    )
    args = ap.parse_args()
    cfg = load_config()

    results_dir = Path(args.results_dir)
    datasets_dir = Path(args.datasets_dir)

    # 1. 初始化 LLM Client (用于推理)
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

    # 2. 初始化 EvidenceCache
    # 注意: EvidenceCache 内部会自己初始化一个 LLMClient 用于 Embedding，避免并发冲突
    evidence_cache = EvidenceCache(capacity=100)

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
    # Check limit from config if args.limit is 0
    limit = args.limit if args.limit > 0 else cfg.experiment.limit

    # Check filter from config if args.dataset is None
    selected_filter = parse_dataset_filter(args.dataset) or parse_dataset_filter(
        cfg.experiment.dataset_filter
    )

    datasets = []
    for ds in cfg.datasets:
        if selected_filter and ds.name.lower() not in selected_filter:
            continue
        datasets.append(ds)

    total_datasets = len(datasets)
    print(f"Running CacheXL experiment on {total_datasets} datasets.")

    for di, ds in enumerate(datasets, start=1):
        # Using load_items from utils
        items = load_items(
            ds.path,
            datasets_root=datasets_dir,
            limit=limit,
            shuffle=cfg.experiment.shuffle,
            seed=cfg.experiment.seed,
        )
        if not items:
            print(f"Skipping empty dataset: {ds.name}")
            continue

        batches = make_batches(items, cfg.experiment.batch_size)
        total_batches = len(batches)

        print(
            f"[{ds.name}] Dataset {di}/{total_datasets}: {len(items)} samples, {total_batches} batches"
        )

        # 加载已完成结果 (Resume logic)
        existing_results = {}
        dataset_results_dir = results_dir / ds.name
        if cfg.experiment.resume and dataset_results_dir.exists():
            # Only load existing results if resume is true
            for f in dataset_results_dir.glob("batch*.jsonl"):
                existing_results.update(load_existing_results(f))

        pbar = tqdm(batches, desc=f"Processing {ds.name}", file=sys.stdout)

        for bi, batch in enumerate(pbar, start=1):
            # 检查批次是否已完成
            # Check if all items in this batch are already done
            all_done = True
            for item in batch:
                if str(item["id"]) not in existing_results:
                    all_done = False
                    break
            if all_done and cfg.experiment.resume:
                continue

            # 运行 CacheXL Batch
            batch_start_time = time.time()
            results = run_cachexl_batch(
                client=client,
                model_name=cfg.model.backend,
                model_params=model_params,
                batch=batch,
                cache=evidence_cache,
                dataset_name=ds.name,
                concurrency=concurrency,
            )
            batch_end_time = time.time()

            # 计算并分发延迟 (Latency)
            batch_duration = batch_end_time - batch_start_time
            if results:
                per_item_latency = batch_duration / len(results)
                for it in results:
                    it["latency"] = per_item_latency

            # 写入结果
            write_results(results_dir, ds.name, bi, results)


if __name__ == "__main__":
    main()
