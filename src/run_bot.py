import argparse
import time
import sys
from pathlib import Path
from tqdm import tqdm
from core.config import load_config
from core.llm import LLMClient
from core.utils import load_items, load_existing_results, write_results
from methods.react.batching import make_batches
from methods.bot.workflow import run_bot_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_dataset_filter(value) -> set[str] | None:
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
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit", type=int, default=0, help="Limit samples per dataset (0=all)"
    )
    ap.add_argument("--dataset", default=None, help="Dataset filter")
    ap.add_argument(
        "--datasets-dir", default=str(PROJECT_ROOT / "data"), help="Datasets directory"
    )
    ap.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "experiments/bot/results"),
        help="Results directory",
    )
    args = ap.parse_args()
    cfg = load_config()

    results_dir = Path(args.results_dir)
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

    # Check limit from config if args.limit is 0
    limit = args.limit if args.limit > 0 else cfg.experiment.limit

    selected_filter = parse_dataset_filter(args.dataset) or parse_dataset_filter(
        cfg.experiment.dataset_filter
    )

    datasets = []
    for ds in cfg.datasets:
        if selected_filter and ds.name.lower() not in selected_filter:
            continue
        datasets.append(ds)

    total_datasets = len(datasets)
    print(f"Running BoT (Reproduction) experiment on {total_datasets} datasets.")

    for di, ds in enumerate(datasets, start=1):
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

        existing_results = {}
        dataset_results_dir = results_dir / ds.name
        if cfg.experiment.resume and dataset_results_dir.exists():
            for f in dataset_results_dir.glob("batch*.jsonl"):
                existing_results.update(load_existing_results(f))

        pbar = tqdm(batches, desc=f"Processing {ds.name}", file=sys.stdout)

        for bi, batch in enumerate(pbar, start=1):
            all_done = True
            for item in batch:
                if str(item["id"]) not in existing_results:
                    all_done = False
                    break
            if all_done and cfg.experiment.resume:
                continue

            batch_start_time = time.time()
            results = run_bot_batch(
                items=batch,
                llm=client,
                model_name=cfg.model.backend,
                config=cfg.experiment,
                dataset_name=ds.name,
                temperature=cfg.model.temperature,
            )
            batch_end_time = time.time()

            batch_duration = batch_end_time - batch_start_time
            if results:
                per_item_latency = batch_duration / len(results)
                for it in results:
                    it["latency"] = per_item_latency

            write_results(results_dir, ds.name, bi, results)


if __name__ == "__main__":
    main()
