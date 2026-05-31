import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


@dataclass
class ExperimentConfig:
    # 实验配置类: 定义实验名称、批次大小、并发数等
    name: str
    batch_size: int
    max_rounds: int
    max_tool_calls: int
    dataset_filter: list[str] | None = None
    concurrency: int | None = None
    resume: bool | None = None
    limit: int = 0
    shuffle: bool = False
    seed: int = 42
    # CacheXL 特有参数
    tau_l: float = 0.5  # 升级阈值
    tau_h: float = 0.8  # 缓存准入阈值
    cache_capacity: int = 100  # 缓存容量
    retrieval_top_k: int = 2  # 检索 top-k
    retrieval_threshold: float = 0.8  # 检索相似度阈值


@dataclass
class ModelConfig:
    # 模型配置类: 定义后端、API参数、超时重试等
    backend: str
    temperature: float
    base_url: str
    api_key: str
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    stop: list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    http_timeout: float | None = None
    http_connect_timeout: float | None = None
    http_read_timeout: float | None = None
    http_write_timeout: float | None = None
    http_pool_timeout: float | None = None
    retry_attempts: int | None = None
    retry_backoff: float | None = None
    ssl_verify: bool | None = None
    httpx_trust_env: bool | None = None
    allow_fallback: bool | None = None
    verbose: bool | None = None


@dataclass
class DatasetConfig:
    # 数据集配置类: 定义名称、路径、切分等
    name: str
    path: str
    split: str | None = None
    revision: str | None = None


@dataclass
class PricingConfig:
    # 计费配置类: 定义输入输出价格
    prompt_per_1k: float = 0.0
    completion_per_1k: float = 0.0


@dataclass
class EmbeddingConfig:
    # Embedding 专用配置：与大模型 API 隔离
    base_url: str
    api_key: str
    model: str = "nvidia/nv-embedqa-e5-v5"
    http_timeout: float = 30.0


@dataclass
class Config:
    # 全局配置类: 聚合所有子配置
    experiment: ExperimentConfig
    model: ModelConfig
    embedding: EmbeddingConfig
    datasets: list[DatasetConfig]
    metrics: list[str]
    pricing: PricingConfig


def load_config() -> Config:
    # 从环境变量加载配置

    # 实验配置
    dataset_filter_str = os.getenv("BOTR_DATASET_FILTER", "")
    dataset_filter = (
        [x.strip() for x in dataset_filter_str.split(",")]
        if dataset_filter_str
        else None
    )

    experiment = ExperimentConfig(
        name=os.getenv("BOTR_EXPERIMENT_NAME", "cachexl"),
        batch_size=int(os.getenv("BOTR_BATCH_SIZE", "8")),
        max_rounds=int(os.getenv("BOTR_MAX_ROUNDS", "1")),
        max_tool_calls=int(os.getenv("BOTR_MAX_TOOL_CALLS", "8")),
        dataset_filter=dataset_filter,
        concurrency=int(os.getenv("BOTR_CONCURRENCY", "4")),
        resume=os.getenv("BOTR_RESUME", "true").lower() == "true",
        limit=int(os.getenv("BOTR_LIMIT", "0")),
        shuffle=os.getenv("BOTR_SHUFFLE", "true").lower() == "true",
        seed=int(os.getenv("BOTR_SEED", "42")),
        # CacheXL 参数
        tau_l=float(os.getenv("BOTR_TAU_L", "0.5")),
        tau_h=float(os.getenv("BOTR_TAU_H", "0.8")),
        cache_capacity=int(os.getenv("BOTR_CACHE_CAPACITY", "100")),
        retrieval_top_k=int(os.getenv("BOTR_RETRIEVAL_TOP_K", "2")),
        retrieval_threshold=float(os.getenv("BOTR_RETRIEVAL_THRESHOLD", "0.8")),
    )

    # 模型配置
    model = ModelConfig(
        backend=os.getenv("BOTR_MODEL_BACKEND", "meta/llama-3.3-70b-instruct"),
        temperature=float(os.getenv("BOTR_MODEL_TEMPERATURE", "0.1")),
        base_url=os.getenv(
            "BOTR_MODEL_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ),
        api_key=os.getenv("LLM_API_KEY", ""),
        top_p=float(os.getenv("BOTR_MODEL_TOP_P", "1.0"))
        if os.getenv("BOTR_MODEL_TOP_P")
        else None,
        max_tokens=int(os.getenv("BOTR_MODEL_MAX_TOKENS", "1024"))
        if os.getenv("BOTR_MODEL_MAX_TOKENS")
        else None,
        seed=int(os.getenv("BOTR_MODEL_SEED", "42"))
        if os.getenv("BOTR_MODEL_SEED")
        else None,
        verbose=os.getenv("LLM_VERBOSE", "1") == "1",
        http_timeout=float(os.getenv("LLM_HTTP_TIMEOUT", "120")),
        retry_attempts=int(os.getenv("LLM_RETRY_ATTEMPTS", "2")),
        retry_backoff=float(os.getenv("LLM_RETRY_BACKOFF", "1.0")),
    )

    # Embedding 专用配置
    embedding = EmbeddingConfig(
        base_url=os.getenv("EMBEDDING_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key=os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", "")),
        model=os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5"),
        http_timeout=float(os.getenv("EMBEDDING_HTTP_TIMEOUT", "30.0")),
    )

    # 数据集配置 (论文中的9个数据集)
    datasets = [
        # General Reasoning
        DatasetConfig(name="ai2_arc", path="ai2_arc"),
        DatasetConfig(name="csqa", path="csqa"),
        DatasetConfig(name="mmlu", path="mmlu"),
        DatasetConfig(name="strategyqa", path="strategyqa"),
        # Scientific and Domain
        DatasetConfig(name="aicrypto", path="aicrypto"),
        DatasetConfig(name="gpqa", path="gpqa"),
        DatasetConfig(name="pubmedqa", path="pubmedqa"),
        # Mathematical
        DatasetConfig(name="gsm8k", path="gsm8k"),
        DatasetConfig(name="math500", path="math500"),
    ]

    # 计费配置
    pricing = PricingConfig(
        prompt_per_1k=float(os.getenv("BOTR_PRICING_PROMPT_PER_1K", "0.0")),
        completion_per_1k=float(os.getenv("BOTR_PRICING_COMPLETION_PER_1K", "0.0")),
    )

    metrics = ["accuracy", "ece", "latency", "cost"]

    return Config(
        experiment=experiment,
        model=model,
        embedding=embedding,
        datasets=datasets,
        metrics=metrics,
        pricing=pricing,
    )
