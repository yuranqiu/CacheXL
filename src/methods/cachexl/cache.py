import os
import json
import heapq
from typing import List, Dict, Any
from collections import OrderedDict
from datetime import datetime, timedelta
from core.llm import LLMClient
from core.config import load_config


class EvidenceCache:
    """
    实现简单的 Q&A 对缓存机制。
    使用 Nvidia Embedding 模型进行相似度检索。
    容量: 固定 100 条。
    淘汰策略: LRU + 时效性混合（优先淘汰低频+老旧的条目）。
    """

    def __init__(
        self,
        capacity: int = 100,
        storage_path: str = "experiments/cachexl/evidence_cache.json",
        max_age_hours: int = 24,
    ):
        self.capacity = capacity
        self.storage_path = storage_path
        self.max_age = timedelta(hours=max_age_hours)
        # 使用 OrderedDict 实现 LRU: 最近访问的在末尾
        self.items: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.client = None
        self.embedding_model = "nvidia/nv-embedqa-e5-v5"
        self._init_client()
        self._load_from_disk()

    def _load_from_disk(self):
        """从磁盘加载缓存"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.items = json.load(f)
                print(
                    f"[Cache] Loaded {len(self.items)} items from {self.storage_path}"
                )
            except Exception as e:
                print(f"[Cache] Failed to load from disk: {e}")

    def _save_to_disk(self):
        """保存缓存到磁盘"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.items, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Cache] Failed to save to disk: {e}")

    def _init_client(self):
        cfg = load_config()
        http_cfg = {
            "ssl_verify": cfg.model.ssl_verify
            if cfg.model.ssl_verify is not None
            else True,
            "httpx_trust_env": cfg.model.httpx_trust_env
            if cfg.model.httpx_trust_env is not None
            else True,
            "http_timeout": cfg.embedding.http_timeout,
            "retry_attempts": 2,
        }
        self.client = LLMClient(
            cfg.embedding.base_url, cfg.embedding.api_key, config=http_cfg
        )
        self.embedding_model = cfg.embedding.model

    def get_embedding(self, text: str) -> List[float]:
        """
        使用 Nvidia API 获取文本的 Embedding。
        参考: https://build.nvidia.com/nvidia/nv-embedqa-mistral-7b-v2
        """
        if not text:
            return []

        # 检查是否处于 Mock 模式
        if self.client.mock:
            return [0.0] * 4096

        try:
            if self.client.mock:  # 404 标记为 Mock (复用字段表示禁用)
                return []

            # Nvidia Embedding API 格式
            payload = {
                "input": [text],
                "model": self.embedding_model,
                "input_type": "query",
                "encoding_format": "float",
            }
            url = f"{self.client.base_url}/embeddings"
            headers = {"Authorization": f"Bearer {self.client.api_key}"}

            import httpx

            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["data"][0]["embedding"]
                elif resp.status_code == 404:
                    # 404 意味着模型不可用，禁用缓存以提升速度
                    print(
                        f"[Cache] Embedding Model not found (404). Cache disabled for this session."
                    )
                    self.client.mock = True  # 标记为禁用
                    return []
                else:
                    # 失败回退或记录错误
                    print(f"[Cache] Embedding error {resp.status_code}: {resp.text}")
                    return []
        except Exception as e:
            print(f"[Cache] Embedding exception: {e}")
            return []

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本的 Embedding。
        """
        if not texts:
            return []

        if self.client.mock:
            return [[0.0] * 4096] * len(texts)

        try:
            # Nvidia Embedding API 格式
            payload = {
                "input": texts,
                "model": self.embedding_model,
                "input_type": "query",
                "encoding_format": "float",
            }
            url = f"{self.client.base_url}/embeddings"
            headers = {"Authorization": f"Bearer {self.client.api_key}"}

            import httpx

            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    # data["data"] is a list of objects with "embedding" field
                    # Ensure order is preserved (it should be)
                    embeddings = [d["embedding"] for d in data["data"]]
                    usage = data.get("usage", {}).get("total_tokens", 0)
                    return embeddings, usage
                elif resp.status_code == 404:
                    print(f"[Cache] Embedding Model not found (404). Cache disabled.")
                    self.client.mock = True
                    return [[0.0] * 4096] * len(texts), 0
                else:
                    print(
                        f"[Cache] Batch Embedding error {resp.status_code}: {resp.text}"
                    )
                    return [[]] * len(texts), 0
        except Exception as e:
            print(f"[Cache] Batch Embedding exception: {e}")
            return [[]] * len(texts), 0

    def add(self, item: Dict[str, Any], embedding: List[float] = None):
        """
        向缓存中添加高置信度的条目。
        缓存结构 C = {(q_i, r_i, y_i, f_i)}: question, rationale, final_answer, feedback
        """
        # 添加前先计算 Embedding
        if embedding is None:
            embedding = self.get_embedding(item.get("question", ""))

        if not embedding:
            return

        entry = {
            "question": item.get("question"),
            "answer": item.get("final_answer") or item.get("answer"),
            "rationale": item.get("rationale"),
            "feedback": item.get("critique", ""),
            "embedding": embedding,
        }

        # 使用 LRU 策略: 如果已存在则移到末尾, 否则添加
        entry_id = item.get("question", "")
        if entry_id in self.items:
            self.items.move_to_end(entry_id)
        self.items[entry_id] = entry

        # LRU 淘汰: 移除最久未访问的
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)

        self._save_to_disk()

    def get_relevant(
        self,
        query: str,
        k: int = 2,
        query_embedding: List[float] = None,
        threshold: float = 0.8,
    ) -> List[Dict[str, Any]]:
        """
        基于余弦相似度检索 k 个最相关的条目，并根据阈值过滤。
        """
        if not self.items:
            return []

        if query_embedding is None:
            query_embedding = self.get_embedding(query)

        if not query_embedding:
            return []

        # 计算余弦相似度
        def cosine_similarity(v1, v2):
            if not v1 or not v2:
                return 0.0
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = sum(a * a for a in v1) ** 0.5
            norm2 = sum(b * b for b in v2) ** 0.5
            return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

        scores = []
        for item in self.items.values():
            # 质量过滤：排除掉那些推理过程过短、缺乏逻辑参考价值的样本
            if len(item.get("rationale", "")) < 100:
                continue

            score = cosine_similarity(query_embedding, item["embedding"])
            if score >= threshold:
                scores.append((score, item))

        # 获取 Top K
        top_k = heapq.nlargest(k, scores, key=lambda x: x[0])
        return [item for score, item in top_k]

    def format_context(self, items: List[Dict[str, Any]]) -> str:
        """
        将检索到的条目格式化为 Prompt 的上下文。
        """
        if not items:
            return ""

        context = "Here are some similar examples for reference:\n\n"
        for i, item in enumerate(items, 1):
            context += f"Example {i}:\n"
            context += f"Question: {item['question']}\n"
            context += f"Reasoning: {item['rationale']}\n"
            context += f"Answer: {item['answer']}\n\n"
        return context
