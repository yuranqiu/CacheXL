#!/usr/bin/env python3
"""简单测试脚本 - 使用 mock 模式验证实验流程"""

import sys
import os

# 设置为空以启用 mock 模式
os.environ['LLM_API_KEY'] = ''
os.environ['EMBEDDING_API_KEY'] = ''

sys.path.insert(0, 'src')

from core.config import load_config
from core.llm import LLMClient
from methods.cachexl.workflow import run_cachexl_batch
from methods.cachexl.cache import EvidenceCache

cfg = load_config()
client = LLMClient('', '', config={})

test_batch = [
    {
        'id': '1',
        'question': 'What is 2+2?',
        'choices': [{'label': 'A', 'text': '3'}, {'label': 'B', 'text': '4'}],
        'answer': 'B',
        'label': 'B'
    },
    {
        'id': '2',
        'question': 'What is the capital of France?',
        'choices': [{'label': 'A', 'text': 'London'}, {'label': 'B', 'text': 'Paris'}],
        'answer': 'B',
        'label': 'B'
    }
]

cache = EvidenceCache(capacity=100)

print('=== CacheXL Mock 测试 ===')
results = run_cachexl_batch(
    client=client,
    model_name='mock',
    model_params={},
    batch=test_batch,
    cache=cache,
    dataset_name='test',
    concurrency=1,
    tau_l=cfg.experiment.tau_l,
    tau_h=cfg.experiment.tau_h
)

print(f'结果: {len(results)} 个样本')
for r in results:
    print(f"  ID: {r['id']}, Answer: {r.get('answer')}, Final: {r.get('final_answer')}")

print('\\n=== 测试通过 ===')
