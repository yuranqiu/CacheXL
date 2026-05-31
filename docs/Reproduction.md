# CacheXL 复现指南

## 论文信息

- **标题**: CacheXL: Cross-Instance Learning via Online Cache for Efficient and Enhanced LLM Inference
- **方法**: CacheXL - 基于在线证据缓存的跨实例学习框架

## 实验设置

### 模型

论文中评估了5个指令微调的LLM：

| 模型 | 简称 | 参数量 |
|------|------|--------|
| Qwen3-Next-80B-A3B-Instruct | Qwen3-80B | 80B |
| Qwen2.5-7B-Instruct | Qwen2.5-7B | 7B |
| Qwen2.5-32B-Instruct | Qwen2.5-32B | 32B |
| Llama-3.3-70B-Instruct | Llama-3.3-70B | 70B |
| DeepSeek-V3 | DeepSeek-V3 | - |

### 数据集

论文中评估了9个推理基准测试：

**通用推理**:
- AI2 ARC: 小学科学问答
- CSQA: 常识问答
- MMLU: 多任务语言理解
- StrategyQA: 隐式多步推理

**科学和领域推理**:
- AICrypto: 密码学相关推理
- GPQA: 研究生级别科学问答
- PubMedQA: 生物医学研究问答

**数学推理**:
- GSM8K: 小学数学应用题
- Math500: 竞赛数学问题

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 8 | 批次大小 |
| τ_l | 0.5 | 升级阈值 |
| τ_h | 0.8 | 缓存准入阈值 |
| cache_capacity | 100 | 缓存容量 |
| retrieval_top_k | 2 | 检索数量 |
| retrieval_threshold | 0.8 | 检索相似度阈值 |
| temperature | 0.1 | 生成温度 |
| seed | 42 | 随机种子 |

### 指标

- **Accuracy**: 准确率
- **ECE**: Expected Calibration Error (10-bin)
- **Latency**: 端到端延迟 (秒)
- **Cost**: 每样本成本 (USD)

## 复现步骤

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/yuranqiu/CacheXL.git
cd CacheXL

# 初始化环境
bash scripts/setup_venv.sh
```

### 2. 配置

```bash
# 复制配置文件
cp .env.example .env

# 编辑 .env 文件，设置 API 密钥
# LLM_API_KEY=your_api_key
```

### 3. 数据准备

```bash
# 下载数据集
python scripts/download_datasets.py --dataset ai2_arc,csqa,mmlu,strategyqa,aicrypto,gpqa,pubmedqa,gsm8k,math500
```

### 4. 运行实验

```bash
# 运行 CacheXL
python src/run_cachexl.py

# 运行 ReAct baseline
python src/run_react.py

# 运行 BoT
python src/run_bot.py

# 运行特定数据集
python src/run_cachexl.py --dataset gpqa,math500
```

### 5. 生成报告

```bash
# 生成对比报告
python scripts/generate_report.py
```

## 方法对比

| 方法 | 描述 | 角色 |
|------|------|------|
| ReAct | 单轮推理 baseline | Actor |
| BoT | 批量反思 | Actor + Reflector (batch) |
| CacheXL | 缓存增强跨实例学习 | Actor + Reflector + Escalator |

## CacheXL 工作流程

1. **Actor** 生成初始答案和置信度
2. **异步检索** 从证据缓存中检索相似案例
3. **Reflector** 使用检索上下文评估答案
4. 如果 c_q < τ_l AND ρ_q < τ_l → **Escalator** 重新解答
5. 缓存准入: accept=true AND reusable=true AND c_q ≥ τ_h AND ρ_q ≥ τ_h

## 参考文献

- Batch-of-Thought (BoT): Yang et al., 2026
- ReAct: Yao et al., 2023
- Chain-of-Thought: Wei et al., 2022
