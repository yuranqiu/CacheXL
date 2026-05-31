# CacheXL: Cross-Instance Learning via Online Cache for Efficient and Enhanced LLM Inference

<p align="center">
  <img src="figures/CacheXL.pdf" alt="CacheXL Framework Overview" width="100%">
</p>

## Abstract

Cross-instance learning is an emerging mechanism for improving LLM reasoning. It applies batch reflection to share information across queries and improve reasoning performance. However, instances aggregated within the same batch often lack semantic relevance to the current query, limiting the specificity of reflective feedback. We propose **CacheXL**, a cache-augmented cross-instance learning framework that maintains an online evidence cache of high-confidence historical reasoning instances and retrieves semantically similar examples as targeted reflective evidence. In CacheXL, the online evidence cache serves as the central mechanism for cross-instance learning. The full system integrates retrieval-based evidence reuse, selective reflection, and escalation, and runs retrieval asynchronously with initial reasoning to reduce additional latency. Extensive experiments on nine reasoning benchmarks and five LLMs show that CacheXL improves average accuracy and calibration across models. It also achieves lower end-to-end latency than batch reflection.

## Method

CacheXL is a cross-instance learning framework built around an online evidence cache. The framework uses three LLM roles: **Actor**, **Reflector**, and **Escalator**.

1. **Actor** produces an initial rationale, an answer, and a self-reported confidence score for each query.
2. In parallel, the system retrieves semantically related historical instances from the **evidence cache**.
3. **Reflector** reviews each sample with the retrieved context, decides whether the current answer can be accepted, and identifies reusable high-confidence cases for cache admission.
4. **Escalator** is used only for difficult cases flagged by Reflector (when both Actor and Reflector confidence are below threshold τ_l).

### Key Design

- **Evidence Cache**: Maintains high-confidence historical instances with queries, rationales, answers, and feedback
- **Asynchronous Retrieval**: Runs cache retrieval in parallel with initial Actor reasoning
- **Selective Escalation**: Triggers Escalator only when both confidence scores < τ_l
- **Cache Admission**: Strict policy requiring acceptance, reusability, and both confidences ≥ τ_h

## Results

CacheXL achieves the highest average accuracy across five LLMs and nine benchmarks:

| Model | ReAct | BoT | CacheXL |
|-------|-------|-----|---------|
| Qwen3-80B | 64.80 | 76.14 | **80.30** |
| Qwen2.5-7B | 56.27 | 59.61 | **65.14** |
| Llama-3.3-70B | 67.38 | 69.95 | **76.86** |
| DeepSeek-V3 | 69.51 | 71.46 | **75.26** |
| Qwen2.5-32B | 62.08 | 62.68 | **68.02** |

CacheXL also achieves lower ECE (better calibration) and lower latency than BoT across all models.

## Quick Start

### Prerequisites

- Python >= 3.11
- API access to LLM services (e.g., NVIDIA API)

### Installation

```bash
# Clone the repository
git clone https://github.com/yuranqiu/CacheXL.git
cd CacheXL

# Initialize environment and install dependencies
bash scripts/setup_venv.sh
```

### Configuration

```bash
# Copy example configuration
cp .env.example .env

# Edit .env file with your API key and settings
```

### Data Preparation

```bash
# Download datasets
python scripts/download_datasets.py --dataset gpqa,mmlu,math500
```

### Run Experiments

```bash
# Run CacheXL method
python src/run_cachexl.py --dataset gpqa,mmlu

# Run ReAct baseline
python src/run_react.py --dataset gpqa,mmlu

# Run BoT method
python src/run_bot.py --dataset gpqa,mmlu
```

## Project Structure

```
CacheXL/
├── src/
│   ├── run_react.py          # ReAct baseline entry
│   ├── run_bot.py            # BoT method entry
│   ├── run_cachexl.py        # CacheXL method entry
│   ├── core/
│   │   ├── config.py         # Configuration
│   │   ├── llm.py            # LLM client
│   │   └── utils.py          # Utilities
│   └── methods/
│       ├── react/            # ReAct baseline
│       ├── bot/              # BoT method
│       └── cachexl/          # CacheXL method
│           ├── workflow.py   # Main workflow
│           ├── cache.py      # Evidence cache
│           └── prompts.py    # Prompt templates
├── data/                     # Datasets
├── scripts/                  # Utility scripts
├── figures/                  # Figures
├── .env.example              # Configuration template
└── pyproject.toml            # Project dependencies
```

## Citation

```bibtex
@article{yue2026cachexl,
  title={CacheXL: Cross-Instance Learning via Online Cache for Efficient and Enhanced LLM Inference},
  author={Yue, Ruiqing and Cui, Yu and Xue, Xianhong and Pan, Sicheng and Sun, Zhuoyu and Liu, Yifei and Huang, Baohan and Cui, Zhe and Zhang, Haibin and Zuo, Cong},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
