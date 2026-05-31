数据集与获取方式

下载脚本
- python scripts/download_datasets.py --dataset gpqa
- python scripts/download_datasets.py --dataset winogrande
- python scripts/download_datasets.py --dataset pubmedqa
- python scripts/download_datasets.py --dataset medqa
- python scripts/download_datasets.py --dataset mmlu
- python scripts/download_datasets.py --dataset sms_spam

下载配置
- 支持通过环境变量指定 split 与 revision
  - <DATASET>_SPLIT: gpqa/winogrande/pubmedqa/medqa/mmlu
  - <DATASET>_REVISION: gpqa/winogrande/pubmedqa/medqa/mmlu
  - HF_DATASET_REVISION: 通用 revision 兜底
- 每个数据集会生成 Meta.json 记录来源与版本信息

GPQA
- Hugging Face: https://huggingface.co/datasets/Idavidrein/gpqa
- GitHub: https://github.com/idavidrein/gpqa
- 默认 split: train

WinoGrande debiased
- Hugging Face: https://huggingface.co/datasets/allenai/winogrande
- 默认 split: validation

PubMedQA
- Hugging Face: https://huggingface.co/datasets/qiaojin/PubMedQA
- 官网与脚本: https://github.com/pubmedqa/pubmedqa
- 默认 config: pqa_labeled
- 默认 split: train

MedQA (USMLE)
- GitHub: https://github.com/jind11/MedQA
- 默认 split: test

MMLU
- Hugging Face: https://huggingface.co/datasets/cais/mmlu
- 默认 config: all
- 默认 split: test

SMS Spam Detection
- UCI: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
- 默认 split: all

Seller Fraud Detection
- 论文说明将公开发布，目前需等待作者发布链接
