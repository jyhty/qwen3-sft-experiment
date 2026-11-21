# 复现说明

以下命令从仓库根目录运行。原实验使用 Linux、Python 3.12.4、PyTorch 2.8.0 CUDA 12.6 构建和单张 NVIDIA A800 80 GB。评测可以串行执行，降低同卡显存争用。

## 1. 创建独立环境

```bash
conda create -n qwen3-sft python=3.12.4 pip
conda activate qwen3-sft
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
python -m pip install -r requirements.txt
```

## 2. 下载输入并准备训练数据

```bash
python scripts/prepare_inputs.py
python scripts/prepare_classroom.py
python scripts/prepare_advanced.py
python scripts/prepare_advanced_validation.py
python scripts/run_sft_pipeline.py check
```

第一条命令需要联网。它按 [输入清单](configs/inputs.json)中的提交版本下载 Qwen3-4B-Base、Tülu 3、MMLU、IFEval 和 GSM8K，并准备 IFEval 用到的 NLTK 分词资源。基座放在 `models/Qwen3-4B-Base/`，原始 SFT 数据放在 `data/raw/`，评测数据放在 `data/benchmarks/`。文件已在本地时，可以运行 `python scripts/prepare_inputs.py --verify-only` 检查文件大小、清单和分词资源。

三条数据准备命令按顺序生成训练与验证 JSONL，并更新抽样清单。仓库已有的 [抽样清单](data/sft/)记录了固定种子、数据版本和文件校验值，可以和重新生成的清单对照。

## 3. 训练和评测

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_sft_pipeline.py train
CUDA_VISIBLE_DEVICES=0 python scripts/run_sft_pipeline.py evaluate --serial
```

训练依次运行 3k、10k、30k。只跑课堂组时，训练命令可加 `--group 3k`。只评测课堂组时，用 `evaluate 3k --serial`。完整评测覆盖 Base 和三组最终 LoRA，沿用 MMLU 5-shot、IFEval 0-shot 和 GSM8K 5-shot 的任务设置。去掉 `--serial` 会在同一张卡上并行执行 MMLU 和 GSM8K，IFEval 随后单独运行。原实验的 3k MMLU 在首次并行时遇到显存不足，最终使用串行补跑结果。

训练记录和适配器保存在 `outputs/`，逐题结果和模型汇总保存在 `outputs/evaluations/`。以后如果要直接复用已训练模型，请另外归档每组最终的 `adapter_model.safetensors` 和 `adapter_config.json`。要重画图或检查逐题判分，可一并归档 `trainer_state.json`、评测汇总和 `samples_*.jsonl`。
