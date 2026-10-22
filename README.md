# Qwen2.5-1.5B-Instruct LoRA 微调演示 - 卷发护发问答模型

一个 LLM LoRA 微调完整流水线 Demo，针对卷发护发领域的中文问答。

## 1. 项目简介

本项目演示了从「数据准备 → LoRA 微调 → 交互式推理」的完整流程，目标是让基座模型学会回答卷发护发相关的领域问题。

- **基座模型**：[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **微调方式**：使用 PEFT LoRA 低秩适配，只训练约 0.6% 的参数，不做全参数微调
- **训练数据**：14 条卷发护发指令问答对（`dataset.jsonl`）

> 💡 **前期踩坑记录**：最初选用 GPT-2 作为基座，但 GPT-2 原生英文词表对中文支持极差，生成结果出现大量乱码与重复。后更换为 Qwen2.5 中文基座（自带中文词表 + 对话微调），中文生成质量大幅提升。

## 2. 环境依赖

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)（推荐，用于隔离环境）

核心 Python 包：

| 包名 | 用途 |
|------|------|
| torch | 深度学习框架（GPU 加速） |
| transformers | 加载模型、训练 Trainer、推理 |
| peft | LoRA 低秩适配微调 |
| datasets | 加载 `dataset.jsonl` 数据集 |
| accelerate | 分布式/设备加速（transformers 依赖） |
| bitsandbytes | 8-bit AdamW 优化器（`paged_adamw_8bit`）|

一键安装：

```powershell
conda create -n ds-template python=3.10 -y
conda activate ds-template
pip install torch transformers peft datasets accelerate bitsandbytes
```

> 脚本内已通过 `HF_ENDPOINT = "https://hf-mirror.com"` 配置了 HuggingFace 镜像，国内可正常下载模型。

## 3. 项目文件结构

```
deepseek-lora-windows/
├── train.py                  # LoRA 训练脚本
├── infer.py                  # 交互式推理脚本
├── dataset.jsonl             # 指令微调数据集（14 条，instruction/response 格式）
└── deepseek-lora-output/     # LoRA 权重输出目录（训练后自动生成）
```

数据集格式（`dataset.jsonl` 每行一条 JSON）：

```json
{"instruction": "烫完卷发多久能洗头", "response": "一般建议烫完48小时之后再洗头，让卷度充分定型……"}
```

## 4. 运行步骤

### 第一步：训练

```powershell
conda activate ds-template
python train.py
```

首次运行会自动从镜像下载模型（约 3GB），训练完成后权重保存在 `deepseek-lora-output/`。

### 第二步：推理

```powershell
conda activate ds-template
python infer.py
```

启动后直接输入问题即可，输入 `quit` / `exit` / `q` 退出。`infer.py` 会自动加载最新的 checkpoint。

## 5. 训练超参

| 参数 | 值 | 说明 |
|------|-----|------|
| `SEQ_LEN` | 512 | 序列最大长度（配合 12GB 显存） |
| `LORA_R` | 8 | LoRA 秩 |
| `LORA_ALPHA` | 16 | LoRA 缩放系数 |
| `LORA_DROPOUT` | 0.05 | LoRA dropout |
| `num_train_epochs` | 10 | 训练轮数 |
| `learning_rate` | 2e-4 | 学习率 |
| `weight_decay` | 0.01 | 权重衰减 |
| `per_device_train_batch_size` | 1 | 单卡 batch size |
| `gradient_accumulation_steps` | 4 | 梯度累积步数 |
| `bf16` | True | bf16 混合精度训练 |
| `optim` | paged_adamw_8bit | 8-bit 优化器（省显存） |

LoRA 目标层：`q_proj / k_proj / v_proj / o_proj / up_proj / gate_proj / down_proj`（Qwen2.5 注意力 + FFN 全量）

## 6. 项目局限

- **数据量极小**：数据集仅 14 条，属于小样本微调，模型对训练数据外的泛化能力有限
- **易混淆相似问题**：相似问法之间容易出现答案混淆、少量信息串答
- **过拟合/欠拟合两难**：epoch 太少记不住具体答案，太多又容易背死导致串答

## 7. 后续优化方向

- **扩充指令数据集**：将问答对从 14 条扩充到几十上百条，覆盖更多问法，提升泛化与稳定性
- **尝试 QLoRA 量化微调**：使用 4-bit 量化加载基座，在同样显存下跑更大的基座模型（如 Qwen 3B / DeepSeek）
- **加入验证集**：拆分训练/验证集，用验证 loss 更科学地选择 epoch，避免盲调
