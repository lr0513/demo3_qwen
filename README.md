# Demo3：Qwen2.5-7B BC2GM基因实体识别

本项目使用BC2GM 数据完成生物医学基因实体识别。项目先用LLaMA Factory和Qwen2.5-0.5B-Instruct验证SFT流程，之后重构为Hugging Face Transformers + PEFT，并完成Qwen2.5-7B的LoRA和QLoRA实验。

核心流程：

```text
BC2GM 原始JSON
    -> Qwen Chat Template
    -> LoRA / QLoRA SFT
    -> 模型输出name/type JSON
    -> Python回查原文得到start/end
    -> 计算实体级P/R/F1
```

## 项目结构
```text
demo3_new/
├── main.py
├── test.py
├── predict.py
├── configs/
│   ├── lora_base.json
│   ├── qlora_base.json
│   ├── lora.json
│   └── qlora.json
├── data/raw/               # BC2GM原始数据
├── src/
│   ├── data.py             # 数据与labels构造
│   ├── model.py            # Qwen、LoRA、QLoRA加载
│   ├── trainer.py          # 训练、早停和保存
│   ├── evaluator.py        # 批量生成评估
│   ├── metrics.py          # TP/FP/FN和P/R/F1
│   └── postprocess.py      # JSON解析和实体定位
└── outputs/
```

## 环境与数据
AutoDL 上复用已有环境：
```bash
cd /root/autodl-tmp/demo3_new

source /root/miniconda3/etc/profile.d/conda.sh
conda activate qwen7b_qlora

python -c "import torch, transformers, peft, datasets, accelerate, swanlab; print('env ok')"
```

数据文件：
```text
data/raw/train.json
data/raw/dev.json
data/raw/test.json
```

模型文件：
```text
/root/autodl-tmp/demo3/models/Qwen2.5-7B
```

## 训练与评估
Base LoRA：

```bash
python main.py --config configs/lora_base.json
```

Base QLoRA：

```bash
python main.py --config configs/qlora_base.json
```

统一参数：

```text
max_length = 512
physical batch size = 2
gradient accumulation = 2
effective batch size = 4
epochs = 5
early stopping = dev F1
LoRA rank = 16
LoRA alpha = 32
learning rate = 2e-5
```

单独测试：

```bash
python test.py \
  --config configs/lora_base.json \
  --adapter outputs/qwen7b_base_bc2gm_lora_final/best \
  --split test
```

单句预测：

```bash
python predict.py \
  --config configs/lora_base.json \
  --adapter outputs/qwen7b_base_bc2gm_lora_final/best \
  --text "Comparison with alkaline phosphatases and 5 - nucleotidase"
```

## SwanLab
https://swanlab.cn/@lr0513/demo3-qwen7b-bc2gm
## 实验结果
早期 Instruct LoRA 基线：

| 方法 | Test P | Test R | Test F1 |
|---|---:|---:|---:|
| Qwen2.5-7B-Instruct LoRA | 0.8383 | 0.8292 | 0.8337 |
| Qwen2.5-7B-Instruct QLoRA | 0.8323 | 0.8189 | 0.8256 |

最终 base LoRA 结果：

| 方法 | Best Dev F1 | Test P | Test R | Test F1 |
|---|---:|---:|---:|---:|
| Qwen2.5-7B-base LoRA | 0.8363 | 0.8427 | 0.8276 | **0.8351** |

## 评估与误差

项目采用严格实体级评估：

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * TP / (2 * TP + FP + FN)
```

前一轮误差分析显示，边界不一致错误约占总错误的 `73%`，主要表现是模型多带或少带上下文词。例如：

```text
gold: COR
pred: COR biosynthetic gene cluster

gold: spectrin Providence
pred: spectrin
```
