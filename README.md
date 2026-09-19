# Demo3：Qwen2.5-7B BC2GM 基因实体识别

本项目使用BC2GM数据集完成生物医学基因实体识别。项目先用LLaMA Factory和Qwen2.5-0.5B-Instruct验证SFT流程，之后重构为Hugging Face Transformers + PEFT，并在Qwen2.5-7B-Instruct上完成LoRA和QLoRA对比实验。

核心流程：
```text
BC2GM 原始 JSON
    -> Qwen Chat Template
    -> LoRA / QLoRA SFT
    -> 模型输出 name/type JSON
    -> Python回查原文得到start/end
    -> 计算实体级P/R/F1
```
## 项目结构

```text
demo3_new/
├── main.py                 # 训练入口
├── test.py                 # 测试集评估
├── predict.py              # 单句预测
├── configs/
│   ├── lora.json
│   └── qlora.json
├── data/raw/               # BC2GM原始数据
├── src/
│   ├── data.py             # 数据与labels构造
│   ├── model.py            # Qwen、LoRA、QLoRA 加载
│   ├── trainer.py          # 训练、早停和保存
│   ├── evaluator.py        # 批量生成评估
│   ├── metrics.py          # TP/FP/FN和P/R/F1
│   └── postprocess.py      # JSON解析和实体定位
└── outputs/
```

## 环境与准备

AutoDL 上复用已有环境：
```bash
cd /root/autodl-tmp/demo3_new

source /root/miniconda3/etc/profile.d/conda.sh
conda activate qwen7b_qlora

python -c "import torch, transformers, peft, datasets, accelerate, swanlab; print('env ok')"
```
确认模型和数据：
```bash
ls -lh /root/autodl-tmp/demo3/models/Qwen2.5-7B-Instruct
ls data/raw
```
需要以下数据：
```text
data/raw/train.json
data/raw/dev.json
data/raw/test.json
```
## 训练与评估
LoRA 训练：
```bash
python main.py --config configs/lora.json
```
QLoRA 训练：
```bash
python main.py --config configs/qlora.json
```
训练配置：
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
单独评估：
```bash
python test.py \
  --config configs/lora.json \
  --adapter outputs/qwen7b_bc2gm_lora_final/best \
  --split test
```

单句预测：
```bash
python predict.py \
  --config configs/lora.json \
  --adapter outputs/qwen7b_bc2gm_lora_final/best \
  --text "Comparison with alkaline phosphatases and 5 - nucleotidase"
```
预测结果：
```python
model_output: {"entities":[{"name":"alkaline phosphatases","type":"GENE"}]}
entities: [{'name': 'alkaline phosphatases', 'type': 'GENE', 'start': 16, 'end': 37}]
```
## SwanLab
https://swanlab.cn/@lr0513/demo3-qwen7b-bc2gm/

## 实验结果

| 方法 | Test P | Test R | Test F1 | Peak GPU |
|---|---:|---:|---:|---:|
| LoRA | 0.8383 | 0.8292 | **0.8337** | 17.08GB |
| QLoRA | 0.8323 | 0.8189 | **0.8256** | 11.30GB |

结论：
```text
LoRA的F1略高，适合追求最佳识别效果。
QLoRA显存降低约33.8%，适合显存受限环境。
项目先用LLaMA Factory和0.5B验证流程，再用当前代码完成7B正式实验。
```

## 评估与误差
项目采用严格实体级评估：
```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * TP / (2 * TP + FP + FN)
```

LoRA 测试集错误统计：
```text
boundary_fp = 751
boundary_fn = 784
invalid_unlocated = 12
extra_fp = 260
missing_fn = 296
```
边界不一致约占错误总量的`73%`，主要表现是模型多带或少带上下文词。例如：
```text
gold: COR
pred: COR biosynthetic gene cluster

gold: spectrin Providence
pred: spectrin
```
无法精确匹配的格式问题只有12个，因此主要损失来自实体边界判断，不建议通过模糊匹配修改正式指标。

