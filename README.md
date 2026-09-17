# Demo3：基于LLaMA Factory的Qwen2.5生成式实体识别
本项目使用**LLaMA Factory 0.9.3** 对Qwen2.5进行LoRA/QLoRA微调，完成BC2GM生物医学基因实体识别任务。

项目目标是让大模型学会：
```text
输入：英文生物医学句子
输出：句子中的GENE实体
```

例如：
```json
[
  {
    "name": "alkaline phosphatases",
    "type": "GENE"
  }
]
```
模型只负责识别实体名称和实体类型。字符位置 `start/end` 由Python后处理脚本根据原句计算，避免让生成式模型直接预测不稳定的字符下标。
## 项目特点
- 使用 LLaMA Factory 0.9.3 完成 SFT 微调。
- 使用 Qwen2.5-0.5B-Instruct 验证完整训练流程。
- 使用 LoRA 进行参数高效微调。
- 使用 BC2GM 数据完成基因实体识别。
- 使用 `name/type-only` 输出格式，降低模型生成位置数字的难度。
- 使用后处理自动补全实体 `start/end`。
- 使用实体级 Precision、Recall、F1 进行评估。
- 支持 SwanLab 记录训练曲线。
- 支持后续迁移到 Qwen2.5-7B QLoRA。

## 项目结构

```text
.
├── configs/
│   ├── identity_lora_smoke.yaml
│   ├── identity_lora_full.yaml
│   ├── bc2gm_lora.yaml
│   ├── bc2gm_lora_name_only.yaml
│   └── bc2gm_lora_name_only_v3.yaml
├── data/
│   ├── raw/
│   │   ├── train.json
│   │   ├── dev.json
│   │   └── test.json
│   ├── identity_demo.json
│   ├── dataset_info.json
│   ├── bc2gm_train.json
│   ├── bc2gm_dev.json
│   ├── bc2gm_test.json
│   ├── bc2gm_train_name_only.json
│   ├── bc2gm_dev_name_only.json
│   └── bc2gm_test_name_only.json
└── scripts/
    ├── convert_bc2gm.py
    ├── postprocess.py
    ├── evaluate_bc2gm.py
    ├── inspect_errors.py
    └── test_postprocess.py
```

## 各目录说明

| 目录 | 作用 |
|---|---|
| `configs/` | LLaMA Factory 训练和推理 YAML |
| `data/raw/` | BC2GM 原始数据 |
| `data/*_name_only.json` | 只包含 name/type 的 SFT 数据 |
| `data/dataset_info.json` | LLaMA Factory 数据集注册表 |
| `scripts/convert_bc2gm.py` | 把原始 BC2GM 转成 SFT 格式 |
| `scripts/postprocess.py` | 根据实体名称计算准确字符位置 |
| `scripts/evaluate_bc2gm.py` | 模型推理和实体级 P/R/F1 评估 |
| `scripts/inspect_errors.py` | 查看 FP、FN 和边界错误 |

## 数据和标签

原始 BC2GM 数据格式：

```json
{
  "sentence": "Comparison with alkaline phosphatases and 5 - nucleotidase",
  "entities": [
    {
      "name": "alkaline phosphatases",
      "type": "GENE",
      "pos": [16, 37]
    }
  ]
}
```

转换为 LLaMA Factory 的 Alpaca SFT 格式：

```json
{
  "instruction": "Extract all GENE entities from the sentence. Return a JSON array. Each item must contain name and type. Return [] if none.",
  "input": "Comparison with alkaline phosphatases and 5 - nucleotidase",
  "output": "[{\"name\": \"alkaline phosphatases\", \"type\": \"GENE\"}]"
}
```

本项目使用实体级评估：
```text
只有实体类型和起止位置都完全匹配，才算 TP
```
指标计算方式：
```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

## 环境配置
```text
Python 3.10
LLaMA Factory 0.9.3
Transformers 4.52.4
PEFT 0.15.2
TRL 0.9.6
Datasets 3.6.0
Accelerate 1.7.0
SwanLab 0.10.0
```

安装依赖：

```bash
pip install \
  llamafactory==0.9.3 \
  transformers==4.52.4 \
  peft==0.15.2 \
  trl==0.9.6 \
  datasets==3.6.0 \
  accelerate==1.7.0 \
  swanlab==0.10.0

pip install --upgrade "huggingface-hub>=0.30.0,<1.0"
```
普通 LoRA 不需要 `bitsandbytes`。  
QLoRA 需要单独安装与 PyTorch 版本兼容的 `bitsandbytes`。

### 显存建议
| 模型/方法 | 建议显存 |
|---|---|
| Qwen2.5-0.5B + LoRA | 8GB 以上 |
| Qwen2.5-7B + QLoRA | 12GB 以上，推荐 24GB |
| Qwen2.5-7B + LoRA | 24GB 以上，推荐 32GB |

## 下载模型
下载 Qwen2.5-0.5B-Instruct：
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-0.5B-Instruct',
    local_dir='/root/autodl-tmp/demo3/llama/models/Qwen2.5-0.5B-Instruct',
    endpoint='https://hf-mirror.com',
    max_workers=1,
    etag_timeout=60,
)
"
```

下载 Qwen2.5-7B-Instruct：
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    local_dir='/root/autodl-tmp/demo3/llama/models/Qwen2.5-7B-Instruct',
    endpoint='https://hf-mirror.com',
    max_workers=1,
    etag_timeout=60,
)
"
```

## 数据转换
```bash
python scripts/convert_bc2gm.py
```

生成：
```text
data/bc2gm_train_name_only.json
data/bc2gm_dev_name_only.json
data/bc2gm_test_name_only.json
```

然后在 `data/dataset_info.json` 中注册为：
```text
bc2gm_train_name_only
bc2gm_dev_name_only
bc2gm_test_name_only
```

## 训练流程
### 1. Identity 小样本验证
先验证 LLaMA Factory、数据和 LoRA 流程是否正常：
```bash
llamafactory-cli train \
  /root/autodl-tmp/demo3/llama/configs/identity_lora_smoke.yaml
```

测试：
```bash
llamafactory-cli chat \
  /root/autodl-tmp/demo3/llama/configs/chat_identity_smoke.yaml
```
如果模型输出固定身份 `DemoBot`，说明最小 SFT 流程已经跑通。
### 2. BC2GM v3 训练
```bash
llamafactory-cli train \
  /root/autodl-tmp/demo3/llama/configs/bc2gm_lora_name_only_v3.yaml
```
关键配置：
```yaml
model_name_or_path: /root/autodl-tmp/demo3/llama/models/Qwen2.5-0.5B-Instruct
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
dataset: bc2gm_train_name_only
max_samples: 12500
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

训练产物默认保存到：
```text
outputs/bc2gm_lora_name_only_v3/
├── adapter_model.safetensors
├── adapter_config.json
├── trainer_state.json
└── training_loss.png
```

`models/`、`outputs/`、权重和缓存文件不会上传到仓库。

## 训练参数说明
| 参数 | 含义 |
|---|---|
| `model_name_or_path` | 基座模型路径 |
| `dataset` | LLaMA Factory 注册的数据集名 |
| `dataset_dir` | `dataset_info.json` 所在目录 |
| `finetuning_type` | 微调方式，LoRA 或 QLoRA |
| `lora_rank` | LoRA 低秩矩阵维度 |
| `lora_alpha` | LoRA 缩放系数 |
| `lora_dropout` | LoRA dropout |
| `lora_target` | 应用 LoRA 的模块 |
| `quantization_bit` | QLoRA 时设为 4 |
| `per_device_train_batch_size` | 单卡 batch size |
| `gradient_accumulation_steps` | 梯度累积步数 |
| `learning_rate` | 学习率 |
| `num_train_epochs` | 训练轮数 |
| `cutoff_len` | 单条样本最大长度 |

`batch_size=1` 且 `gradient_accumulation_steps=8` 表示：

```text
每次计算 1 条样本
连续累积 8 次梯度
最后统一更新一次模型参数
```

这样可以降低显存占用，近似等效于 batch size 为 8。

## 模型推理与后处理

模型输出：

```json
[
  {
    "name": "alkaline phosphatases",
    "type": "GENE"
  }
]
```

`postprocess.py` 会回查原句，计算：

```json
{
  "name": "alkaline phosphatases",
  "type": "GENE",
  "start": 16,
  "end": 37
}
```

这样做的作用：

1. 模型只负责语义识别。
2. Python 负责精确字符定位。
3. 减少模型生成错误数字位置造成的误差。

## 测试集评估

评估 200 条：

```bash
python scripts/evaluate_bc2gm.py --limit 200
```

评估 500 条：

```bash
python scripts/evaluate_bc2gm.py --limit 500
```

输出：

```text
precision
recall
f1
tp
fp
fn
```

## 当前实验结果

### 0.5B 版本对比

| 实验 | 输出格式 | 样本数 | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| v1 | name + type + start + end | 100 | 0.7207 | 0.6349 | 0.6751 |
| v2 | name + type | 200 | 0.7406 | 0.7166 | 0.7284 |
| v3 | name + type，全量训练 | 200 | 0.7724 | 0.7692 | **0.7708** |
| v3 | name + type，500 条测试 | 500 | 0.7772 | 0.7460 | **0.7613** |

当前最佳 0.5B 模型：

```text
configs/bc2gm_lora_name_only_v3.yaml
outputs/bc2gm_lora_name_only_v3/adapter_model.safetensors
```

## 错误分析

当前错误主要集中在以下情况：

1. 长实体只识别出一部分。
2. 实体边界多带或少带修饰词。
3. 长句或多实体句子出现漏识别。
4. 一些生物医学复合词被错误识别为 GENE。
5. BC2GM 标注规则和模型理解之间存在差异。

典型错误：

```text
真实：COR
预测：COR biosynthetic gene cluster

真实：prolactin
预测：Plasma prolactin

真实：P - ITIM - compelled multi - phosphoprotein complex
预测：P - ITIM
```

