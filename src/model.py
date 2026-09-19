import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _dtype(config):
    if torch.cuda.is_available() and config.get("bf16", True):
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_base_model(model_path, method, config):
    if method == "qlora":
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requires a CUDA device.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=config.get(
                "bnb_4bit_use_double_quant",
                True,
            ),
            bnb_4bit_compute_dtype=_dtype(config),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto", # 自动把模型分配到CPU/GPU
            trust_remote_code=True,
        )
        return prepare_model_for_kbit_training(model)

    kwargs = {
        "torch_dtype": _dtype(config),
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def build_training_model(model_path, config):
    method = config.require("method")
    model = load_base_model(model_path, method, config)

    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.print_trainable_parameters() # 打印可训练参数占比
    return model

'''推理预测阶段使用'''
def load_inference_model(base_model_path, adapter_path, method, config):
    model = load_base_model(base_model_path, method, config)
    if adapter_path: # lora权重文件夹
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    try:
        model.base_model.model.config.use_cache = True
    except AttributeError:
        pass
    return model

