import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from config import MODEL_NAME, MAX_SEQ_LEN

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.model_max_length = MAX_SEQ_LEN
    return tokenizer

def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model

def apply_lora(model):
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=['up_proj','down_proj','gate_proj','k_proj','q_proj','v_proj','o_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, lora_config)
