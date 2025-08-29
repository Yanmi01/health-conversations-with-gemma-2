import torch, gc, wandb, os
from huggingface_hub import login
from trl import SFTTrainer, SFTConfig

from config import NEW_MODEL_REPO, WANDB_PROJECT, NUM_EPOCHS, BATCH_SIZE, GRAD_ACCUM, LEARNING_RATE
from model import load_tokenizer, load_base_model, apply_lora
from data import load_health_dataset, load_instruction_dataset, combine_datasets, format_chat

# Hugging Face login
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

wandb_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_key)
wandb.init(project=WANDB_PROJECT)

# Load tokenizer and model
tokenizer = load_tokenizer()
model = load_base_model()
model = apply_lora(model)
model.print_trainable_parameters()

# Dataset
health_ds = load_health_dataset()
inst_ds = load_instruction_dataset()
combined_ds = combine_datasets(health_ds, inst_ds)

train_ds = combined_ds["train"]
eval_ds  = combined_ds["eval"]


training_args = SFTConfig(
    output_dir="./gemma-2b-health-output",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,

    learning_rate=1e-6,                  
    lr_scheduler_type="cosine_with_restarts",            
    lr_scheduler_kwargs={"num_cycles": 2},
    warmup_ratio=0.01,                     
    num_train_epochs = 3,

    weight_decay=0.05,                     
    max_grad_norm=1.0,                    
    optim="adamw_torch_fused",            
    adam_beta2=0.95,                       

    bf16=True,                             
    logging_strategy="steps",
    logging_steps=10,                     
    eval_strategy="steps",
    eval_steps=500,                        
    save_strategy="steps",
    save_steps=500,                        
    save_total_limit=2,

    dataset_kwargs={"packing": True},
    max_length=512,

    push_to_hub=True,
    hub_model_id=NEW_MODEL_REPO,
    hub_strategy="every_save",
    load_best_model_at_end=True,
)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    formatting_func=lambda x: formatting_func(x, tokenizer),
    args=training_args,
)

# Cleanup before training
gc.collect()
torch.cuda.empty_cache()

trainer.train()
