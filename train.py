import torch, gc, wandb
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
from trl import SFTTrainer, SFTConfig

from config import NEW_MODEL_REPO, WANDB_PROJECT, NUM_EPOCHS, BATCH_SIZE, GRAD_ACCUM, LEARNING_RATE
from model import load_tokenizer, load_base_model, apply_lora
from data import load_health_dataset, formatting_func

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
train_ds, eval_ds = load_health_dataset()

training_args = SFTConfig(
    output_dir="./gemma-2b-health-output",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    optim="adamw_torch",
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    num_train_epochs=NUM_EPOCHS,
    bf16=True,
    logging_steps=1,

    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,

    push_to_hub=True,
    hub_model_id=NEW_MODEL_REPO,
    hub_strategy="every_save",
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
