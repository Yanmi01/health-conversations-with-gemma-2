import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_REPO = "Yanmife/gemma-2b-health-fp-it"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9
)

prompt = "Hi Doctor, I’ve been having constant headaches and dizziness. What should I do?"
output = pipe(prompt)
print(output[0]["generated_text"])
