from datasets import load_dataset

def load_health_dataset(test_size=0.01, seed=42):
    ds = load_dataset("Oluwadara/health_conversations", split="train")
    split = ds.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]

def formatting_func(example, tokenizer):
    row_json = [
        {"role": "user", "content": f"Description: {example['Description']}\nPatient: {example['Patient']}"},
        {"role": "model", "content": example["Doctor"]}
    ]
    return tokenizer.apply_chat_template(row_json, tokenize=False)
