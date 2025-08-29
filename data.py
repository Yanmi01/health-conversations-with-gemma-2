from datasets import load_dataset, concatenate_datasets
import re


seed = 42

def process_patient_data(dataset):
    processed_data = {"input": [], "output": []}
    for desc, patient, doctor in zip(dataset["Description"], dataset["Patient"], dataset["Doctor"]):
        desc = re.sub(r'\s+', ' ', desc).strip()
        patient = re.sub(r'\s+', ' ', patient).strip()
        doctor = re.sub(r'\s+', ' ', doctor).strip()
        query = f"Query: {desc} {patient}" if desc else f"Query: {patient}"
        processed_data["input"].append(query)
        processed_data["output"].append(doctor)
    return processed_data


def process_AI_data(dataset):
    processed_data = {"input": [], "output": []}
    for conv in dataset["Conversation"]:
        parts = re.split(r'\[\|Human\|\]|\[\|AI\|\]', conv)
        if len(parts) >= 3:
            human_text, ai_text = parts[1].strip(), parts[2].strip()
        else:
            human_text, ai_text = conv, ""
        human_text = re.sub(r'\s+', ' ', human_text).strip()
        ai_text = re.sub(r'\s+', ' ', ai_text).strip()
        processed_data["input"].append(f"Query: {human_text}")
        processed_data["output"].append(ai_text)
    return processed_batch


def load_health_dataset(split_ratio=0.05, seed=seed):
    ds = load_dataset("Oluwadara/health_conversations", split="train")
    ds = ds.map(process_patient_data, batched=True, num_proc=8, remove_columns=ds.column_names)
    return ds.train_test_split(test_size=split_ratio, seed=seed)


def load_instruction_dataset(split_ratio=0.05, seed=seed):
    ds = load_dataset("Mohammed-Altaf/medical-instruction-120k")
    train = ds["train"]
    test = ds["test"]

    train = train.map(process_instruction_batch, batched=True, num_proc=8, remove_columns=train.column_names)
    test = test.map(process_instruction_batch, batched=True, num_proc=8, remove_columns=test.column_names)
    return {"train": train, "test": test}


def combine_datasets(ds1, ds2, seed=42):
    train = concatenate_datasets([ds1["train"], ds2["train"]]).shuffle(seed=seed)
    eval = concatenate_datasets([ds1["test"], ds2["test"]]).shuffle(seed=seed)
    return {"train": train, "eval": eval}


SYSTEM_PROMPT = (
    "You are an empathetic medical assistant. Provide general health information, "
    "not a diagnosis. Encourage consulting a clinician for urgent or uncertain cases."
)

def format_chat(data, tokenizer, system_prompt=SYSTEM_PROMPT):
    combined_user_content = f"{system_prompt}\n\n{data['input']}"
    row_data = [
        {"role": "user", "content": combined_user_content},
        {"role": "model", "content": data["output"]},
    ]
    return tokenizer.apply_chat_template(row_data, tokenize=False)
