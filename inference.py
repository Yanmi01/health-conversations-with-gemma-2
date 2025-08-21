import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(base_model: str, finetuned_model: str, device: str = "auto"):

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map=device,
        trust_remote_code=False,
        revision="main",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    model = PeftModel.from_pretrained(base_model, finetuned_model)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    return model, tokenizer


def build_prompt(user_input: dict, tokenizer):

    row_json = [
        {
            "role": "user", 
            "content": f"Description: {user_input['Description']}\nPatient: {user_input['Patient']}"
        }
    ]

    return tokenizer.apply_chat_template(
        row_json,
        tokenize=False,
        add_generation_prompt=True,
    )

def generate_response(model, tokenizer, prompt: str,
                      max_new_tokens=512, temperature=0.7, top_k=50, top_p=0.95):

    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    base_model = "google/gemma-2b-it"
    finetuned_model = "Yanmife/gemma-2b-it-health"

    model, tokenizer = load_model(base_model, finetuned_model)

    new_input = {
        "Description": "I'm a 35-year-old male with a history of hypertension. "
                       "I've been having headaches and dizziness for the past week. "
                       "My blood pressure readings have been higher than usual.",
        "Patient": "What could be causing this? I'm worried it's something serious.",
    }

    formatted_prompt = build_prompt(new_input, tokenizer)
    response = generate_response(model, tokenizer, formatted_prompt)

    print(response)
