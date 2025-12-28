import re
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer(text):
    matches = re.findall(r'\d+', text)
    if matches:
        return int(matches[-1])
    return None


def build_prompt(example):
    prompt = f"{example['problem']}\n"
    prompt += "Please reason step by step.\n"
    return prompt


def build_inputs(tokenizer, prompts, device, reasoning="low"):
    messages = [{
        "role": "user", 
        "content": prompts,
    }]
    prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True, 
        return_tensors="pt",
    )
    prompts = prompts.replace("Reasoning: medium", f"Reasoning: {reasoning}")
    inputs = tokenizer(prompts, return_tensors="pt").to(device)
    return prompts, inputs


@torch.inference_mode()
def compute_aime(model, tokenizer, test_df, reasoning="low", max_new_tokens=2048):
    cors = []
    device = model.device
    
    for i in range(test_df.shape[0]):
        example = test_df.iloc[i]
        user_prompts = build_prompt(example)
        prompts, inputs = build_inputs(tokenizer, user_prompts, device, reasoning)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        final_prompt = "\nPut your final answer within \\boxed{{}}."
        final_prompt += "\n\\boxed"

        inputs = torch.cat(
            [outputs,
            tokenizer(final_prompt, return_tensors="pt").input_ids.to(device)],
            dim=1,
        )
        attention_mask = torch.ones_like(inputs)

        outputs = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        outputs = outputs[0][inputs.shape[-1]-2:]
        pred = tokenizer.decode(outputs, skip_special_tokens=True)
        pred = extract_answer(pred)
        gt = example["solution"] if "solution" in example else example["answer"]
        gt = extract_answer(gt)
        cors.append(gt == pred)

    return cors


def main(model, tokenizer, n_samples=None):
    dataset = load_dataset("math-ai/aime24", split="test")

    if n_samples:
        dataset = dataset.select(range(n_samples))

    test_df = pd.DataFrame(dataset)
    cors = compute_aime(model, tokenizer, test_df)
    weighted_acc = np.mean(cors)
    print("Average accuracy: {:.3f} - AIME24".format(weighted_acc))

    dataset = load_dataset("math-ai/aime25", split="test")

    if n_samples:
        dataset = dataset.select(range(n_samples))

    test_df = pd.DataFrame(dataset)
    cors = compute_aime(model, tokenizer, test_df)
    weighted_acc = np.mean(cors)
    print("Average accuracy: {:.3f} - AIME25".format(weighted_acc))


if __name__ == "__main__":
    # model_path = "/home/models/llama-3.1-8b-it"
    model_path = "/home/models/gpt-oss-20b"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device('cuda:1')
    model = model.to(device)
    model.eval()

    n_samples = None
    main(model, tokenizer, n_samples=n_samples)
