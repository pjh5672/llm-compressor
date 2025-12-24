import os
import re
import random

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def build_user_prompt(df, idx):
    # GPQA: The following is a graduate-level multiple choice question in {domain}.
    # AIME: The following is an AIME 2024 Problem. Problem: {question} Give only the final answer as an a non-negative integer. Answer:
    subject_line = (
        "The following are multiple choice questions (with answers) "
        f"about {format_subject(df.iloc[idx, 1])}.\n"
    )

#     harmony = f"""
# <harmony>
#   <question>
#     {df.iloc[idx, 0]}
#   </question>

#   <choices>
#     A. {df.iloc[idx, -2][0]}
#     B. {df.iloc[idx, -2][1]}
#     C. {df.iloc[idx, -2][2]}
#     D. {df.iloc[idx, -2][3]}
#   </choices>

#   <instruction>
#     Select the single correct answer.
#   </instruction>
# </harmony>

# <harmony>
#   <final>
# """
    harmony = f"""
Question:
{df.iloc[idx, 0]}

Choices:
A. {df.iloc[idx, -2][0]}
B. {df.iloc[idx, -2][1]}
C. {df.iloc[idx, -2][2]}
D. {df.iloc[idx, -2][3]}

Select the single correct answer.
Answer:
"""
    return subject_line + harmony


def build_messages(df, idx):
    # GPQA : You are an expert-level academic assistant. same belows"
    # AIME: You are a competition-level mathematician. Return only the final numeric answer. Do not provide explanations."
#     system_prompt = """You are an AI assistant participating in an academic benchmark evaluation.

# Use the Harmony format strictly.
# Do NOT reveal chain-of-thought or intermediate reasoning.
# Output only the final answer choice (A, B, C, or D) in the <final> field.
# """
    system_prompt = """
You are an AI assistant participating in an academic benchmark evaluation.
Answer with a single capital letter (A, B, C, or D).
Do not provide explanations.
"""
    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": build_user_prompt(df, idx)
        }
    ]


def parse_answer(text):
    pattern = r"\b[A-D]\b"
    # AIME : r"\b\d+\b"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        print("Not matching [A-D], randomize choices.")
        return choices[random.randint(0, len(choices)-1)]


@torch.no_grad()
def compute_mmlu(subject, model, tokenizer, test_df):
    cors = []
    device = model.device

    gen_kwargs = dict(
        max_new_tokens=1,
        do_sample=False,      # temperature = 0
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    for i in range(test_df.shape[0]):
        messages = build_messages(test_df, i)
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # <assistant> 시작
        )
        inputs = tokenizer(
            prompt_str,
            return_tensors="pt",
            add_special_tokens=False
        ).to(device)

        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[0][inputs.input_ids.shape[-1]:]
        pred = tokenizer.decode(outputs, skip_special_tokens=True)
        pred = parse_answer(pred)

        gt = choices[test_df.iloc[i, -1]]
        cors.append(gt == pred)

    print("Average accuracy {:.3f} - {}".format(np.mean(cors), subject))
    return cors


def main(model, tokenizer):
    ds = load_dataset("cais/mmlu", "all")
    test_df = pd.DataFrame(ds["test"])

    all_cors = []
    subjects = sorted(set(test_df["subject"]))

    for subject in subjects:
        cate_test_df = test_df[test_df["subject"] == subject]
        cors = compute_mmlu(subject, model, tokenizer, cate_test_df)
        all_cors.append(cors)

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    model_path = "d:\\models\\llama-3.2-1b-it"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device('cuda')
    model = model.to(device)
    model.eval()

    main(model, tokenizer)

    