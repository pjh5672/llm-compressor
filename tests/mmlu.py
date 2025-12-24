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


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    options = df.iloc[idx, -2]
    for i, o in enumerate(options):
        prompt += "\n{}. {}".format(choices[i], o)
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(choices[df.iloc[idx, -1]])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def parse_answer(text):
    pattern = r"\b[A-D]\b"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        print("Not matching [A-D], randomize choices.")
        return choices[random.randint(0, len(choices)-1)]


@torch.no_grad()
def compute_mmlu(subject, model, tokenizer, dev_df, test_df):
    cors = []
    device = model.device

    gen_kwargs = dict(
        max_new_tokens=1,     # A/B/C/D + </final>
        do_sample=False,      # temperature = 0
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    for i in range(test_df.shape[0]):
        k = 0
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        while inputs.input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

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
    dev_df = pd.DataFrame(ds["dev"])

    all_cors = []
    subjects = sorted(set(test_df["subject"]))

    for subject in subjects:
        cate_test_df = test_df[test_df["subject"] == subject]
        cate_dev_df = dev_df[dev_df["subject"] == subject]
        cors = compute_mmlu(subject, model, tokenizer, cate_dev_df, cate_test_df)
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
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()

    main(model, tokenizer)

    