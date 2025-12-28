import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(example):
    prompt = (
        f"Question: {example['goal']}\n\n"
        f"Answer:"
    )
    return prompt


@torch.inference_mode()
def log_likelihood(prompt, continuation):
    """
    log P(continuation | prompt)
    """
    full_text = prompt + continuation

    enc = tokenizer(
        full_text,
        return_tensors="pt",
        padding=False,
    ).to(device)

    input_ids = enc.input_ids

    # continuation 토큰 위치
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    cont_len = input_ids.shape[1] - prompt_ids.shape[1]

    outputs = model(input_ids).logits

    # continuation 부분 logits
    cont_logits = outputs[:, -cont_len-1:-1, :]
    cont_ids = input_ids[:, -cont_len:]

    log_probs = F.log_softmax(cont_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=cont_ids.unsqueeze(-1),
    ).squeeze(-1)

    return token_log_probs.mean().item()


def compute_piqa(model, tokenizer, val_df):
    cors = []
    device = model.device

    for i in range(val_df.shape[0]):
        example = val_df.iloc[i]
        prompt = build_prompt(example)

        scores = {}
        scores[0] = log_likelihood(prompt, f" {example['sol1']}")
        scores[1] = log_likelihood(prompt, f" {example['sol2']}")

        pred = max(scores, key=scores.get)
        cors.append(example["label"] == pred)

    return cors


def main(model, tokenizer, n_samples=None):
    dataset = load_dataset("regisss/piqa", split="validation")

    if n_samples:
        dataset = dataset.select(range(n_samples))

    val_df = pd.DataFrame(dataset)
    cors = compute_piqa(model, tokenizer, val_df)
    weighted_acc = np.mean(cors)
    print("Average accuracy: {:.3f} - PIQA".format(weighted_acc))


if __name__ == "__main__":
    model_path = "/home/models/llama-3.2-1b-it"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device('cuda:3')
    model = model.to(device)
    model.eval()

    n_samples = 30
    main(model, tokenizer, n_samples=n_samples)
