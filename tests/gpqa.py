import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

choices = ["A", "B", "C", "D"]


def build_prompt(example):
    prompt = f"Question: {example['question']}\n"
    prompt += "\nExpress your final answer as the corresponding option 'A', 'B', 'C', or 'D'."
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
def compute_gpqa(model, tokenizer, test_df, reasoning="low", max_new_tokens=512):
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

        inputs = torch.cat(
            [outputs,
            tokenizer("\nAnswer:", return_tensors="pt").input_ids.to(device)],
            dim=1,
        )
        logits = model(input_ids=inputs).logits[0, -1]
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer(" A").input_ids[-1]],
                        logits[tokenizer(" B").input_ids[-1]],
                        logits[tokenizer(" C").input_ids[-1]],
                        logits[tokenizer(" D").input_ids[-1]],
                    ]
                ).float(),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
        cors.append(example["answer"] == pred)
    
    return cors


def main(model, tokenizer, n_samples=None):
    dataset = load_dataset("fingertap/GPQA-Diamond", split="test")

    if n_samples:
        dataset = dataset.select(range(n_samples))

    test_df = pd.DataFrame(dataset)
    cors = compute_gpqa(model, tokenizer, test_df)
    weighted_acc = np.mean(cors)
    print("Average accuracy: {:.3f} - GPQA".format(weighted_acc))


if __name__ == "__main__":
    model_path = "/home/models/gpt-oss-20b"
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

    n_samples = None
    main(model, tokenizer, n_samples=n_samples)
