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


def format_example(df, idx):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(df.iloc[idx, 1])
    )
    prompt += f"Question: {df.iloc[idx, 0]}\n"

    options = df.iloc[idx, -2]
    for i, o in enumerate(options):
        prompt += "\n{}. {}".format(choices[i], o)
    prompt += "\nAnswer:\nLet's think step by step.\n"
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
def compute_mmlu(subject, model, tokenizer, test_df, n_samples=None, reasoning="low", max_new_tokens=512):
    cors = []
    device = model.device

    for i in range(test_df.shape[0]):
        user_prompts = format_example(test_df, i)
        prompts, inputs = build_inputs(tokenizer, user_prompts, device, reasoning)
        # print(user_prompts)
        # print(prompts)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        # gen_ids = outputs[0][inputs.input_ids.shape[-1]:]
        # decoded = tokenizer.decode(gen_ids, skip_special_tokens=False)
        # print(decoded)

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
        gt = choices[test_df.iloc[i, -1]]
        cors.append(gt == pred)

        if i == n_samples:
            break
    
    print("Average accuracy {:.3f} - {}".format(np.mean(cors), subject))
    return cors


def main(model, tokenizer, n_samples=None):
    test_dataset = load_dataset("cais/mmlu", "all", split="test")
    test_df = pd.DataFrame(test_dataset)

    all_cors = []
    subjects = sorted(set(test_df["subject"]))

    for subject in subjects:
        cate_test_df = test_df[test_df["subject"] == subject]
        cors = compute_mmlu(subject, model, tokenizer, cate_test_df, n_samples=n_samples)
        all_cors.append(cors)

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))


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

    n_samples = 30
    main(model, tokenizer, n_samples=n_samples)
