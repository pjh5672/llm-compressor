import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

choices = ["A", "B", "C", "D"]

def load_model(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return tokenizer, model

def build_inputs(tokenizer, question, device):
    reasoning = "low"
    messages = [{
        "role": "user", 
        "content": question + "\nLet's think step by step.\n",
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
def compare_logits(tokenizer, model, inputs):
    logits = model(**inputs).logits[0, -1]
    probs = (
        torch.nn.functional.softmax(
            torch.tensor(
                [
                    max(logits[tokenizer("A").input_ids[-1]], logits[tokenizer(" A").input_ids[-1]]),
                    max(logits[tokenizer("B").input_ids[-1]], logits[tokenizer(" B").input_ids[-1]]),
                    max(logits[tokenizer("C").input_ids[-1]], logits[tokenizer(" C").input_ids[-1]]),
                    max(logits[tokenizer("D").input_ids[-1]], logits[tokenizer(" D").input_ids[-1]]),
                ]
            ).float(),
            dim=0,
        )
        .detach()
        .cpu()
        .numpy()
    )
    for i, p in enumerate(probs):
        print("{}. {:.4f}".format(choices[i], p))
    return

@torch.inference_mode()
def compare_cot(tokenizer, model, inputs, max_new_tokens=512):
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    gen_ids = outputs[0][inputs.input_ids.shape[-1]:]
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=False)
    print(decoded)

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
                    max(logits[tokenizer("A").input_ids[-1]], logits[tokenizer(" A").input_ids[-1]]),
                    max(logits[tokenizer("B").input_ids[-1]], logits[tokenizer(" B").input_ids[-1]]),
                    max(logits[tokenizer("C").input_ids[-1]], logits[tokenizer(" C").input_ids[-1]]),
                    max(logits[tokenizer("D").input_ids[-1]], logits[tokenizer(" D").input_ids[-1]]),
                ]
            ).float(),
            dim=0,
        )
        .detach()
        .cpu()
        .numpy()
    )
    for i, p in enumerate(probs):
        print("{}. {:.4f}".format(choices[i], p))
    return 


if __name__ == "__main__":
    model_path = "/home/models/gpt-oss-20b"
    device = torch.device("cuda")
    tokenizer, model = load_model(model_path, device)
    question = """The following are multiple choice questions (with answers) about mathmetics.
Question: 2+2=?
A. 3
B. 4
C. 5
D. 6

Answer:"""
    prompts, inputs = build_inputs(tokenizer, question, device)
    print(prompts)
    # compare_logits(tokenizer, model, inputs)
    compare_cot(tokenizer, model, inputs)