import random

from datasets import load_dataset
from transformers import AutoTokenizer


def get_wikitext2(tokenizer_path, nsamples, seqlen, seed):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(tokenizer_path, nsamples, seqlen, seed):
    traindata = load_dataset(
        "ptb_text_only", "penn_treebank", split="train", trust_remote_code=True
    )
    testdata = load_dataset(
        "ptb_text_only", "penn_treebank", split="test", trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(tokenizer_path, nsamples, seqlen, seed):
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, tokenizer_path, nsamples=128, seqlen=2048, seed=0):
    if "wikitext2" in name:
        return get_wikitext2(tokenizer_path, nsamples, seqlen, seed)
    elif "ptb" in name:
        return get_ptb(tokenizer_path, nsamples, seqlen, seed)
    elif "c4" in name:
        return get_c4(tokenizer_path, nsamples, seqlen, seed)
    raise RuntimeError(f"Invalid dataset name, got {name}")


if __name__ == "__main__":
    for dataset in ["wikitext2", "ptb", "c4"]:
        get_loaders(dataset, tokenizer_path=r"d:\\models\\opt-125M")
