import random
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

if __package__:
    from .general import LOGGER  # noqa: E402
else:
    from general import LOGGER  # noqa: E402


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
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
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


def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    LOGGER.debug(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]


class CustomJsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer, block_size: int = 1024) -> None:
        raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        self.tokenizer = tokenizer
        self.block_size = block_size
        tokenized_datasets = []
        for d in raw_data:
            tokenized_datasets.append(self.tokenize_function(d))

        grouped_dataset = self.group_texts(tokenized_datasets)
        self.input_ids = grouped_dataset["input_ids"]
        self.labels = grouped_dataset["labels"]
        self.data = [
            dict(input_ids=self.input_ids[i], labels=self.labels[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, Any]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        return iter(self.data)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    def group_texts(self, examples):
        # Concatenate all texts.
        # Initialize an empty dictionary
        concatenated_examples = {}

        # Loop through the list of dictionaries
        for d in examples:
            # Loop through the keys in each dictionary
            for key in d.keys():
                # If the key is not already a key in the dict_of_lists, create a new list
                if key not in concatenated_examples:
                    concatenated_examples[key] = []
                # Append the value to the list associated with the key in dict_of_lists
                concatenated_examples[key].extend(d[key])
        total_length = len(concatenated_examples["input_ids"])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


if __name__ == "__main__":
    tokenizer_path = r"d:\\models\\opt-125M"

    for dataset in ["wikitext2", "ptb", "c4"]:
        get_loaders(dataset, tokenizer_path=tokenizer_path)

    train_data = CustomJsonDataset(
        tokenizer_path=tokenizer_path,
        block_size=2048,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    samples = get_calib_dataset(tokenizer=tokenizer)
    print(samples)
