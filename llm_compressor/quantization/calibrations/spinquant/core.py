import os
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM

PATH = Path(__file__).resolve().parents[3]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from utils.general import LOGGER  # noqa: E402
from quantization.calibrations.rtn.core import rtn  # noqa: E402, F401
from quantization.calibrations.gptq.core import gptq  # noqa: E402, F401
from quantization.calibrations.spinquant.modeling.llama import SpinLlamaForCausalLM  # noqa: E402
from quantization.calibrations.spinquant.rotation_utils import (
    rotate_model,
    random_hadamard_matrix,
)  # noqa: E402
from quantization.calibrations.spinquant.fuse_norm_utils import fuse_layer_norms  # noqa: E402
from quantization.calibrations.spinquant.optimizer import SGDG  # noqa: E402
from utils.dataset import CustomJsonDataset  # noqa: E402
from utils.torch_utils import cleanup_memory  # noqa: E402


class RotateModule(nn.Module):
    def __init__(self, R_init, device):
        super(RotateModule, self).__init__()
        self.weight = nn.Parameter(R_init.to(torch.float32).to(device))

    def forward(self, x, transpose=False):
        if transpose:
            return x @ self.weight
        else:
            return self.weight @ x


def spinquant(
    model,
    device,
    mode="hadamard",
    n_samples=128,
    seq_len=2048,
    mse=False,
    verbose=True,
    **kwargs,
):
    if verbose:
        LOGGER.info("Rotating model... [Quant-method : SpinQuant]")

    if mode == "optimize":
        LOGGER.info("Optimizing rotation matrix...")
        model_path = model.config._name_or_path
        quant_config = kwargs.get("quant_config")

        if isinstance(model, LlamaForCausalLM):
            spin_model = SpinLlamaForCausalLM.from_pretrained(
                model_path,
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            raise RuntimeError(f"Not support model yet, got {model_path}")

        spin_model._prepare_model(quant_config=quant_config, mse=mse)
        spin_model.config.use_cache = False
        process_word_embeddings = spin_model.config.tie_word_embeddings
        if process_word_embeddings:
            spin_model.config.tie_word_embeddings = False
            spin_model.lm_head.weight.data = (
                spin_model.model.embed_tokens.weight.data.clone()
            )

        for param in spin_model.parameters():
            param.requires_grad = False

        R1 = random_hadamard_matrix(spin_model.config.hidden_size, spin_model.device)
        spin_model.R1 = RotateModule(R1, spin_model.device)
        for i in range(spin_model.config.num_hidden_layers):
            head_dim = (
                spin_model.config.hidden_size // spin_model.config.num_attention_heads
            )
            R2 = random_hadamard_matrix(head_dim, spin_model.device)
            spin_model.model.layers[i].self_attn.R2 = RotateModule(
                R2, spin_model.device
            )

        spin_model.seqlen = seq_len
        trainable_parameters = [spin_model.R1.weight] + [
            spin_model.model.layers[i].self_attn.R2.weight
            for i in range(spin_model.config.num_hidden_layers)
        ]
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path,
            model_max_length=seq_len,
            padding_side="right",
            use_fast=True,
            add_eos_token=False,
            add_bos_token=False,
        )
        train_data = CustomJsonDataset(tokenizer=tokenizer, block_size=seq_len)
        training_args = TrainingArguments(
            fp16=False,
            bf16=True,
            log_on_each_node=False,
            per_device_train_batch_size=1,
            logging_steps=1,
            learning_rate=1.5,
            weight_decay=0.0,
            lr_scheduler_type="cosine",
            gradient_checkpointing=True,
            save_safetensors=False,
            max_steps=100,
            save_strategy="no",
        )
        optimizer = SGDG(
            trainable_parameters, lr=training_args.learning_rate, stiefel=True
        )
        trainer = Trainer(
            model=spin_model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=None,
            data_collator=default_data_collator,
            optimizers=(optimizer, None),
        )
        trainer.train()
        cpu_state = trainer.model.state_dict()
        R_dict = {
            key.replace(".weight", ""): value
            for key, value in cpu_state.items()
            if "R1.weight" in key or "self_attn.R2" in key
        }
        torch.save(R_dict, os.path.join(kwargs.get("rotation_path"), "R.bin"))
        del spin_model
        cleanup_memory(verbose=False)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()

    process_word_embeddings = model.config.tie_word_embeddings
    if process_word_embeddings:
        model.config.tie_word_embeddings = False
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    fuse_layer_norms(model)
    if mode == "optimize":
        rotate_model(model, "hadamard", device, rotation_path=kwargs.get("rotation_path"))
    else:
        rotate_model(model, mode, device)

    # rtn(model, device, mse=mse, verbose=False)
    gptq(model, device, n_samples, seq_len, mse=mse, verbose=False)

    model.config.use_cache = use_cache
    if verbose:
        LOGGER.info("Quantization complete !")
    return


if __name__ == "__main__":
    from easydict import EasyDict
    from models.opt import CompressOPTForCausalLM  # noqa: F401
    from models.bloom import CompressBloomForCausalLM  # noqa: F401
    from models.llama import CompressLlamaForCausalLM  # noqa: F401
    from models.phi import CompressPhiForCausalLM  # noqa: F401

    device = torch.device("cuda:0")
    quant_config = EasyDict({})
    quant_config.linear = EasyDict({})
    quant_config.linear.weight = {
        "type": "int",
        "format": "int4",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }
    quant_config.linear.act_in = {
        "type": None,
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }
    quant_config.linear.act_out = {
        "type": None,
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }
    quant_config.matmul = EasyDict({})
    quant_config.matmul.act_in = {
        "type": None,
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }
    quant_config.matmul.act_out = {
        "type": None,
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }
    quant_config.head = EasyDict({})
    quant_config.head.weight = {
        "type": None,
        "format": "int8",
        "group_size": -1,
        "axes": -1,
        "zero_point": False,
        "device": device,
    }
    quant_config.head.act_in = {
        "type": None,
        "format": None,
        "group_size": None,
        "axes": None,
        "zero_point": None,
        "device": None,
    }
    quant_config.head.act_out = {
        "type": None,
        "format": None,
        "group_size": None,
        "axes": None,
        "zero_point": None,
        "device": None,
    }

    # model_path = "d:\\models\\opt-125m"
    # model = CompressOPTForCausalLM.from_pretrained(
    #     model_path,
    #     attn_implementation="eager",
    #     torch_dtype=torch.bfloat16,
    #     device_map="cpu",
    # )
    # model_path = "d:\\models\\bloom-560m"
    # model = CompressBloomForCausalLM.from_pretrained(
    #     model_path,
    #     attn_implementation="eager",
    #     torch_dtype=torch.bfloat16,
    #     device_map="cpu",
    # )
    model_path = "d:\\models\\llama-3.2-1b-it"
    model = CompressLlamaForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    # model_path = "d:\\models\\phi-1.5"
    # model = CompressPhiForCausalLM.from_pretrained(
    #     model_path,
    #     attn_implementation="eager",
    #     torch_dtype=torch.bfloat16,
    #     device_map="cpu",
    # )
    model._prepare_attention_module(quant_config)
    spinquant(model, device)
    print(model)
