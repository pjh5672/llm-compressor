import sys
from pathlib import Path

import torch
from accelerate import dispatch_model
from accelerate.utils import infer_auto_device_map, get_balanced_memory

PATH = Path(__file__).resolve().parents[1]
if str(PATH) not in sys.path:
    sys.path.append(str(PATH))

from utils.torch_utils import cleanup_memory  # noqa: E402
from pruning.magnitude.core import magnitude  # noqa: E402
from quantization.calibrations.rtn.core import rtn  # noqa: E402
from quantization.calibrations.awq.core import awq  # noqa: E402
from quantization.calibrations.gptq.core import gptq  # noqa: E402
from quantization.calibrations.awq_plus.core import awq_plus  # noqa: E402
from quantization.calibrations.spinquant.core import spinquant  # noqa: E402
from utils.module import (
    text_to_token_ids,
    token_ids_to_text,
    chat_template,
    extract_response,
)  # noqa: E402


class CompressForCausalLM:
    def _prepare_attention_module(self):
        raise NotImplementedError

    def save_compressed(self):
        raise NotImplementedError

    def get_layers(self):
        raise NotImplementedError

    def get_sequential(self):
        raise NotImplementedError

    def move_embed(self):
        raise NotImplementedError

    def quantize(self, tokenizer, quant_method, quant_config, device, **kwargs):
        if kwargs.get("quantize"):
            self._prepare_attention_module(quant_config)

            if quant_method == "rtn":
                rtn(self, device, mse=True, verbose=True)

            elif quant_method == "awq":
                n_samples = kwargs.get("n_samples", 128)
                seq_len = kwargs.get("seq_len", 2048)
                awq(
                    self,
                    device,
                    tokenizer,
                    n_samples=n_samples,
                    seq_len=seq_len,
                    verbose=True,
                )
            elif quant_method == "gptq":
                n_samples = kwargs.get("n_samples", 128)
                seq_len = kwargs.get("seq_len", 2048)
                gptq(
                    self,
                    device,
                    n_samples=n_samples,
                    seq_len=seq_len,
                    mse=True,
                    verbose=True,
                )
            elif quant_method == "awq_plus":
                n_samples = kwargs.get("n_samples", 128)
                seq_len = kwargs.get("seq_len", 2048)
                awq_plus(
                    self,
                    device,
                    tokenizer,
                    n_samples=n_samples,
                    seq_len=seq_len,
                    verbose=True,
                )
            elif quant_method == "spinquant-opt":
                n_samples = kwargs.get("n_samples", 128)
                seq_len = kwargs.get("seq_len", 2048)
                rotation_path = kwargs.get("rotation_path", "./")
                spinquant(
                    self,
                    device,
                    mode="optimize",
                    n_samples=n_samples,
                    seq_len=seq_len,
                    mse=True,
                    verbose=True,
                    quant_config=quant_config,
                    rotation_path=rotation_path,
                )
            elif quant_method == "spinquant-had":
                n_samples = kwargs.get("n_samples", 128)
                seq_len = kwargs.get("seq_len", 2048)
                rotation_path = kwargs.get("rotation_path")
                spinquant(
                    self,
                    device,
                    mode="hadamard",
                    n_samples=n_samples,
                    seq_len=seq_len,
                    mse=True,
                    verbose=True,
                    quant_config=quant_config,
                    rotation_path=rotation_path,
                )
        else:
            return

    def prune(self, tokenizer, prune_method, prune_config, device, **kwargs):
        if kwargs.get("prune"):
            sparsity_ratio = prune_config.pop("sparsity_ratio")

            if prune_method == "magnitude":
                magnitude(self, device, sparsity_ratio)
                return
        else:
            return

    @torch.inference_mode()
    def generate_text(self, prompt, tokenizer, **kwargs):
        seq_len = kwargs.get("seq_len", 512)
        max_new_tokens = kwargs.get("max_new_tokens", 100)
        temperature = kwargs.get("temperature", 0.0)
        top_k = kwargs.get("top_k", None)
        eos_id = tokenizer.eos_token_id
        device = self.device

        mem_kwargs = {"max_memory": get_balanced_memory(self)}
        device_map = infer_auto_device_map(
            model=self, no_split_module_classes=self._no_split_modules, **mem_kwargs
        )
        model = dispatch_model(self, device_map=device_map)

        prompt = chat_template(prompt)
        idx = text_to_token_ids(prompt, tokenizer)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -seq_len:]
            outputs = model(idx_cond.to(model.device))
            logits = outputs.logits[:, -1, :]

            # Filter logits with top_k sampling
            if top_k is not None:
                # Keep only top_k values
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float("-inf")).to(logits.device),
                    logits,
                )

            # Apply temperature scaling
            if temperature > 0.0:
                logits = logits / temperature

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            # get idx of the vocab entry with the highest logits value
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

            # stop generating early if end-of-sequence token is encountered and eos_id is specified
            if idx_next == eos_id:
                break

            # Same as before: append sampled index to the running sequence
            idx = torch.cat(
                (idx, idx_next.clone().detach().cpu()), dim=1
            )  # (batch_size, num_tokens+1)

        del model, idx_cond, logits
        cleanup_memory(verbose=True)
        self.to(device)

        return extract_response(token_ids_to_text(idx, tokenizer), prompt)
