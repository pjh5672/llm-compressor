import sys
from pathlib import Path

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
                rotation_path = kwargs.get("rotation_path")
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

    def generate_text(self, prompt, tokenizer, max_new_tokens=512):
        device = self.device
        mem_kwargs = {"max_memory": get_balanced_memory(self)}
        device_map = infer_auto_device_map(
            model=self, no_split_module_classes=self._no_split_modules, **mem_kwargs
        )
        model = dispatch_model(self, device_map=device_map)

        inputs = tokenizer(prompt, reteurn_tensors="pt").to(self.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        del model
        cleanup_memory(verbose=True)
        self.to(device)
        return outputs[0]
