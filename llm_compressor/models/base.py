from typing import List, Union, Dict
from typing_extensions import Doc, Annotated

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizer,
)

from quantization.config import QuantConfig


class CompressBaseForCausalLM(nn.Module):
    def __init__(
        self,
        model: Annotated[PreTrainedModel, Doc("The pretrained or quantized model.")],
        model_type: Annotated[str, Doc("The model type, found in config.json.")],
        is_quantized: Annotated[bool, Doc("Indicates if the current model is quantized.")],
        config: Annotated[PretrainedConfig, Doc("The config of the model.")],
        quant_config: Annotated[QuantConfig, Doc("The quantization config of the model.")],
    ):
        """The base model for all AutoCompress models."""
        super().__init__()
        self.model: PreTrainedModel = model
        self.model_type: str = model_type
        self.is_quantized: bool = is_quantized
        self.search_result = None
        self.config: PretrainedConfig = config
        self.quant_config = quant_config

    def to(self, device: Annotated[str, Doc("The device to move your model to.")]):
        """A utility function for moving the model to a device."""
        return self.model.to(device)

    def forward(self, *args, **kwargs):
        """A forward function that mimics the torch forward."""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """A generate function that mimics the HF generate function."""
        with torch.inference_mode():
            return self.model.generate(*args, **kwargs)
    
    def quantize(
        self,
        tokenizer: Annotated[PreTrainedTokenizer, Doc("The tokenizer to use for quantization.")] = None,
        quant_config: Annotated[Dict, Doc("The quantization config you want to use.")] = {},
        calib_data: Annotated[
            Union[str, List[str]], 
            Doc("The calibration dataset. Either a string pointing to Huggingface or a list of preloaded examples.")
            ] = "pileval",
        split: Annotated[str, Doc("The split of calib_data.")] = "train",
        text_column: Annotated[str, Doc("The text column of calib_data.")] = "text",
        n_parallel_calib_samples: Annotated[
            int,
            Doc(
                "The number of parallel samples to run through the model. "
                "A high number of parallel samples can result in OOM during quantization if max_calib_samples is high enough. "
                "If None, runs through all samples at the same time. "
                "You can set this to a low number for more memory efficient quantization."
            ),
        ] = None,
        max_calib_samples: Annotated[int, Doc("The maximum number of samples to run through the model.")] = 128,
        max_calib_seq_len: Annotated[
            int,
            Doc("The maximum sequence length of the calibration dataset. Discard samples greater than max_calib_seq_len."),
        ] = 512,
        **kwargs,
    ):
        """
        The main quantization function that you can use to quantize your model.

        Example:

        ```python
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        model_path = "..."
        model = AutoAWQForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
        model.quantize(tokenizer, quant_config)
        ```
        """

    def save_quantized(
        self,
        save_dir: Annotated[str, Doc("The directory to save your model to.")],
        safetensors: Annotated[
            bool, Doc("Whether to save the model as safetensors or torch files.")
        ] = True,
        shard_size: Annotated[
            str, Doc("The shard size for sharding large models into multiple chunks.")
        ] = "5GB",
    ):
        pass

    @classmethod
    def from_pretrained(
        self,
        model_path: Annotated[str, Doc("A Huggingface path or local path to a model.")],
        model_type: Annotated[str, Doc("The model type, loaded from config.json.")],
        torch_dtype: Annotated[
            torch.dtype,
            Doc("The dtype to load the model as. May not work with other values than float16."),
        ] = torch.float16,
        trust_remote_code: Annotated[
            bool,
            Doc("Useful for Huggingface repositories that have not been integrated into transformers yet."),
        ] = True,
        safetensors: Annotated[
            bool, 
            Doc("Whether to download/load safetensors instead of torch weights.")
        ] = True,
        device_map: Annotated[
            Union[str, Dict],
            Doc("A device map that will be passed onto the model loading method from transformers."),
        ] = "auto",
        low_cpu_mem_usage: Annotated[
            bool,
            Doc("Use low_cpu_mem_usage when loading from transformers.")
        ] = True,
        use_cache: Annotated[
            bool,
            Doc("Use use_cache argument in transformers")
        ] = False,
    ):
        pass

    @classmethod
    def from_quantized(
        self,
        model_path: Annotated[str, Doc("A Huggingface path or local path to a model.")],
        model_type: Annotated[str, Doc("The model type, loaded from config.json.")],
        model_filename: Annotated[str, Doc("Load a specific model's filename by specifying this argument.")] = "",
        max_seq_len: Annotated[
            int,
            Doc("The maximum sequence cached sequence length of the model. Larger values may increase loading time and memory usage."),
        ] = None,
        torch_dtype: Annotated[
            torch.dtype,
            Doc("The dtype to load the model as. May not work with other values than float16."),
        ] = torch.float16,
        trust_remote_code: Annotated[
            bool,
            Doc("Useful for Huggingface repositories that have not been integrated into transformers yet."),
        ] = True,
        safetensors: Annotated[
            bool, Doc("Whether to download/load safetensors instead of torch weights.")
        ] = True,
        device_map: Annotated[
            Union[str, Dict],
            Doc("A device map that will be passed onto the model loading method from transformers."),
        ] = "balanced",
        max_memory: Annotated[
            Dict[Union[int, str], Union[int, str]],
            Doc('A dictionary device identifier to maximum memory which will be passed onto the model loading method from transformers. For example：{0: "4GB",1: "10GB"'),
        ] = None,
    ):
        pass
