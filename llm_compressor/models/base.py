from abc import ABC, abstractmethod


class CompressForCausalLM(ABC):
    @abstractmethod
    def _prepare_attention_module(self):
        pass

    @abstractmethod
    def quantize(self):
        pass

    @abstractmethod
    def prune(self):
        pass

    @abstractmethod
    def save_compressed(self):
        pass

    @abstractmethod
    def get_layers(self):
        pass

    @abstractmethod
    def get_sequential(self):
        pass