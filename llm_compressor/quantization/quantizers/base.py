from abc import ABC, abstractmethod


class BaseQuantizer(ABC):
    @abstractmethod
    def configure(self):
        pass

    @abstractmethod
    def find_params(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def fake_quantize(self):
        pass

    @abstractmethod
    def extra_repr(self):
        pass
