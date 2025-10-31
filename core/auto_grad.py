from abc import ABC, abstractmethod

class Context:
    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

class AutoGradFunction(ABC):
    @staticmethod
    @abstractmethod
    def forward(ctx, *args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def backward(ctx, grad_output):
        pass