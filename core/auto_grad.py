from abc import ABC, abstractmethod

class Context:
    def __init__(self):
        self.saved_tensors = ()
        self.__axis = None

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    @property.setter
    def axis(self, value):
        self.__axis = value
    
    @property
    def axis(self):
        return self.__axis
    

class AutoGradFunction(ABC):
    @staticmethod
    @abstractmethod
    def forward(ctx, *args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def backward(ctx, grad_output):
        pass