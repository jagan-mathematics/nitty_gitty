from abc import ABC, abstractmethod

class Context:
    def __init__(self):
        self.saved_tensors = ()
        self.__axis = None
        self.__reserved_args = {}

    def register_arg(self, key, value):
        self.__reserved_args[key] = value
    
    def get_arg(self, key):
        return self.__reserved_args.get(key)

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    @property
    def axis(self):
        return self.__axis

    @axis.setter
    def axis(self, value):
        self.__axis = value
    

class AutoGradFunction(ABC):
    @staticmethod
    @abstractmethod
    def forward(ctx, *args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def backward(ctx, grad_output):
        pass