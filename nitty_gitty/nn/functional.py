from ..nn.module import Module
from ..core.device import Device, get_cupy_engine, get_numpy_engine, is_cuda_available, normalize_device, normalize_dtype
from ..core.tensor import Tensor, apply_op
from ..ops import activation_op
from ..core.auto_grad import Context


class function(Module):
    def __init__(self, ops):
        super().__init__()
        self._ops = ops

    def __call__(self, x, *args, **kwargs):
        if self._ops is None:
            raise NotImplementedError("Op Not found")
        return apply_op(self._ops, x, *args, **kwargs)
    
    def get_parameters(self):
        return None
    

class ReLU(function):
    def __init__(self):
        super().__init__(ops=activation_op.ReLUOp)


class Sigmoid(function):
    def __init__(self):
        super().__init__(ops=activation_op.SigmoidOp)



class TanH(function):
    def __init__(self):
        super().__init__(ops=activation_op.TanHOp)




class SoftSign(function):
    def __init__(self):
        super().__init__(ops=activation_op.SoftSignOp)



class SoftPlus(function):
    def __init__(self):
        super().__init__(ops=activation_op.SoftPlusOp)




class LeakyReLU(function):
    def __init__(self):
        super().__init__(ops=activation_op.LeakyReLUOp)
    
    def __call__(self, x, alpha=0.01):
        return super().__call__(x, alpha=0.01)





class PReLU(function):
    def __init__(self, num_parameters=1, init=0.25, device=Device.CPU, dtype=None):
        super().__init__(ops=activation_op.PReLUOp)
        self.num_parameters = num_parameters
        self.init = init
        device = normalize_device(device)
        self._device = device
        self._dtype = dtype
        dtype = normalize_dtype(dtype)
        engine = get_cupy_engine() if is_cuda_available() and device == Device.GPU else get_numpy_engine()
        self.weight = Tensor(engine.full((num_parameters,), init), device=device, dtype=dtype, requires_grad=True)
    
    def __call__(self, x):
        return apply_op(self._ops, x, self.weight)

    def get_parameters(self):
        return [self.weight]

    def reset_parameters(self) -> None:
        engine = get_cupy_engine() if is_cuda_available() and self._device == Device.GPU else get_numpy_engine()
        self.weight.data = engine.full((self.num_parameters,), self.init)




class RReLU(function):
    def __init__(self, lower=0.125, upper=1.0/3):
        self.lower = lower
        self.upper = upper
        super().__init__(ops=activation_op.RReluOp)
    
    def __call__(self, x):
        return apply_op(self._ops, x, lower=self.lower, upper=self.upper, training=self.training)




class Elu(function):
    def __init__(self):
        super().__init__(ops=activation_op.ELUOp)
    
    def __call__(self, x, alpha=1.0):
        return super().__call__(x, alpha=alpha)
    


class SeLU(function):
    def __init__(self):
        super().__init__(ops=activation_op.SELUOp)
    
    def __call__(self, x, alpha = 1.6732632423543772, scale = 1.0507009873554805):
        return super().__call__(x, alpha=alpha, scale=scale)
                                



class GeLU(function):
    def __init__(self):
        super().__init__(ops=activation_op.GELUOp)



class SiLU(function):
    def __init__(self):
        super().__init__(ops=activation_op.SiLUOp)
    


class Mish(function):
    def __init__(self):
        super().__init__(ops=activation_op.MishOp)


class SoftMax(function):
    def __init__(self):
        super().__init__(ops=activation_op.SoftMaxOp)



class GLU(function):
    def __init__(self):
        super().__init__(ops=activation_op.GLUOp)
    
    def __call__(self, x, y):
        return apply_op(self._ops, x, y)



class SwiGLU(function):
    def __init__(self):
        super().__init__(ops=activation_op.SwiGLUOp)
    
    def __call__(self, x, y):
        return apply_op(self._ops, x, y)


