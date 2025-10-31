from ..core.auto_grad import AutoGradFunction
from ..core.device import check_device_compatibility, get_array_module, get_cupy_engine, get_numpy_engine
from ..core.exceptions import OpGradNonImplemented
from ..core.utils import reduce_broadcast


class Add(AutoGradFunction):
    @staticmethod
    def forward(ctx, a, b):
        check_device_compatibility(a, b)
        ctx.save_for_backward(a, b)
        return a + b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output, grad_output


class Sub(AutoGradFunction):
    @staticmethod
    def forward(ctx, a, b):
        check_device_compatibility(a, b)
        ctx.save_for_backward(a, b)
        return a - b
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output, -grad_output


class Mul(AutoGradFunction):
    @staticmethod
    def forward(ctx, a, b):
        check_device_compatibility(a, b)
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output * b, grad_output * a


class Div(AutoGradFunction):
    @staticmethod
    def forward(ctx, a, b):
        check_device_compatibility(a, b)
        ctx.save_for_backward(a, b)
        return a / b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output / b, -grad_output * a / (b ** 2)


class MatMul(AutoGradFunction):
    @staticmethod
    def forward(ctx, a, b):
        check_device_compatibility(a, b)
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        engine = get_array_module(a)
        grad_a = engine.matmul(grad_output, engine.swapaxes(b, -1, -2))
        grad_b = engine.matmul(engine.swapaxes(a, -1, -2), grad_output)
        return grad_a, grad_b


class Pow(AutoGradFunction):
    @staticmethod
    def forward(ctx, a, power):
        check_device_compatibility(a, power)
        ctx.save_for_backward(a, power)
        xp = get_array_module(a)
        return xp.power(a, power)

    @staticmethod
    def backward(ctx, grad_output):
        a, power = ctx.saved_tensors
        xp = get_array_module(a)
        grad_a = grad_output * (power * xp.power(a, power - 1))
        grad_power = grad_output * xp.power(a, power) * xp.log(a)
        return grad_a, grad_power


class Sum(AutoGradFunction):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        ctx.save_for_backward(a)
        ctx.axis = axis
        ctx.keepdims = keepdims
        xp = get_array_module(a)
        return xp.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        xp = get_array_module(a)
        grad = xp.ones_like(a) * grad_output
        return grad



class ReLU(AutoGradFunction):
    @staticmethod
    def forward(ctx, a):
        check_device_compatibility(a)
        mask = (a > 0).astype(a.dtype)
        ctx.save_for_backward(mask)
        return a * mask

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output * mask


class Exp(AutoGradFunction):
    @staticmethod
    def forward(ctx, a):
        check_device_compatibility(a)
        xp = get_array_module(a)
        out = xp.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        return grad_output * out


class Log(AutoGradFunction):
    @staticmethod
    def forward(ctx, a):
        check_device_compatibility(a)
        xp = get_array_module(a)
        ctx.save_for_backward(a)
        return xp.log(a)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        return grad_output / a
    

class LessThan(AutoGradFunction):
    @staticmethod
    def forward(ctx, a, b):
        check_device_compatibility(a, b)
        ctx.save_for_backward(a)
        return a < b

    @staticmethod
    def backward(ctx, grad_output):
        raise OpGradNonImplemented()
    

class GreaterThan(AutoGradFunction):
    @staticmethod
    def forward(ctx, a, b):
        check_device_compatibility(a, b)
        ctx.save_for_backward(a)
        return a > b

    @staticmethod
    def backward(ctx, grad_output):
        raise OpGradNonImplemented()


class LessThanEqual(AutoGradFunction):
    @staticmethod
    def forward(ctx, a, b):
        check_device_compatibility(a, b)
        ctx.save_for_backward(a)
        return a <= b

    @staticmethod
    def backward(ctx, grad_output):
        raise OpGradNonImplemented()
    

class GreaterThanEqual(AutoGradFunction):
    @staticmethod
    def forward(ctx, a, b):
        check_device_compatibility(a, b)
        ctx.save_for_backward(a)
        return a >= b

    @staticmethod
    def backward(ctx, grad_output):
        raise OpGradNonImplemented()


class EqualTo(AutoGradFunction):
    @staticmethod
    def forward(ctx, a, b):
        check_device_compatibility(a, b)
        ctx.save_for_backward(a)
        return a == b

    @staticmethod
    def backward(ctx, grad_output):
        raise OpGradNonImplemented()
    

class NotEqualTo(AutoGradFunction):
    @staticmethod
    def forward(ctx, a, b):
        check_device_compatibility(a, b)
        ctx.save_for_backward(a)
        return a != b

    @staticmethod
    def backward(ctx, grad_output):
        raise OpGradNonImplemented()