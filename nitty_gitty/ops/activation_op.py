from ..core.device import check_device_compatibility, get_array_module
from ..core.auto_grad import AutoGradFunction


class ReLUOp(AutoGradFunction):
    """
    Rectified Linear Unit (ReLU)

    Forward:
        y = x if x > 0 else 0

    Backward:
        dy/dx = 1 if x > 0 else 0
    """
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
    

class SigmoidOp(AutoGradFunction):
    """
    Sigmoid activation

    Forward:
        y = 1 / (1 + exp(-x))

    Backward:
        dy/dx = y * (1 - y)
    """
    @staticmethod
    def forward(ctx, a):
        check_device_compatibility(a)
        xp = get_array_module(a)
        out = 0.5 * (1.0 + xp.tanh(0.5 * a))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (activation_output,) = ctx.saved_tensors
        return grad_output * activation_output * (1 - activation_output)
    

class TanHOp(AutoGradFunction):
    """
    Hyperbolic Tangent (Tanh)

    Forward:
        y = tanh(x)

    Backward:
        dy/dx = 1 - y^2
    """
    @staticmethod
    def forward(ctx, a):
        check_device_compatibility(a)
        xp = get_array_module(a)
        out = xp.tanh(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (activation_output,) = ctx.saved_tensors
        return grad_output * (1.0 - (activation_output * activation_output))


class SoftSignOp(AutoGradFunction):
    """
    SoftSign activation

    Forward:
        y = x / (1 + |x|)

    Backward:
        dy/dx = 1 / (1 + |x|)^2
    """
    @staticmethod
    def forward(ctx, a):
        check_device_compatibility(a)
        xp = get_array_module(a)
        out = a / (1.0 + xp.abs(a))
        ctx.save_for_backward(a)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        |x| = sqrt(x^2)
        df/dx = d(|x|)/dx = d((x^2)^1/2)/dx 
                          = 1/2*((x^2)^-1/2).(2x)
                          = x/(x^2)^1/2
                          = x/|x|
        """
        (a,) = ctx.saved_tensors
        xp = get_array_module(a)
        denom = (1.0 + xp.abs(a)) ** 2
        return grad_output * (1.0 / denom)


class SoftPlusOp(AutoGradFunction):
    """
    SoftPlus activation

    Forward:
        y = log(1 + exp(x))

    Backward:
        dy/dx = 1 / (1 + exp(-x))
    """
    @staticmethod
    def forward(ctx, a):
        check_device_compatibility(a)
        xp = get_array_module(a)
        out = xp.log(1.0 + xp.exp(a))
        ctx.save_for_backward(a)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        xp = get_array_module(a)
        grad_a = 0.5 * (1.0 + xp.tanh(0.5 * a))
        return grad_output * grad_a
    

class LeakyReLUOp(AutoGradFunction):
    """
    Leaky ReLU

    Forward:
        y = x if x > 0 else alpha * x

    Backward:
        dy/dx = 1 if x > 0 else alpha
    """
    @staticmethod
    def forward(ctx, a, alpha=0.01):
        check_device_compatibility(a)
        xp = get_array_module(a)
        mask = a > 0
        out = xp.where(mask, a, alpha * a)
        ctx.save_for_backward(mask, xp.asarray(alpha, dtype=a.dtype))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (mask, alpha) = ctx.saved_tensors
        xp = get_array_module(mask)
        grad_a = xp.where(mask, 1.0, alpha)
        return grad_output * grad_a
    

class PReLUOp(AutoGradFunction):
    """
    Parametric ReLU

    Forward:
        y = x if x > 0 else a_param * x

    Backward:
        dy/dx = 1 if x > 0 else a_param
        dy/da_param = x if x <= 0 else 0
    """
    @staticmethod
    def forward(ctx, a, a_param):
        # a_param expected to be scalar or broadcastable
        check_device_compatibility(a)
        xp = get_array_module(a)

        if isinstance(a, (int, float)):
            raise ValueError(f"prelu: Expected `a` to be tensor, but got: {type(a)}")

        if isinstance(a_param, (int, float)):  # Allow scalar-like
            raise ValueError(f"prelu: Expected `weight` to be tensor or scalar, but got: {type(a_param)}")


        w_numel = 1
        if hasattr(a_param, 'size'):
            w_numel = a_param.size 
        elif hasattr(a_param, "numel"):
            w_numel = a_param.numel 


        if w_numel != 1:
            if a.ndim == 0:
                raise ValueError("Not allowed zero-dim input tensor when weight.numel() != 1.")
            channel_size = a.shape[1] if a.ndim >= 2 else 1
            if w_numel != channel_size:
                raise ValueError(
                    f"Mismatch of parameter numbers and input channel size. "
                    f"Found parameter numbers = {w_numel} and channel size = {channel_size}."
                )
            
        if a_param.ndim not in (0, 1):
            raise ValueError(
                f"prelu: Expected `weight` to be a scalar or 1D tensor, but got: ndim = {a_param.ndim}"
            )
            
        if a.ndim == 0:
            a_param_broadcast = a_param.flatten()[0] if a_param.ndim == 1 else a_param
        else:
            if a_param.ndim == 0:
                a_param_broadcast = a_param
            else:
                if a.ndim == 1:
                    a_param_broadcast = xp.tile(a_param, a.shape[0])
                else:
                    axes_to_insert = (0,) + tuple(range(2, a.ndim))
                    w_base = xp.expand_dims(a_param, axis=axes_to_insert)
                    target_shape = (a.shape[0],) + w_base.shape[1:]
                    a_param_broadcast = xp.broadcast_to(w_base, target_shape)
                    
        mask = a > 0.0
        out = xp.where(mask, a, a_param_broadcast * a)
        ctx.save_for_backward(a, mask, a_param_broadcast)
        ctx.register_arg("num_parameters", w_numel)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (a, mask, a_param_broadcast) = ctx.saved_tensors
        xp = get_array_module(mask)
        num_param = ctx.get_arg("num_parameters") or a_param_broadcast.shape[0]
        grad_a = xp.where(mask, 1.0, a_param_broadcast) * grad_output
        
        contrib = grad_output * xp.where(mask, 0.0, a)
        n_dim = a.ndim
        if num_param == 1:
            total_grad = xp.sum(contrib)
            grad_param = xp.full(a_param_broadcast.shape, total_grad)
        else:
            if n_dim < 2:
                grad_param = contrib
            else:
                axes = tuple(i for i in range(n_dim) if i != 1)
                grad_param = xp.sum(contrib, axis=axes)
        return grad_a, grad_param


class RReluOp(AutoGradFunction):
    """
    Randomized Leaky ReLU

    Forward:
        y = x if x > 0 else r_value * x

    Backward:
        dy/dx = 1 if x > 0 else r_value
    """
    @staticmethod
    def forward(ctx, a, lower, upper, training):
        check_device_compatibility(a)
        xp = get_array_module(a)
        mask = a > 0
        if training:
            r_value = xp.random.uniform(lower, upper, a.shape)
        else:
            r_value = lower + upper / 2.0

        out = xp.where(mask, a, r_value * a)
        ctx.save_for_backward(a, mask, r_value)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, mask, r_value = ctx.saved_tensors
        xp = get_array_module(grad_output)
        grad = xp.where(mask, 1.0, r_value)
        return grad_output * grad
    

class ELUOp(AutoGradFunction):
    """
    Exponential Linear Unit

    Forward:
        y = x if x > 0 else alpha * (exp(x) - 1)

    Backward:
        dy/dx = 1 if x > 0 else alpha * exp(x)
    """
    @staticmethod
    def forward(ctx, a, alpha=1.0):
        check_device_compatibility(a)
        xp = get_array_module(a)
        
        mask = a > 0
        e_a = xp.exp(a)
        out = xp.where(mask, a, alpha * (e_a - 1.0))
        ctx.save_for_backward(mask, e_a, xp.asarray(alpha, dtype=a.dtype))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        mask, e_a, alpha = ctx.saved_tensors
        xp = get_array_module(mask)
        grad_input = xp.where(mask, 1.0, e_a * alpha)
        return grad_output * grad_input
    

class SELUOp(AutoGradFunction):
    """
    Scaled Exponential Linear Unit

    Forward:
        y = scale * x if x > 0 else scale * alpha * (exp(x) - 1)

    Backward:
        dy/dx = scale if x > 0 else scale * alpha * exp(x)
    """
    @staticmethod
    def forward(ctx, a, alpha = 1.6732632423543772, scale = 1.0507009873554805):
        check_device_compatibility(a)
        xp = get_array_module(a)
        
        mask = a > 0
        e_a = xp.exp(a)
        out = xp.where(mask, scale * a, scale * alpha * (e_a - 1.0))
        ctx.save_for_backward(mask, e_a, xp.asarray(scale, dtype=a.dtype), xp.asarray(alpha, dtype=a.dtype))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        mask, e_a, scale, alpha = ctx.saved_tensors
        xp = get_array_module(mask)
        grad_input = xp.where(mask, scale, scale * e_a * alpha)
        return grad_output * grad_input
    

class GELUOp(AutoGradFunction):
    """
    Gaussian Error Linear Unit (approximation)

    Forward:
        y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Backward:
        dy/dx = 0.5 * (1 + tanh(...)) + 0.5 * x * sech^2(...) * sqrt(2/pi) * (1 + 3*0.044715*x^2)
    """
    @staticmethod
    def forward(ctx, a):
        check_device_compatibility(a)
        xp = get_array_module(a)
        c = xp.sqrt(2.0 / xp.pi)
        inner = c * (a + 0.044715 * xp.power(a, 3))
        tanh_inner = xp.tanh(inner)
        out = 0.5 * a * ( 1.0 + tanh_inner)
        ctx.save_for_backward(a, tanh_inner)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, tanh_inner = ctx.saved_tensors
        xp = get_array_module(a)

        c = xp.sqrt(2.0 / xp.pi)

        sech2 = 1.0 - tanh_inner * tanh_inner
        term1 = 0.5 * (1.0 + tanh_inner)
        deriv_inner = 1.0 + 3.0 * 0.044715 * xp.power(a, 2)
        term2 = 0.5 * a * (sech2 * c * deriv_inner)
        grad = term1 + term2
        return grad_output * grad
    

class SiLUOp(AutoGradFunction): #SiLU / Swish
    """
    Sigmoid Linear Unit (Swish)

    Forward:
        y = x * sigmoid(x)

    Backward:
        dy/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    """
    @staticmethod
    def forward(ctx, a):
        check_device_compatibility(a)
        xp = get_array_module(a)
        sig = 1.0 / (1.0 + xp.exp(-a))
        out = a * sig
        ctx.save_for_backward(a, sig)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, sig = ctx.saved_tensors
        grad = sig + a * sig * (1.0 - sig)
        return grad_output * grad
    

class MishOp(AutoGradFunction):
    """
    Mish activation

    Forward:
        y = x * tanh(softplus(x)), where softplus(x) = log(1 + exp(x))

    Backward:
        dy/dx = tanh(softplus(x)) + x * sigmoid(x) * (1 - tanh(softplus(x))^2)
    """
    @staticmethod
    def forward(ctx, a):
        check_device_compatibility(a)
        xp = get_array_module(a)
        softplus = xp.log1p(xp.exp(a))
        tanh_softplus = xp.tanh(softplus)
        out = a * tanh_softplus
        ctx.save_for_backward(a, tanh_softplus)
        return out


    @staticmethod
    def backward(ctx, grad_output):
        a, tanh_softplus = ctx.saved_tensors
        xp = get_array_module(a)
        sigma = 1.0 / (1.0 + xp.exp(-a))
        grad = tanh_softplus + a * sigma * (1.0 - tanh_softplus * tanh_softplus)
        return grad_output * grad
    

class SoftMaxOp(AutoGradFunction):
    """
    SoftMax activation

    Forward:
        y_i = exp(x_i) / sum(exp(x_j)) for all j

    Backward:
        dy/dx = y * (grad_output - sum(grad_output * y))
    """
    @staticmethod
    def forward(ctx, a, axis= -1):
        check_device_compatibility(a)
        xp = get_array_module(a)
        shift = a - xp.max(a, axis=axis, keepdims=True)
        exp_a = xp.exp(shift)
        exp_sum = xp.sum(exp_a, axis=axis, keepdims=True)
        out = exp_a / exp_sum
        ctx.save_for_backward(out)
        ctx.axis = axis
        return out


    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        axis = ctx.axis
        xp = get_array_module(out)
        dot = xp.sum(grad_output * out, axis=axis, keepdims=True)
        return out * (grad_output - dot)
    


class GLUOp(AutoGradFunction):
    """
    Gated Linear Unit

    Forward:
        y = a * sigmoid(b)

    Backward:
        dy/da = sigmoid(b)
        dy/db = a * sigmoid(b) * (1 - sigmoid(b))
    """
    @staticmethod
    def forward(ctx, a, b):
        check_device_compatibility(a)
        xp = get_array_module(a)
        sig = 1.0 / (1.0 + xp.exp(-b))
        out = a * sig
        ctx.save_for_backward(a, sig)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, sig = ctx.saved_tensors
        grad_a = grad_output * sig
        grad_b = grad_output * a * sig * (1.0 - sig)
        return grad_a, grad_b
    

class SwiGLUOp(AutoGradFunction):
    """
    SwiGLU activation

    Forward:
        y = a * (b * sigmoid(b))

    Backward:
        dy/da = b * sigmoid(b)
        dy/db = a * (sigmoid(b) + b * sigmoid(b) * (1 - sigmoid(b)))
    """
    @staticmethod
    def forward(ctx, a, b):
        check_device_compatibility(a)
        xp = get_array_module(a)
        sig = 1.0 / (1.0 + xp.exp(-b))
        sw = b * sig
        out = a * sw
        ctx.save_for_backward(a, b, sig, sw)
        return out


    @staticmethod
    def backward(ctx, grad_output):
        a, b, sig, sw = ctx.saved_tensors
        dsw_db = sig + b * sig * (1.0 - sig)
        grad_a = grad_output * sw
        grad_b = grad_output * a * dsw_db
        return grad_a, grad_b