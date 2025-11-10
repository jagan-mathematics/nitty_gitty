from ..core.tensor import apply_op
from ..ops import activation_op

def relu(a):
    return apply_op(activation_op.ReLU, a)


def sigmoid(a):
    return apply_op(activation_op.Sigmoid, a)

def tanh(a):
    return apply_op(activation_op.Tanh, a)


def soft_sign(a):
    return apply_op(activation_op.SoftSign, a)

def soft_plus(a):
    return apply_op(activation_op.SoftPlus, a)

def leaky_relu(a, alpha: float = 0.01):
    return apply_op(activation_op.LeakyReLU, a, alpha=alpha)

def prelu(a, alpha):
    return apply_op(activation_op.PReLU, a, alpha)

def rrelu(a, r_value):
    return apply_op(activation_op.RRelu, a, r_value=r_value)
    
def elu(a, alpha=1.0):
    return apply_op(activation_op.ELU, a, alpha=alpha)

def selu(a, alpha = 1.6732632423543772, scale = 1.0507009873554805):
    return apply_op(activation_op.SELU, a, alpha=alpha, scale=scale)

def gelu(a):
    return apply_op(activation_op.GELU, a)

def silu(a):
    return apply_op(activation_op.SiLU, a)

def mish(a):
    return apply_op(activation_op.Mish, a)

def softmax(a):
    return apply_op(activation_op.SoftMax, a)

def gilu(a, b):
    return apply_op(activation_op.GELU, a, b)

def swiglu(a, b):
    return apply_op(activation_op.SwiGLU, a, b)
