from ..core.tensor import apply_op
from ..ops.basic_op import ReLU as ReLUOp

def relu(tensor):
    return apply_op(ReLUOp, tensor)