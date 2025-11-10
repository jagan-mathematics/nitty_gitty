import uuid
from .device import check_device_compatibility, get_array_module, is_cuda_available, Device, normalize_device, normalize_dtype
from .device import get_numpy_engine, get_cupy_engine
from .auto_grad import Context
from .utils import reduce_broadcast
from ..ops import basic_op

def apply_op(op_class, *tensors, **kwargs):
    ctx = Context()
    data = [t.data for t in tensors]
    device = check_device_compatibility(*data)
    
    result_data = op_class.forward(ctx, *data, **kwargs)
    requires_grad = any(t.requires_grad for t in tensors)
    result = Tensor(result_data, dtype=tensors[0].dtype, device=device, requires_grad=requires_grad)

    if requires_grad:
        result._op = op_class
        result._ctx = ctx
        result._children = list(tensors)
    return result


class Tensor:
    def __init__(self, data, dtype=None, device=Device.CPU, requires_grad=False, children=None):
        self.__device = normalize_device(device)
        engine = get_cupy_engine() if is_cuda_available() and self.__device == Device.GPU else get_numpy_engine()

        self.data = engine.array(data, dtype=dtype)
        self.grad = None
        if requires_grad:
            self.grad = engine.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._op = None
        self._ctx = None
        self._is_leaf = self._op is None
        self._children = children or []
        self.dtype = normalize_dtype(dtype)
        self.__object_id = str(uuid.uuid4())
    

    def zero_grad(self):
        self.grad = None
    
    @property
    def numel(self):
        return self.data.size
    
    @property
    def device(self):
        return self.__device
    
    def __hash__(self):
        return hash((self.__object_id))
    
    def to(self, device: str | Device):
        target_device = normalize_device(device)
        if target_device == self.device:
            return self

        # Use cupy functions for data transfer
        cp = get_cupy_engine()
        new_data = None
        
        if target_device == Device.GPU:
            # Moving from CPU to GPU
            new_data = cp.asarray(self.data)
        else:
            # Moving from GPU to CPU
            new_data = cp.asnumpy(self.data)
            
        return Tensor(new_data, dtype=self.dtype, device=target_device, requires_grad=self.requires_grad)
    
    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, dtype={self.dtype}, requires_grad={self.requires_grad})"

    def astype(self, dtype):
        normalized_new_dtype = normalize_dtype(dtype)
        if self.dtype == normalized_new_dtype:
            return self
        
        new_data = self.data.astype(normalized_new_dtype)
        
        return Tensor(new_data, dtype=normalized_new_dtype, device=self.device, requires_grad=self.requires_grad)

    def backward(self, grad_output=None):
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on a tensor that does not require gradients.")
        
        engine = get_array_module(self.data)
        if grad_output is None:
            if self.data.size != 1:
                raise RuntimeError("grad_output must be specified for non-scalar tensors.")
            grad_output = engine.ones_like(self.data)

        self.grad = grad_output
        
        topo = []
        visited = set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._children:
                    build_topo(child)
                topo.append(t)
        
        build_topo(self)

        for t in reversed(topo):
            if t._op:
                grads = t._op.backward(t._ctx, t.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                
                for child, g in zip(t._children, grads):
                    if g is not None and child.requires_grad:
                        grad_reduced = reduce_broadcast(g, child.data.shape)
                        if child.grad is None:
                            child.grad = grad_reduced
                        else:
                            child.grad += grad_reduced

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        new_data = self.data.T
        out = Tensor(new_data, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad, children=(self,))
        out._op = type('MatrixTransposeOp', (), {'ctx': None, 'backward': lambda ctx, grad_output: grad_output.T})()
        return out

    @property
    def ndim(self):
        return self.data.ndim


    def reshape(self, *shape):
        new_data = self.data.reshape(*shape)
        out = Tensor(new_data, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad, children=(self,))
        
        out._op = type('ReshapeOp', (), {'ctx': None, 'backward': lambda ctx, grad_output: grad_output.reshape(self.shape)})()
        return out


    def view(self, *shape):
        return self.reshape(*shape)


    def transpose(self, *axes):
        np = get_numpy_engine()

        new_data = self.data.transpose(*axes)
        out = Tensor(new_data, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad, children=(self,))

        inv_axes = np.argsort(axes) if axes else None
        
        op = type('TransposeOp', (), {})()
        op.ctx =  None

        def _backward(ctx, grad_output):
            return grad_output.transpose(inv_axes)
        
        op.backward = _backward
        out._op = op
        return out
    
    def permute(self, *dims):
        np = get_numpy_engine()
        new_data = self.data.transpose(dims)
        out = Tensor(new_data, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad, children=(self,))

        inv_dims = np.argsort(dims)
        if self.requires_grad:
            op = type('PermuteOp', (), {})()
            op.ctx = None
            op.backward = lambda ctx, grad_output: grad_output.transpose(inv_dims)
            out._op = op
        return out
    
    def flatten(self):
        new_data = self.data.reshape(-1)
        out = Tensor(new_data, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad, children=(self,))

        if self.requires_grad:
            op = type('FlattenOp', (), {})()
            op.ctx = None
            op.backward = lambda ctx, grad_output: grad_output.reshape(self.shape)
            out._op = op
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device, dtype=self.dtype)
        return apply_op(basic_op.Add, self, other)
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device, dtype=self.dtype)
        return apply_op(basic_op.Sub, self, other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device, dtype=self.dtype)
        return apply_op(basic_op.Mul, self, other)
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device, dtype=self.dtype)
        return apply_op(basic_op.Div, self, other)
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device, dtype=self.dtype)
        return apply_op(basic_op.MatMul, self, other)
    
    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device, dtype=self.dtype)
        return apply_op(basic_op.Pow, self, other)
    

    def __neg__(self):
        return -1 * self
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device, dtype=self.dtype)
        return self * other
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device, dtype=self.dtype)
        return other / self

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device, dtype=self.dtype)
        return other - self
    
    def __eq__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device, dtype=self.dtype)
        return apply_op(basic_op.EqualTo, self, other)
    
    def __ne__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device, dtype=self.dtype)
        return apply_op(basic_op.NotEqualTo, self, other)
    
    def __lt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device, dtype=self.dtype)
        return apply_op(basic_op.LessThan, self, other)
    

    def __gt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device, dtype=self.dtype)
        return apply_op(basic_op.GreaterThan, self, other)
    
    def __le__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device, dtype=self.dtype)
        return apply_op(basic_op.LessThanEqual, self, other)
    
    def __ge__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device, dtype=self.dtype)
        return apply_op(basic_op.GreaterThanEqual, self, other)
    
    def sum(self, axis=None, keepdims=False):
        return apply_op(basic_op.Sum, self, axis=axis, keepdims=keepdims)
    
    def relu(self):
        return apply_op(basic_op.ReLU, self)
    
    def exp(self):
        return apply_op(basic_op.Exp, self)
    
    def log(self):
        return apply_op(basic_op.Log, self)
    
