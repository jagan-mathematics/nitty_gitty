"""
A simplified educational re-implementation of **`nitty_gitty`** (inspired by Andrej Karpathy's *micrograd*).
https://github.com/karpathy/micrograd

This module extends the original idea to support tensors alongside scalar values,
providing a minimal, pure-Python implementation focused on conceptual clarity
rather than performance or optimization.

Note:
    The code is intentionally straightforward and non-optimized — it is designed
    purely for learning and experimentation purposes, illustrating the underlying
    mechanics in the simplest possible form.
"""

import numpy as np


class AutoGradFunction:
    def __init__(self, data, children=None, _op="", required_grad=False):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.required_grad = required_grad
        self._op = _op
        self.children = children if children is not None else set()
        self._backward = lambda: None
    
    
    def backward(self):
        if self.ndim !=0:
            raise RuntimeError("grad is not implicable to non scalar tensors")
    
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

        for v in topo:
            if not v.required_grad:
                v.grad = None
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim


    def __repr__(self):
        if self.required_grad:
            return f"Tensor(data={self.data}, required_grad={self.required_grad})"    

        return f"Tensor(data={self.data})"

    

class Tensor(AutoGradFunction):
    def __init__(self, data, name="", children=None, _op="", required_grad=False):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.required_grad = required_grad
        self.name = name
        self._op = _op
        self.children = children if children is not None else set()
        self._backward = lambda: None

    @staticmethod
    def __reduce_to_broadcast(data, target_shape):
            while data.ndim > len(target_shape):
                data = data.sum(axis=0)
            
            for i, dim in enumerate(target_shape):
                if dim == 1:
                    data = data.sum(axis=i, keepdims=True)
            return data

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        required_grad = self.required_grad or other.required_grad
        out = Tensor(self.data * other.data, children=(self, other), _op="*", required_grad=required_grad)

        def _backward():
            self.grad += Tensor.__reduce_to_broadcast(other.data * out.grad, self.data.shape)
            other.grad += Tensor.__reduce_to_broadcast(self.data * out.grad, other.data.shape)
        
        if required_grad:
            out._backward = _backward
        return out
    

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        required_grad = self.required_grad or other.required_grad
        out = Tensor(self.data + other.data, children=(self, other), _op="+", required_grad=required_grad)

        def _backward():
            self.grad += Tensor.__reduce_to_broadcast(out.grad, self.data.shape)
            other.grad += Tensor.__reduce_to_broadcast(out.grad, other.data.shape)
        
        if required_grad:
            out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        required_grad = self.required_grad or other.required_grad

        out = Tensor(self.data - other.data, children=(self, other), _op="-", required_grad=required_grad)

        def _backward():
            self.grad += Tensor.__reduce_to_broadcast(out.grad, self.data.shape)
            other.grad -= Tensor.__reduce_to_broadcast(out.grad, other.data.shape)

        if required_grad:
            out._backward = _backward
        return out
    
    def __neg__(self):
        out = Tensor(-self.data, children=(self,), _op="neg", required_grad=self.required_grad)
        
        def _backward():
            self.grad += Tensor.__reduce_to_broadcast(-1 * out.grad, self.data.shape)
        
        if self.required_grad:
            out._backward = _backward
        return out
    
    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        required_grad = self.required_grad or other.required_grad
        
        out = Tensor(-self.data ** other.data, children=(self,other), _op="**", required_grad=required_grad)
        
        def _backward():
            self.grad += Tensor.__reduce_to_broadcast((other.data * self.data**(other.data - 1)) * out.grad, self.data.shape)
            other.grad += Tensor.__reduce_to_broadcast((self.data ** other.data * np.log(self.data)) * out.grad, other.data.shape)
        
        if required_grad:
            out._backward = _backward
        return out
    
    def le(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data <= other.data, _op=">=", required_grad=False)

        def _backward():
            raise RuntimeError("grad_fn not implemented for this code")
        
        out._backward = _backward
        return out

    def ge(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data >= other.data, _op="<=", required_grad=False)

        def _backward():
            raise RuntimeError("grad_fn not implemented for this code")
        
        out._backward = _backward
        return out
    
    def eq(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data == other.data, _op="==", required_grad=False)

        def _backward():
            raise RuntimeError("grad_fn not implemented for this code")
        
        out._backward = _backward
        return out
    
    def lt(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        out = Tensor(self.data < other.data, _op= "<", required_grad=False)

        def _backward():
            raise RuntimeError("`grad_fn` not implemented for this code")
        
        out._backward = _backward
        return out
    
    def gt(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        out = Tensor(self.data > other.data, _op= ">", required_grad=False)

        def _backward():
            raise RuntimeError("`grad_fn` not implemented for this code")
        
        out._backward = _backward
        return out
    
    def ne(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        out = Tensor(self.data != other.data, _op= "!=", required_grad=False)

        def _backward():
            raise RuntimeError("`grad_fn` not implemented for this code")
        
        out._backward = _backward
        return out
    
    def __eq__(self, other):
        return self.eq(other)
    
    def __ne__(self, other):
        return self.ne(other)
    
    def __lt__(self, other):
        return self.lt(other)
    
    def __le__(self, other):
        return self.le(other)
    
    def __gt__(self, other):
        return self.gt(other)
    
    def __ge__(self, other):
        return self.ge(other)
    
    def __radd__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + other

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other -self

    def __rmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        required_grad = self.required_grad or other.required_grad

        out = Tensor(self.data / other.data, children=(self, other), _op="/", required_grad=required_grad)
        
        def _backward():
            self.grad += Tensor.__reduce_to_broadcast((1 / other.data) * out.grad, self.data.shape)
            other.grad += Tensor.__reduce_to_broadcast((-self.data / (other.data**2)) * out.grad, other.data.shape)

        if required_grad:
            out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self
    

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), children=(self,), _op="sum", required_grad=self.required_grad)

        def _backward():
            print(out.grad)
            self.grad += np.ones_like(self.data) * out.grad
        
        if self.required_grad:
            out._backward = _backward
        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        required_grad = self.required_grad or other.required_grad

        out = Tensor(self.data @ other.data, children=(self, other), _op="@", required_grad=required_grad)

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        
        if required_grad:
            out._backward = _backward
        return out
