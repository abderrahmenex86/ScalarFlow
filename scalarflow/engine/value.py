import math


class Value:

    def __init__(self, value, children=(), op="", label=""):
        self.data = value
        self.grad = 0.0
        self._backward = lambda: None
        self.label = label
        self._prev = set(children)
        self.op = op

    def __repr__(self):
        return f"Value(data={self.data}, label={self.label}, grad={self.grad})"

    def __add__(self, other):
        assert isinstance(
            other, (Value, int, float)
        ), "Can only add Value or numeric types"
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        assert isinstance(other, (int, float)), "Can only add numeric types"
        return self + Value(other)

    def __neg__(self):
        return self * Value(-1)

    def __sub__(self, other):
        assert isinstance(
            other, (Value, int, float)
        ), "Can only subtract Value or numeric types"
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __rsub__(self, other):
        assert isinstance(other, (int, float)), "Can only subtract numeric types"
        return other + (-self)

    def __mul__(self, other):
        assert isinstance(
            other, (Value, int, float)
        ), "Can only multiply Value or numeric types"
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        assert isinstance(other, (int, float)), "Can only multiply numeric types"
        return self * Value(other)

    def __truediv__(self, other):
        assert isinstance(
            other, (Value, int, float)
        ), "Can only divide by Value or numeric types"
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), "/")

        def _backward():
            self.grad += (1 / other.data) * out.grad
            other.grad -= (self.data / (other.data**2)) * out.grad

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        assert isinstance(other, (int, float)), "Can only divide numeric types"
        return Value(other) / self

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "Only supports int/float powers"
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += (exponent * self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        # t = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        t = math.tanh(x)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topological = []
        visited = set()

        def build_topological_order(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topological_order(child)
                topological.append(node)

        build_topological_order(self)

        self.grad = 1.0
        for node in reversed(topological):
            node._backward()
