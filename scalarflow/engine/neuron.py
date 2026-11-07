import random

from scalarflow.engine.value import Value


class Neuron:

    def __init__(self, n_inputs):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, input):
        if len(input) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} inputs, got {len(input)}")

        activation = (
            sum(
                weight_ith * input_ith
                for weight_ith, input_ith in zip(self.weights, input)
            )
            + self.bias
        )
        out = activation.tanh()
        return out

    def parameters(self):
        return self.weights + [self.bias]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
