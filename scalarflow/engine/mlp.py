from scalarflow.engine.layer import Layer


class MLP:

    def __init__(self, n_inputs, n_outputs):
        assert (
            isinstance(n_inputs, int) and n_inputs > 0
        ), "n_inputs must be a positive integer"
        assert (
            isinstance(n_outputs, list) and len(n_outputs) > 0
        ), "n_outputs must be a non-empty list of integers"
        sizes = [n_inputs] + n_outputs
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def parameters(self):
        return [params for layer in self.layers for params in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
