from scalarflow.engine.neuron import Neuron


class Layer:

    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, inputs):
        outs = [neuron(inputs) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
