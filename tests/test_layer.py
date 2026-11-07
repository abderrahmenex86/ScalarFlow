import random

import pytest

from scalarflow.engine.layer import Layer
from scalarflow.engine.neuron import Neuron
from scalarflow.engine.value import Value


class TestLayerCreation:
    def test_init_basic(self):
        n_inputs = 3
        n_outputs = 2
        layer = Layer(n_inputs, n_outputs)

        assert len(layer.neurons) == n_outputs
        assert all(isinstance(neuron, Neuron) for neuron in layer.neurons)

        for neuron in layer.neurons:
            assert len(neuron.weights) == n_inputs

    def test_init_different_sizes(self):
        test_cases = [(1, 1), (2, 3), (5, 1), (10, 5)]

        for n_inputs, n_outputs in test_cases:
            layer = Layer(n_inputs, n_outputs)
            assert len(layer.neurons) == n_outputs
            for neuron in layer.neurons:
                assert len(neuron.weights) == n_inputs

    def test_init_zero_outputs(self):
        layer = Layer(3, 0)
        assert len(layer.neurons) == 0

    def test_init_zero_inputs(self):
        layer = Layer(0, 2)
        assert len(layer.neurons) == 2
        for neuron in layer.neurons:
            assert len(neuron.weights) == 0

    def test_neurons_are_different(self):
        layer = Layer(3, 2)

        assert layer.neurons[0] is not layer.neurons[1]

        weights1 = [w.data for w in layer.neurons[0].weights]
        weights2 = [w.data for w in layer.neurons[1].weights]
        assert weights1 != weights2


class TestLayerForwardPass:
    def test_call_basic(self):
        layer = Layer(2, 3)

        for i, neuron in enumerate(layer.neurons):
            neuron.weights[0].data = 0.5 * (i + 1)
            neuron.weights[1].data = -0.3 * (i + 1)
            neuron.bias.data = 0.1 * (i + 1)

        inputs = [Value(2.0), Value(1.0)]
        outputs = layer(inputs)

        assert len(outputs) == 3
        assert all(isinstance(output, Value) for output in outputs)

        for output in outputs:
            assert -1 <= output.data <= 1

    def test_call_with_value_inputs(self):
        layer = Layer(3, 2)

        inputs = [Value(1.0), Value(2.0), Value(-1.0)]
        outputs = layer(inputs)

        assert len(outputs) == 2
        assert all(isinstance(output, Value) for output in outputs)
        for output in outputs:
            assert -1 <= output.data <= 1

    def test_call_with_numeric_inputs(self):
        layer = Layer(2, 2)

        inputs = [1.0, 2.0]
        outputs = layer(inputs)

        assert len(outputs) == 2
        assert all(isinstance(output, Value) for output in outputs)
        for output in outputs:
            assert -1 <= output.data <= 1

    def test_call_single_input_single_output(self):
        layer = Layer(1, 1)
        layer.neurons[0].weights[0].data = 1.0
        layer.neurons[0].bias.data = 0.0

        inputs = [Value(0.5)]
        outputs = layer(inputs)

        assert len([outputs]) == 1
        assert isinstance(outputs, Value)
        assert -1 <= outputs.data <= 1

    def test_call_empty_inputs(self):
        layer = Layer(0, 2)

        inputs = []
        outputs = layer(inputs)

        assert len(outputs) == 2
        assert all(isinstance(output, Value) for output in outputs)
        for output in outputs:
            assert -1 <= output.data <= 1

    def test_call_large_layer(self):
        layer = Layer(10, 5)

        inputs = [Value(random.uniform(-1, 1)) for _ in range(10)]
        outputs = layer(inputs)

        assert len(outputs) == 5
        assert all(isinstance(output, Value) for output in outputs)
        for output in outputs:
            assert -1 <= output.data <= 1


class TestLayerBackwardPass:
    def test_backward_simple(self):
        layer = Layer(2, 2)

        # Set known weights
        for neuron in layer.neurons:
            neuron.weights[0].data = 0.5
            neuron.weights[1].data = -0.3
            neuron.bias.data = 0.1

        inputs = [Value(2.0), Value(1.0)]
        outputs = layer(inputs)

        total_output = sum(outputs)
        total_output.backward()

        for neuron in layer.neurons:
            assert neuron.weights[0].grad != 0.0
            assert neuron.weights[1].grad != 0.0
            assert neuron.bias.grad != 0.0

        for input_val in inputs:
            assert input_val.grad != 0.0

    def test_backward_single_output(self):
        layer = Layer(3, 1)

        inputs = [Value(1.0), Value(2.0), Value(-1.0)]
        outputs = layer(inputs)

        outputs.backward()

        neuron = layer.neurons[0]
        for weight in neuron.weights:
            assert weight.grad != 0.0
        assert neuron.bias.grad != 0.0

        for input_val in inputs:
            assert input_val.grad != 0.0

    def test_backward_multiple_outputs(self):
        layer = Layer(2, 3)

        inputs = [Value(1.0), Value(2.0)]
        outputs = layer(inputs)

        loss = outputs[0] * outputs[1] + outputs[2]
        loss.backward()

        for neuron in layer.neurons:
            for weight in neuron.weights:
                assert weight.grad != 0.0
            assert neuron.bias.grad != 0.0

    def test_backward_chain(self):
        layer = Layer(2, 2)

        inputs = [Value(1.0), Value(2.0)]
        outputs = layer(inputs)

        intermediate = [output * Value(2.0) for output in outputs]
        final_output = sum(intermediate)
        final_output.backward()

        for neuron in layer.neurons:
            for weight in neuron.weights:
                assert weight.grad != 0.0
            assert neuron.bias.grad != 0.0


class TestLayerEdgeCases:
    def test_zero_outputs(self):
        layer = Layer(3, 0)

        inputs = [Value(1.0), Value(2.0), Value(3.0)]
        outputs = layer(inputs)

        assert len(outputs) == 0
        assert outputs == []

    def test_zero_inputs(self):
        layer = Layer(0, 2)

        inputs = []
        outputs = layer(inputs)

        assert len(outputs) == 2
        for output in outputs:
            assert -1 <= output.data <= 1

    def test_large_inputs(self):
        layer = Layer(2, 2)

        inputs = [Value(100.0), Value(200.0)]
        outputs = layer(inputs)

        assert len(outputs) == 2
        for output in outputs:
            assert -1 <= output.data <= 1

    def test_small_inputs(self):
        layer = Layer(2, 2)

        inputs = [Value(1e-10), Value(1e-10)]
        outputs = layer(inputs)

        assert len(outputs) == 2
        for output in outputs:
            assert -1 <= output.data <= 1


class TestLayerParameters:
    def test_parameter_access(self):
        layer = Layer(3, 2)

        assert hasattr(layer, "neurons")
        assert len(layer.neurons) == 2

        for neuron in layer.neurons:
            assert hasattr(neuron, "weights")
            assert hasattr(neuron, "bias")
            assert len(neuron.weights) == 3

    def test_parameter_modification(self):
        layer = Layer(2, 2)

        layer.neurons[0].weights[0].data = 0.5
        layer.neurons[0].weights[1].data = -0.5
        layer.neurons[0].bias.data = 0.1

        assert layer.neurons[0].weights[0].data == 0.5
        assert layer.neurons[0].weights[1].data == -0.5
        assert layer.neurons[0].bias.data == 0.1

    def test_all_parameters_are_values(self):
        layer = Layer(3, 2)

        for neuron in layer.neurons:
            for weight in neuron.weights:
                assert isinstance(weight, Value)
            assert isinstance(neuron.bias, Value)

    def test_parameter_gradients(self):
        layer = Layer(2, 2)

        inputs = [Value(1.0), Value(2.0)]
        outputs = layer(inputs)
        sum(outputs).backward()

        for neuron in layer.neurons:
            for weight in neuron.weights:
                assert hasattr(weight, "grad")
                assert isinstance(weight.grad, (int, float))
            assert hasattr(neuron.bias, "grad")
            assert isinstance(neuron.bias.grad, (int, float))


class TestLayerReproducibility:
    def test_deterministic_with_fixed_weights(self):
        layer = Layer(2, 2)

        for i, neuron in enumerate(layer.neurons):
            neuron.weights[0].data = 0.5 * (i + 1)
            neuron.weights[1].data = -0.3 * (i + 1)
            neuron.bias.data = 0.1 * (i + 1)

        inputs = [Value(1.0), Value(2.0)]

        outputs1 = layer(inputs)
        outputs2 = layer(inputs)

        assert len(outputs1) == len(outputs2)
        for out1, out2 in zip(outputs1, outputs2):
            assert out1.data == out2.data

    def test_different_random_initialization(self):
        random.seed(42)
        layer1 = Layer(3, 2)

        random.seed(24)
        layer2 = Layer(3, 2)

        for neuron1, neuron2 in zip(layer1.neurons, layer2.neurons):
            weights1 = [w.data for w in neuron1.weights]
            weights2 = [w.data for w in neuron2.weights]
            assert weights1 != weights2
            assert neuron1.bias.data != neuron2.bias.data


class TestLayerIntegration:
    def test_layer_chaining(self):
        layer1 = Layer(3, 2)
        layer2 = Layer(2, 1)

        inputs = [Value(1.0), Value(2.0), Value(-1.0)]

        intermediate = layer1(inputs)
        outputs = layer2(intermediate)

        assert isinstance(outputs, Value)
        assert -1 <= outputs.data <= 1

    def test_layer_in_network(self):
        layer1 = Layer(2, 3)
        layer2 = Layer(3, 1)

        inputs = [Value(1.0), Value(2.0)]

        hidden = layer1(inputs)
        output = layer2(hidden)

        output.backward()

        for neuron in layer1.neurons:
            for weight in neuron.weights:
                assert weight.grad != 0.0

        for neuron in layer2.neurons:
            for weight in neuron.weights:
                assert weight.grad != 0.0
