import random

import pytest

from scalarflow.engine.neuron import Neuron
from scalarflow.engine.value import Value


class TestNeuronCreation:
    def test_init_basic(self):
        n_inputs = 3
        neuron = Neuron(n_inputs)

        assert len(neuron.weights) == n_inputs
        assert all(isinstance(w, Value) for w in neuron.weights)
        assert isinstance(neuron.bias, Value)

        for weight in neuron.weights:
            assert -1 <= weight.data <= 1
        assert -1 <= neuron.bias.data <= 1

    def test_init_different_sizes(self):
        for n_inputs in [1, 2, 5, 10]:
            neuron = Neuron(n_inputs)
            assert len(neuron.weights) == n_inputs

    def test_init_zero_inputs(self):
        neuron = Neuron(0)
        assert len(neuron.weights) == 0
        assert isinstance(neuron.bias, Value)

    def test_weights_are_different(self):
        random.seed(42)
        neuron1 = Neuron(5)
        random.seed(43)
        neuron2 = Neuron(5)

        weight_values1 = [w.data for w in neuron1.weights]
        weight_values2 = [w.data for w in neuron2.weights]
        assert weight_values1 != weight_values2


class TestNeuronForwardPass:
    def test_call_basic(self):
        neuron = Neuron(2)
        neuron.weights[0].data = 0.5
        neuron.weights[1].data = -0.3
        neuron.bias.data = 0.1

        inputs = [Value(2.0), Value(1.0)]
        output = neuron(inputs)

        expected_activation = 0.5 * 2.0 + (-0.3) * 1.0 + 0.1
        expected_output = Value(expected_activation).tanh()

        assert isinstance(output, Value)
        assert abs(output.data - expected_output.data) < 1e-10

    def test_call_with_value_inputs(self):
        neuron = Neuron(3)
        neuron.weights[0].data = 1.0
        neuron.weights[1].data = 0.5
        neuron.weights[2].data = -0.5
        neuron.bias.data = 0.0

        inputs = [Value(1.0), Value(2.0), Value(-1.0)]
        output = neuron(inputs)

        assert isinstance(output, Value)
        assert -1 <= output.data <= 1

    def test_call_with_numeric_inputs(self):
        neuron = Neuron(2)
        neuron.weights[0].data = 0.5
        neuron.weights[1].data = -0.5
        neuron.bias.data = 0.0

        inputs = [1.0, 2.0]
        output = neuron(inputs)

        assert isinstance(output, Value)
        assert -1 <= output.data <= 1

    def test_call_single_input(self):
        neuron = Neuron(1)
        neuron.weights[0].data = 2.0
        neuron.bias.data = 1.0

        inputs = [Value(0.5)]
        output = neuron(inputs)

        assert isinstance(output, Value)
        assert -1 <= output.data <= 1

    def test_call_zero_inputs(self):
        neuron = Neuron(0)
        neuron.bias.data = 0.5

        inputs = []
        output = neuron(inputs)

        assert isinstance(output, Value)
        assert -1 <= output.data <= 1

    def test_call_negative_inputs(self):
        neuron = Neuron(2)
        neuron.weights[0].data = 1.0
        neuron.weights[1].data = 1.0
        neuron.bias.data = 0.0

        inputs = [Value(-1.0), Value(-2.0)]
        output = neuron(inputs)

        assert isinstance(output, Value)
        assert -1 <= output.data <= 1


class TestNeuronBackwardPass:
    def test_backward_simple(self):
        neuron = Neuron(2)
        neuron.weights[0].data = 0.5
        neuron.weights[1].data = -0.3
        neuron.bias.data = 0.1

        inputs = [Value(2.0), Value(1.0)]
        output = neuron(inputs)
        output.backward()

        assert neuron.weights[0].grad != 0.0
        assert neuron.weights[1].grad != 0.0
        assert neuron.bias.grad != 0.0
        assert inputs[0].grad != 0.0
        assert inputs[1].grad != 0.0

    def test_backward_chain(self):
        neuron = Neuron(2)
        neuron.weights[0].data = 1.0
        neuron.weights[1].data = 1.0
        neuron.bias.data = 0.0

        inputs = [Value(1.0), Value(2.0)]
        output = neuron(inputs)

        final_output = output * Value(2.0)
        final_output.backward()

        assert neuron.weights[0].grad != 0.0
        assert neuron.weights[1].grad != 0.0
        assert neuron.bias.grad != 0.0
        assert inputs[0].grad != 0.0
        assert inputs[1].grad != 0.0

    def test_gradient_accumulation(self):
        neuron = Neuron(1)
        neuron.weights[0].data = 1.0
        neuron.bias.data = 0.0

        input_val = Value(1.0)

        output1 = neuron([input_val])
        output2 = neuron([input_val])
        total_output = output1 + output2
        total_output.backward()

        assert neuron.weights[0].grad != 0.0
        assert neuron.bias.grad != 0.0
        assert input_val.grad != 0.0


class TestNeuronEdgeCases:
    def test_large_inputs(self):
        neuron = Neuron(2)
        neuron.weights[0].data = 0.1
        neuron.weights[1].data = 0.1
        neuron.bias.data = 0.0

        inputs = [Value(100.0), Value(200.0)]
        output = neuron(inputs)

        assert isinstance(output, Value)
        assert -1 <= output.data <= 1
        assert output.data > 0.9

    def test_zero_weights_and_bias(self):
        neuron = Neuron(3)
        for weight in neuron.weights:
            weight.data = 0.0
        neuron.bias.data = 0.0

        inputs = [Value(1.0), Value(2.0), Value(3.0)]
        output = neuron(inputs)

        assert abs(output.data - 0.0) < 1e-10

    def test_wrong_input_size(self):
        neuron = Neuron(3)

        inputs_short = [Value(1.0), Value(2.0)]
        with pytest.raises(ValueError):
            neuron(inputs_short)

    def test_very_small_inputs(self):
        neuron = Neuron(2)
        neuron.weights[0].data = 1.0
        neuron.weights[1].data = 1.0
        neuron.bias.data = 0.0

        inputs = [Value(1e-10), Value(1e-10)]
        output = neuron(inputs)

        assert abs(output.data - 0.0) < 1e-5


class TestNeuronParameters:
    def test_parameter_access(self):
        neuron = Neuron(3)

        assert hasattr(neuron, "weights")
        assert hasattr(neuron, "bias")
        assert len(neuron.weights) == 3
        assert isinstance(neuron.bias, Value)

    def test_parameter_modification(self):
        neuron = Neuron(2)

        neuron.weights[0].data = 0.5
        neuron.weights[1].data = -0.5
        neuron.bias.data = 0.1

        assert neuron.weights[0].data == 0.5
        assert neuron.weights[1].data == -0.5
        assert neuron.bias.data == 0.1

    def test_parameters_are_values(self):
        neuron = Neuron(5)

        for weight in neuron.weights:
            assert isinstance(weight, Value)
        assert isinstance(neuron.bias, Value)

    def test_parameter_gradients(self):
        neuron = Neuron(2)

        inputs = [Value(1.0), Value(2.0)]
        output = neuron(inputs)
        output.backward()

        for weight in neuron.weights:
            assert hasattr(weight, "grad")
            assert isinstance(weight.grad, (int, float))
        assert hasattr(neuron.bias, "grad")
        assert isinstance(neuron.bias.grad, (int, float))


class TestNeuronReproducibility:
    def test_deterministic_with_fixed_weights(self):
        neuron = Neuron(2)
        neuron.weights[0].data = 0.5
        neuron.weights[1].data = -0.3
        neuron.bias.data = 0.1

        inputs = [Value(1.0), Value(2.0)]

        output1 = neuron(inputs)
        output2 = neuron(inputs)

        assert output1.data == output2.data

    def test_different_random_seeds(self):
        random.seed(42)
        neuron1 = Neuron(3)

        random.seed(24)
        neuron2 = Neuron(3)

        weights1 = [w.data for w in neuron1.weights]
        weights2 = [w.data for w in neuron2.weights]
        assert weights1 != weights2
        assert neuron1.bias.data != neuron2.bias.data
