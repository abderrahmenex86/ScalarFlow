# micrograd

A scalar autograd engine and the neural network library built on top of it, in
about a thousand lines of Python. It follows Andrej Karpathy's micrograd
lectures and exists to make reverse-mode autodiff readable end to end.

## What it does

- Tracks every operation on a `Value` node, then walks the resulting graph in
  topological order to compute exact gradients from `.backward()`.
- Covers the scalar operators (`+ - * / **`, unary negation) and the activations
  (`relu`, `tanh`, `exp`), each with its own local gradient rule.
- Builds `Neuron`, `Layer`, and `MLP` on top of `Value`, so a small network is
  the same graph as a single addition.
- Renders the graph with Graphviz, showing values, operations, and the gradients
  each node received.
- Ships 100 pytest functions covering operator edge cases, gradient
  accumulation, and network integration.

## Run it

```bash
git clone https://github.com/abderrahmenex86/micrograd
cd micrograd
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest tests/
```

Graphviz is only needed for the renderer, not for the engine or the tests, but
install it if you want the graph images.

## What is inside

```text
micrograd/engine/   value.py, neuron.py, layer.py, mlp.py
micrograd/utils/    graph.py (Graphviz renderer)
tests/              test_value.py, test_neuron.py, test_layer.py
```

## Limits

Scalar only. There are no tensors and no vectorised operations, so it is far too
slow for real training. That is the point: the engine exists to be read.

## Stack

Python 3.10+ · pytest · graphviz
