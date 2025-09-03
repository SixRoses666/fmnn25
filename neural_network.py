from __future__ import annotations
from typing import Callable
import math

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

class NeuralNetwork:
    
    def __init__(self, w_layers: list[int]):

        self.layers: list[Layer] = []
        for w in w_layers:
            previous_layer = self.layers[-1] if self.layers else []
            self.layers.append(Layer(w, previous_layer))

        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

    def compute(self, input):
        self.set_input_layer(input)
        for layer in self.layers:
            self.compute_layer(layer)
        # Do something with the output layer

    def set_input_layer(self, input):
        for data, neuron in zip(input, self.input_layer):
            neuron.output = data

    def compute_layer(self, layer: list[Neuron]):
        for neuron in layer:
            neuron.compute()

class Neuron:

    def __init__(self):
        self.incoming: list[Edge] = []
        self.outgoing: list[Edge] = []
        self.output = None

    def compute(self):
        sum = 0
        for edge in self.incoming:
            sum += edge.weight * edge.parent.output
        self.output = sigmoid(sum)
        
Layer = list[Neuron]

class Edge:

    def __init__(self, parent: Neuron, child: Neuron):
        self.parent = parent
        self.child = child
        self.weight = None
        self.value = None

class Layer:

    def __init__(self, w: int, previous_layer: Layer | None):
        self.neurons = [Neuron() for _ in range(w)]
        if previous_layer:
            for parent in previous_layer.neurons:
                for neuron in self.neurons:
                    edge = Edge(parent, neuron)
                    parent.outgoing.append(edge)
                    neuron.incoming.append(edge)
                    