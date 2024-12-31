#!/usr/bin/env python3

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

inputs = np.array([0.35, 0.7])

y_true = 0.5

weights = np.array([
    [0.2, 0.2],
    [0.3, 0.3]
])

weights2 = np.array([0.3, 0.9])

def forward(inputs, weights):
    return np.dot(inputs, weights.T)

epochs = 8000

for _ in range(epochs):
    layer1 = forward(inputs, weights)

    for i in range(len(layer1)):
        layer1[i] = sigmoid(layer1[i])

    output2 = forward(layer1, weights2)
    output2 = sigmoid(output2)

    L = y_true - output2

    delta3 = L * sigmoid_derivative(output2)
    delta2 = layer1[1] * (1 - layer1[1]) * weights2[1] * delta3
    delta1 = layer1[0] * (1 - layer1[0]) * weights2[0] * delta3

    delta_w_1_3 = delta3 * layer1[0]
    delta_w_2_3 = delta3 * layer1[1]

    weights2[0] += delta_w_1_3
    weights2[1] += delta_w_2_3

    delta_w_1_1 = delta1 * inputs[0]
    delta_w_1_2 = delta2 * inputs[0]
    delta_w_2_1 = delta1 * inputs[1]
    delta_w_2_2 = delta2 * inputs[1]

    weights[0][0] += delta_w_1_1
    weights[0][1] += delta_w_2_1
    weights[1][0] += delta_w_1_2
    weights[1][1] += delta_w_2_2

    print(f"Loss: {L}\nOutput: {output2}")
