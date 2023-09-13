import numpy as np


class DotProject:
    def __init__(self, input_vector_1, weight) -> None:
        assert len(input_vector_1) == len(weight), "all lists must be the same size for dot multiplication"

        self.input_vector_1 = input_vector_1
        self.weight = weight

    def getIndexMulti(self, index):
        return self.input_vector_1[index] * self.weight[index]

    def getDotProduct(self):
        dot = 0
        for n in range(len(self.input_vector_1)):
            dot += self.getIndexMulti(n)
        return dot



def sigmoid(x):
    # np.exp(x) is e^x
    return 1 / (1 + np.exp(-x))

def make_prediction(dotter:DotProject, bias):
    layer_1 = dotter.getDotProduct() + bias
    # layer_2 = sigmoid(layer_1)
    return layer_1



input_vector_1 =    [0.0, 0.2]
weight =            [0.1, 0.1]
bias =              0.0         # ignore for now



dp_accuracy = 40

error = 1 #ignore
while round(error, dp_accuracy) != 0:
    prediction = make_prediction(dotter=DotProject(input_vector_1, weight), bias=bias) # weight (x) + bias


    target = 0
    x = (prediction-target)

    equation = error = x**20
    x_derived = gradient = 2*x

    print(f"Prediction: {f'{x_derived};':<27} Target: {f'{x_derived};':<27} Derivative: {f'{x_derived};':<27} ERROR: {f'{round(error, dp_accuracy)} ({dp_accuracy}dp);':<27}")

    weight[0] = weight[0] - gradient
    weight[1] = weight[1] - gradient

print(weight)