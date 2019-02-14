import numpy as np
import random

class Perceptron(object):
    def __init__(self, input, learning_rate=1, iterations=50):
        w_list = []
        for i in range(input):
            w_list.append(random.uniform(0, 1)) # initializing random values for the weights
        w_list.append(1) # adding the bias node equal to 1

        self.weights = np.array(w_list) # adding the values to a numpy array for simplicity later when performing vector operations
        self.iterations = iterations
        self.input = input
        self.learning_rate = learning_rate

    def activation_function(self, x):
        if x >= 0:
            return 1
        else:
            return 0

    def predict(self, x):
        weighted_sum = self.weights.T.dot(x) # taking the dot product of the weights vector
        activation = self.activation_function(weighted_sum)
        return activation

    def fit(self, gate, correct_output):
        for i in range(self.iterations):
            for j in range(correct_output.shape[0]):
                x = np.insert(gate[j], self.input, 1) # input vector with a bias value
                y = self.predict(x)
                # print statements for comparing the network guess to the actual output
                # print("perceptron guess: " + str(y))
                # print("correct answer: " + str(correct_output[j]), "\n")
                error = correct_output[j] - y # backpropagation - adjusting weights after comparing prediction to actual output
                self.weights = self.weights + self.learning_rate * error * x
                print(self.weights)

and_gate = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
and_result = np.array([0, 0, 0, 1])

or_gate = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
or_result = np.array([0, 1, 1, 1])

perceptron1 = Perceptron(input=2)
print("Results over 100 iterations:", "\n")

print("***** AND *****")
print("STARTING 'AND' WEIGHTS: ", perceptron1.weights)
perceptron1.fit(and_gate, and_result)
print("FINAL 'AND' WEIGHTS: ", perceptron1.weights, "\n")

print("***** OR *****")
perceptron2 = Perceptron(input=2)
print("STARTING 'OR' WEIGHTS: ", perceptron2.weights)
perceptron2.fit(or_gate, or_result)
print("FINAL 'OR' WEIGHTS: ", perceptron2.weights)

