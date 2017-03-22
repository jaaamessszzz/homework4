import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
import sys
import copy
import pprint

class neural_network():
    def __init__(self, input_layer_nodes=8, hidden_layer_nodes=3, output_layer_nodes=8):
        self.input_layer_nodes = input_layer_nodes
        self.hidden_layer_nodes = hidden_layer_nodes
        self.output_layer_nodes = output_layer_nodes

        # Initialize matrices with random weights 0.1 > x > -0.1
        # Set bottom row to all ones
        self.connection_matrix_1 = np.random.randn(self.input_layer_nodes, self.hidden_layer_nodes) / 10
        self.connection_matrix_2 = np.random.randn(self.hidden_layer_nodes, self.output_layer_nodes) / 10

        # Input/output vectors
        self.input_vector = None
        self.hidden_layer_output = None
        self.output_layer_output = None

        # Biased Vectors and Matrices
        self.input_with_bias = None
        self.matrix_1_bias = None
        self.matrix_2_bias = None

        # Derivative matrices
        self.hidden_dx_matrix = np.zeros([self.hidden_layer_nodes, self.hidden_layer_nodes])
        self.output_dx_matrix = np.zeros([self.output_layer_nodes, self.output_layer_nodes])

        # Learning Rate
        self.learning_rate = 1

        # Bit conversion for DNA into NN inputs
        self.base_binary_conversion = {'A': '0001',
                                       'C': '0010',
                                       'T': '0100',
                                       'G': '1000'
                                       }

        # Expected Values
        self.expected_values = None

    def initialize_values(self):
        bias_ones_1 = np.ones([1, self.hidden_layer_nodes])
        self.matrix_1_bias = np.append(self.connection_matrix_1, bias_ones_1, axis=0)

        bias_ones_2 = np.ones([1, self.output_layer_nodes])
        self.matrix_2_bias = np.append(self.connection_matrix_2, bias_ones_2, axis=0)

    def set_input_and_expected_values(self, input_DNA, autoencoder=True, negative=True):
        # Convert input DNA sequence into binary bits
        self._construct_input_vector(input_DNA)
        # Set expected value depending on autoencoder or werk
        self._set_expected_values(autoencoder, negative)
        # Weight matrices and input/output vectors with bias applied
        self.input_with_bias = np.append(self.input_vector, [1])

    def _construct_input_vector(self, input_DNA):
        """
        Convert input DNA string into a vector of 1's and 0's
        """
        temp_vector_list = []

        for base in input_DNA:
            for number in self.base_binary_conversion[base]:
                temp_vector_list.append(float(number))

        self.input_vector = np.asarray(temp_vector_list)

    def _set_expected_values(self, autoencoder=True, negative=True):
        if autoencoder == True:
            self.expected_values = self.input_vector

        if autoencoder == False:
            if negative == True:
                self.expected_values = 0
            if negative == False:
                self.expected_values = 1

    def forward_propogation(self):
        # Generate hidden layer outputs
        output_one_list = []

        for element in np.nditer(np.dot(self.input_with_bias, self.matrix_1_bias)):
            output_one_list.append(self._sigmoid_function(element))
        self.hidden_layer_output = np.asarray(output_one_list)

        # calculate square derivate matrix for hidden layer outputs
        for position, value in enumerate(self.hidden_layer_output):
            self.hidden_dx_matrix[position][position] = self._sigmoid_function_derivative(value)

        # Output Layer output
        # Add bias to hidden_layer_output
        self.hidden_output_bias = np.append(self.hidden_layer_output, [1])

        output_two_list = []
        for element in np.nditer(np.dot(self.hidden_output_bias, self.matrix_2_bias)):
            output_two_list.append(self._sigmoid_function(element))
        self.output_layer_output = np.asarray(output_two_list)

        # calculate square derivate matrix for output layer outputs
        for position, value in enumerate(self.output_layer_output):
            self.output_dx_matrix[position][position] = self._sigmoid_function_derivative(value)

    def backward_propogation(self):
        # Output Layer error
        deviations = self.output_layer_output - self.expected_values

        output_layer_errors = np.dot(self.output_dx_matrix, deviations)

        # Hidden Layer error
        hidden_layer_errors = np.dot(np.dot(self.hidden_dx_matrix, self.connection_matrix_2), output_layer_errors)

        # Matrix 2 Errors
        output_rated_row_vector = np.asmatrix(-(self.learning_rate * output_layer_errors)).transpose()
        matrix_2_errors_transposed = np.dot(output_rated_row_vector, np.asmatrix(self.hidden_output_bias))
        self.matrix_2_errors = matrix_2_errors_transposed.transpose()

        # Matrix 1 Errors
        hidden_rated_row_vector = np.asmatrix(-(self.learning_rate * hidden_layer_errors)).transpose()
        matrix_1_errors_transposed = np.dot(hidden_rated_row_vector, np.asmatrix(self.input_with_bias))
        self.matrix_1_errors = matrix_1_errors_transposed.transpose()

    def update_weights_and_bias(self):
        self.matrix_1_bias = self.matrix_1_bias + self.matrix_1_errors
        self.matrix_2_bias = self.matrix_2_bias + self.matrix_2_errors

        self.connection_matrix_1 = self.matrix_1_bias[:-1]
        self.connection_matrix_2 = self.matrix_2_bias[:-1]

    def _sigmoid_function(self, input):
        return float(1 / (1 + np.exp(-input)))

    def _sigmoid_function_derivative(self, input):
        return float(self._sigmoid_function(input) * (1 - self._sigmoid_function(input)))


