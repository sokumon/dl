# NeuralNetworkforNANDfunction
import numpy as np


class nand_NN:
    def __init__(self, iv):
        self.input_vector = np.array(iv)
        self.weight_vector = np.array([[1.5, -1, -1]])
        self.output = np.array([])

    def compute(self):
        mfv = np.c_[np.ones(4), self.input_vector]  # paddingwithbias
        result = mfv.dot(self.weight_vector.transpose())  # multipy
        self.output = [0 if r < 0 else 1 for r in result]  # simplefilter

    def show_result(self):
        print("InputVector:\n", self.input_vector)
        print("Output:\n", self.output)

    def get_output(self):
        return self.output


n = nand_NN(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))

n.compute()
n.show_result()
