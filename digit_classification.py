import numpy as np
import random
from data_loader import load_data_wrapper
class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]] #creating 1 random bias to start with for each neuron
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]#creating random weights to start to connect neurons from adjacent layers
    def get_output(self,a):#given a specific input, a, for a network, returns the output
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a
    def gradient_descent(self, training_data, epochs, batch_size, eta, test=None):#mini batch stochastic gradient descent
        if test:#if test data is provided, network is evaluated against test data after each epoch
            test_num = len(test)
        n = len(training_data)#training data is a list of tuples which represents training inputs/desired outputs
        for i in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[j:j+batch_size]for j in range(0,n,batch_size)]
            for batch in batches:
                self.update_mini_batch(batch,eta)
            if test:
                print(f"Epoch {i}: {self.evaluate(test)} / {test_num}")
            else:
                print(f"Epoch {i} complete")
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]
    def backprop(self, x, y):
        #more complicated back propagation algorithm to help adjust weights accordingly
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    def evaluate(self, test_data):
        #tells us the number of test inputs which the neural networks correctly assumes
        test_results = [(np.argmax(self.get_output(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

def sigmoid(y):#applies sigmoid function to inputs
    return 1/(np.exp(-y)+1)
def sigmoid_prime(y):
    return sigmoid(y) * (1-sigmoid(y))

if __name__ == "__main__":
    training_data, validation_data, test_data = load_data_wrapper()
    net = Network([784, 30, 10])  # input: 784 pixels, 1 hidden layer of 30, output: 10 digits
    net.gradient_descent(training_data, epochs=10, batch_size=10, eta=3.0, test=test_data)