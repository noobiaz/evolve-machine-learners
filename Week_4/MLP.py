import numpy as np 
#Input array 
X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]]) 
#Output 
y=np.array([[1],[1],[0]])

#Sigmoid Function 
def sigmoid (x): return 1/(1 + np.exp(-x)) 
#Derivative of Sigmoid Function 
def derivatives_sigmoid(x): return x * (1 - x)

#Variable initialization 
epoch=5000 #Setting training iterations 
lr=0.1 #Setting learning rate 
inputlayer_neurons = X.shape[1] #number of features in data set 
hiddenlayer_neurons = 3 #number of hidden layers neurons 
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization 
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons)) 
bh=np.random.uniform(size=(1,hiddenlayer_neurons)) 
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons)) 
bout=np.random.uniform(size=(1,output_neurons))

#Training 
for i in range(epoch): 
    #forward_propagation 
    hidden_layer_input1=np.dot(X,wh) 
    hidden_layer_input=hidden_layer_input1 + bh 
    hiddenlayer_activations = sigmoid(hidden_layer_input) 
    output_layer_input1=np.dot(hiddenlayer_activations,wout) 
    output_layer_input= output_layer_input1 + bout 
    output = sigmoid(output_layer_input) 
    #backward_propagation 
    E = y-output 
    slope_output_layer = derivatives_sigmoid(output) 
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations) 
    d_output = E * slope_output_layer 
    Error_at_hidden_layer = d_output.dot(wout.T) 
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer 
    wout += hiddenlayer_activations.T.dot(d_output) *lr 
    bout += np.sum(d_output, axis=0,keepdims=True) *lr 
    wh += X.T.dot(d_hiddenlayer) *lr 
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
    
print(output)

from random import seed 
from random import random 
# Initialize a network 
def initialize_network(n_inputs, n_hidden, n_outputs): 
    network = list() 
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)] 
    network.append(hidden_layer) 
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)] 
    network.append(output_layer) 
    return network

seed(1) 
network = initialize_network(2, 1, 2) 
for layer in network: 
    print(layer)

# Calculate neuron activation for an input 
def activate(weights, inputs): 
    activation = weights[-1] 
    for i in range(len(weights)-1): 
        activation += weights[i] * inputs[i] 
    return activation
    
# activation function 
def sigmoid(activation): 
    return 1.0 / (1.0 + exp(-activation)) 
def tanh(activation): 
    return (2*sigmoid(activation)-1.0) 

# Forward propagate input to a network output 
def forward_propagate(network, row): 
    inputs = row 
    for layer in network: new_inputs = [] 
    for neuron in layer: 
        activation = activate(neuron['weights'], inputs) 
        neuron['output'] = transfer(activation) 
        new_inputs.append(neuron['output']) 
        inputs = new_inputs 
    return inputs

# test forward propagation 
network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}], [{'weights': [0.2550690257394217, 0.49543508709194095]}, 
             {'weights': [0.4494910647887381, 0.651592972722763]}]] 
row = [1, 0, None] 
output = forward_propagate(network, row) 
print(output)

# Calculate the derivative of sigmoid 
def sigmoid_derivative(output): 
    return output * (1.0 - output)

# Backpropagate error and store in neurons 
def backward_propagate_error(network, expected): 
    for i in reversed(range(len(network))): 
        layer = network[i] 
        errors = list() 
        if i != len(network)-1: 
            for j in range(len(layer)): 
                error = 0.0 
                for neuron in network[i + 1]: 
                    error += (neuron['weights'][j] * neuron['delta']) 
                    errors.append(error)
        else: 
            for j in range(len(layer)): 
                neuron = layer[j] 
                errors.append(expected[j] - neuron['output']) 
            for j in range(len(layer)): 
                neuron = layer[j] 
                neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])

# test backpropagation of error 
network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}], [{'output': 0.6213859615555266, 
           'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]] 
expected = [0, 1] 
backward_propagate_error(network, expected) 
for layer in network: 
    print(layer)
    
# Update network weights with error 
def update_weights(network, row, l_rate): 
    for i in range(len(network)): inputs = row[:-1] 
    if i != 0: 
        inputs = [neuron['output'] for neuron in network[i - 1]] 
    for neuron in network[i]: 
        for j in range(len(inputs)): 
            neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j] 
        neuron['weights'][-1] += l_rate * neuron['delta']
        
        
# Train a network for a fixed number of epochs 
def train_network(network, train, l_rate, n_epoch, n_outputs): 
    for epoch in range(n_epoch): 
        sum_error = 0 
        for row in train: 
            outputs = forward_propagate(network, row) 
            expected = [0 for i in range(n_outputs)] 
            expected[row[-1]] = 1 
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))]) 
            backward_propagate_error(network, expected) 
            update_weights(network, row, l_rate) 
        print('>epoch=%d, lrate=%.3f, error=%.3f'%(epoch, l_rate, sum_error))
        
# Test training backprop algorithm 
seed(1) 
dataset = [[2.7810836,2.550537003,0], [1.465489372,2.362125076,0], [3.396561688,4.400293529,0], [1.38807019,1.850220317,0], [3.06407232,3.005305973,0], [7.627531214,2.759262235,1], 
           [5.332441248,2.088626775,1], [6.922596716,1.77106367,1], [8.675418651,-0.242068655,1], [7.673756466,3.508563011,1]] 
n_inputs = len(dataset[0]) - 1 
n_outputs = len(set([row[-1] for row in dataset])) 
network = initialize_network(n_inputs, 2, n_outputs) 
train_network(network, dataset, 0.5, 20, n_outputs) 
for layer in network: 
    print(layer)
    
# Make a prediction with a network 
def predict(network, row): 
    outputs = forward_propagate(network, row) 
    return outputs.index(max(outputs))

# Test making predictions with the network 
dataset = [[2.7810836,2.550537003,0], [1.465489372,2.362125076,0], [3.396561688,4.400293529,0], [1.38807019,1.850220317,0], [3.06407232,3.005305973,0], [7.627531214,2.759262235,1], 
           [5.332441248,2.088626775,1], [6.922596716,1.77106367,1], [8.675418651,-0.242068655,1], [7.673756466,3.508563011,1]] 
network = [[{'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]}, {'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]}], 
           [{'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]}, {'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]}]] 
for row in dataset: 
    prediction = predict(network, row) 
    print('Expected=%d, Got=%d' % (row[-1], prediction))