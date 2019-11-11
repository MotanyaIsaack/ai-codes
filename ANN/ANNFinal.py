#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[125]:


#x= (input1, input2, input3) y= (expected output)
x = np.array([[30, 40, 50],[40, 50, 20],[50, 20, 15],[20, 15, 60],[15, 60, 70],[60, 70, 50]],dtype=float)
y = np.array([20, 15, 60, 70, 50, 40],dtype=float)
 #Scale units to a number less than 1
def bind_function(input, smallest, largest):
    return (input - smallest) / (largest - smallest)
def scale_values(values):
    
    min_value = values[np.unravel_index(np.argmin(values, axis=None), values.shape)]
    max_value = values[np.unravel_index(np.argmax(values, axis=None), values.shape)]
    new_values = np.array([bind_function(i, min_value, max_value)
        for i in np.nditer(values)], dtype=np.float64).reshape(values.shape)
    np.put(values, range(values.size), new_values)
        
scale_values(x) #Scale units
scale_values(y) #Scale units


# In[138]:


class NeuralNetwork:
    def __init__(self, x, y,learn_rate):
        #Outer layer Weight matrix(2*3) weight of matrix from input to hidden
        self.w1 = [[0.2, 0.3, 0.2], [0.1, 0.1, 0.1]]
        #Hidden layer weight matrix(1*2)weight of matrix from hiddent to output
        self.w2 = [[0.5, 0.1]]
        self.learn_rate = learn_rate

    def feedfoward(self):
          #forward propagation through the network
        output = []
        for occur in x:
            occur = np.array(occur, ndmin=2).T 
            l1 = np.dot(self.w1, occur)# dot product of x(input) and first set of weights
            outerlayer=self.sigmoid(l1)#Activation function and
            l2 = np.dot(self.w2, outerlayer) #Dot product of hidden layer (layer1) and second set of weights
            output=self.sigmoid(l2)# Output
        return output
    
    def sigmoid(self, s, deriv=False):
        if (deriv == True):
            return s*(1-s)
        return 1/(1+np.exp(-s))
    def backward(self, input, output_target):
        #Backward propagate through the network
        output_target = self.feedfoward()
        input = np.array(input, ndmin=2).T#Value of the input
        hiddenoutput = self.sigmoid(np.dot(self.w1, input))#Output of the hidden layer
        networkoutput = self.sigmoid(np.dot(self.w2, hiddenoutput))#Actual Output of the network
        output_errors = output_target - networkoutput#Error in output ie expected output- actual output
        wu = output_errors * networkoutput *             (1.0 - networkoutput)
        wu = self.learn_rate * np.dot(wu, hiddenoutput.T)
        self.w2 += wu # update the weights:Adjusting second set (hidden -> output) weights
        hidden_errors = np.dot(self.w2.T, output_errors)# calculate hidden errors:
        wu = hidden_errors * hiddenoutput *             (1.0 - hiddenoutput)
        self.w1 += self.learn_rate *             np.dot(wu, input.T) # update the weights:Adjusting first set (input-> hidden) weights
        print("Input: \n", input)
        print("Expected Output: ", output_target)
        print("Actual Output: ", networkoutput)
        print("Error :", output_errors, "\n")
    def train(self):#Training the model
        for i in range(len(x)):#For loop to loop through inputs
            NN.backward(x[i], y[i])# call function for back propagation
    def weight_matrix(self):
        print("Outer layer Weight matrix: ", self.w1)#final trained Outer layer Weight matrix 
        print("Hidden layer weight matrix: ", self.w2)#final trained Hidden layer weight matrix


# In[139]:


NN = NeuralNetwork(x, y, 0.5)#input, expected output, learning rate
NN.train()#Training the model
NN.weight_matrix()


# In[ ]:





# In[ ]:




