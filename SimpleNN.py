import numpy as np
#import pandas as pd
import matplotlib.pylab as plt

# Simple neural network classify MNIST dataset
class SimpleNN:
    def __init__(self, layers_size,input_layer_node):
        self.layers_size = layers_size
        self.parameters = {}
        self.L = len(self.layers_size)
        self.n = 0
        self.batch = 0
        self.costs = []
        self.layers_size.insert(0,input_layer_node)
        self.initialize_parameters()
 
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)
 
    def softmax(self, Z):
        # Prevent extreme large Z cause unstable 
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)
 
    def initialize_parameters(self):
        np.random.seed(1)
 
        for l in range(1, len(self.layers_size)):
            self.parameters["W" + str(l)] = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / np.sqrt(
                self.layers_size[l - 1])
            self.parameters["b" + str(l)] = np.zeros((self.layers_size[l], 1))
 
    def forward(self, X):
        cache = {}
 
        A = X.T
        for l in range(self.L - 1):
            Z = self.parameters["W" + str(l + 1)].dot(A) + self.parameters["b" + str(l + 1)]
            A = self.sigmoid(Z)
            cache["A" + str(l + 1)] = A
            cache["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
            cache["Z" + str(l + 1)] = Z
 
        Z = self.parameters["W" + str(self.L)].dot(A) + self.parameters["b" + str(self.L)]
        A = self.softmax(Z)
        cache["A" + str(self.L)] = A
        cache["W" + str(self.L)] = self.parameters["W" + str(self.L)]
        cache["Z" + str(self.L)] = Z
 
        return A, cache
 
    def backward(self, X, Y, cache):
 
        derivatives = {}
 
        cache["A0"] = X.T
 
        A = cache["A" + str(self.L)]
        dZ = A - Y.T
 
        dW = dZ.dot(cache["A" + str(self.L - 1)].T) / self.batch
        db = np.sum(dZ, axis=1, keepdims=True) / self.batch
        dAPrev = cache["W" + str(self.L)].T.dot(dZ)
 
        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db
 
        for l in range(self.L - 1, 0, -1):
            dZ = dAPrev * self.sigmoid_derivative(cache["Z" + str(l)])
            dW = 1. / self.batch * dZ.dot(cache["A" + str(l - 1)].T)
            db = 1. / self.batch * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = cache["W" + str(l)].T.dot(dZ)
 
            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db
 
        return derivatives


    def create_mini_batches(self,X, y, batch_size):
        
        mini_batches = []
        data = np.hstack((X, y))
        np.random.shuffle(data)
        n_minibatches = data.shape[0] // batch_size
        i = 0
     
        for i in range(n_minibatches + 1):
            mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
            X_mini = mini_batch[:, :-10]
            Y_mini = mini_batch[:, -10:].reshape((-1, 10))
            mini_batches.append((X_mini, Y_mini))
        if data.shape[0] % batch_size != 0:
            mini_batch = data[i * batch_size:data.shape[0]]
            X_mini = mini_batch[:, :-10]
            Y_mini = mini_batch[:, -10:].reshape((-1, 10))
            mini_batches.append((X_mini, Y_mini))
        return mini_batches
 
    def fit(self, X, Y, learning_rate=1, n_iterations=10,batch=32):
        np.random.seed(1)
        self.batch = batch

        for loop in range(n_iterations):

            mini_batches = self.create_mini_batches(X, Y, self.batch)
            loss = 0
            acc = 0
            for mini_batch in mini_batches:
                X_mini, y_mini = mini_batch
                A, store = self.forward(X_mini)
                loss += -1*np.mean(y_mini * np.log(A.T+ 1e-8))# CCE cost function A.T is updated weight 
                derivatives = self.backward(X_mini, y_mini, store)
     
                for l in range(1, self.L + 1):
                    self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * derivatives[
                        "dW" + str(l)]
                    self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * derivatives[
                        "db" + str(l)]

                acc += self.predict(X_mini, y_mini)

            
            self.costs.append(loss)
            print("Epoch",loop+1,"\steps ",len(mini_batches),"Train loss: ", "{:.4f}".format(loss/len(mini_batches)),
                                                "Train acc:", "{:.4f}".format(acc/len(mini_batches)))
                    
                
                    

    def save_weights(self):
        np.save("weights.npy",SimpleNN.parameters ,allow_pickle=True)
    
    def load_weights(self,dir):
        
        weights=np.load(dir,allow_pickle=True)
        for i in SimpleNN.parameters.keys():
            SimpleNN.parameters[str(i)] = weights.item().get(str(i))
            
        print('Weight loaded')
        
    def predict(self, X, Y):
        A, _ = self.forward(X)
        y_hat = np.argmax(A, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (y_hat == Y).mean()
        return accuracy
 
    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.grid()
        plt.show()
