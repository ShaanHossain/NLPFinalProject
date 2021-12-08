# from os import XATTR_SIZE_MAX #Linux
# from os import PC_XATTR_SIZE_BITS #MacOS
import torch 
import torch.nn as nn
import scipy.io as sp
import sklearn.metrics
import numpy as np
import sys
import matplotlib.pyplot as plt

def plot_data(X, Y):

    data_x_class_pos = []
    data_y_class_pos = []
    data_x_class_neg = []
    data_y_class_neg = []

    for i in range(0, len(X)):
        if Y[i] == -1:
            data_x_class_pos.append(X[i][0]) 
            data_y_class_pos.append(X[i][1])
        else:
            data_x_class_neg.append(X[i][0]) 
            data_y_class_neg.append(X[i][1])

    plt.scatter(data_x_class_pos, data_y_class_pos, color='orange')
    plt.scatter(data_x_class_neg, data_y_class_neg, color='blue')
    plt.show()

def converge_to_binary(x):
    x = np.where(x < .5, -1, 1)
    return x

def converge_to_prob(x):
    x = np.where(x < 0., 0., 1.)
    return x

data = sp.loadmat("hw04_dataset.mat")

# print(data)

X = data["X_trn"]
Y = data["y_trn"]

# print(X)

X_test = data["X_tst"]
Y_test = data["y_tst"]

# Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h, n_out, hidden_layers, activation_function = 2, 50, 1, 2, 'relu'

x = torch.tensor(X).float()
y = torch.tensor(Y).float()

x_test = torch.tensor(X_test).float()
y_test = torch.tensor(Y_test).float()


class MyModule(nn.Module):
    def __init__(self, n_in, neurons_per_hidden, n_out, hidden_layers, activation_function):
        super(MyModule, self).__init__()

        self.n_in = n_in
        self.n_h = neurons_per_hidden
        self.n_out = n_out
        self.h_l = hidden_layers

        self.a_f = activation_function

        self.embeddings = []

        # Inputs to hidden layer linear transformation
        self.input = nn.Linear(n_in, self.n_h)

        # Embedding layer
        self.embedding_layer = nn.Embedding(num_embedding=len(self.embeddings), embedding_dim=n_in)
        # Load weights into embedding layer
        self.embedding_layer.load_state_dict({'weight': self.embeddings})
        # Do not train embedding layer
        self.weight.requires_grad = False

        #probably need a variable for recurrent dropout and dropout layer dropout

        # Defaults to Relu if activation_function is improperly sp
        self.activation_layer = nn.ReLU()

        if activation_function == 'relu':
            self.activation_layer = nn.ReLU()
        elif activation_function == 'tanh':
            self.activation_layer = nn.Tanh()
        elif activation_function == 'sigmoid':
            self.activation_layer == nn.Sigmoid()
        elif activation_function == 'identity':
            self.activation_layer == nn.Identity()
        elif activation_function == 'lstm':
            self.activation_layer == nn.LSTM(bidirectional=True, dropout=0.2)
        else:
            print("Invalid activation function specified")
            sys.exit(1)

        # self.linears = nn.ModuleList([nn.Linear(self.n_h, self.n_h) for i in range(self.h_l - 1)])
        self.activation_layers = nn.ModuleList([self.activation_layer for i in range(self.h_l - 1)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(0.5) for i in range(self.h_l - 1)])
        
        # Output layer, 10 units - one for each digit
        # self.output = nn.Linear(self.n_h, n_out)
        self.output == nn.ReLU()
        # Define sigmoid output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.input(x)
        x = self.activation_layer(x)

        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            # x = self.linears[i // 2](x) + l(x)
            x = self.activation_layers[i // 2](x) + l(x)
            x = self.dropout_layers[i // 2](x) + l(x)

        x = self.output(x)
        x = self.sigmoid(x)

        return x

model = nn.Sequential(MyModule(n_in, n_h, n_out, hidden_layers, activation_function))
print(model)

# Construct the loss function
criterion = torch.nn.BCELoss()
# Construct the optimizer (Adam in this case)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.05)

# Optimization
for epoch in range(50):
   # Forward pass: Compute predicted y by passing x to the model
   y_pred = model(x)

   y_pred_numpy = y_pred.detach().numpy()
   y_pred_tanh_range = converge_to_binary(y_pred_numpy)
   y_numpy = y.detach().numpy()
   y_sigmoid_range = converge_to_prob(y_numpy)

#    print(sklearn.metrics.accuracy_score(y_pred_tanh_range, y))

   # Compute and print loss
   loss = criterion(y_pred, torch.tensor(y_sigmoid_range).float())
#    print('epoch: ', epoch,' loss: ', loss.item())

   # Zero gradients, perform a backward pass, and update the weights.
   optimizer.zero_grad()

   # perform a backward pass (backpropagation)
   loss.backward()

   # Update the parameters
   optimizer.step()

# Plotting training data
# plot_data(X, Y)

y_pred_tanh_range_from_train = y_pred_tanh_range

y_pred = model(x_test)
y_pred_numpy = y_pred.detach().numpy()
y_pred_tanh_range = converge_to_binary(y_pred_numpy)

# print("Training Accuracy")
# print(sklearn.metrics.accuracy_score(y_pred_tanh_range_from_train, y))

# print("Testing Accuracy")
# print(sklearn.metrics.accuracy_score(y_pred_tanh_range, y_test))

# Plotting the test data
# plot_data(x_test, y_pred_tanh_range)

# Stuff I want to save:

## Check function
x_train_sample = ["Lorem Ipsum is simply dummy text of the printing and typesetting industry", "It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout"]
# X_train_Glove_s, word_index_s, embeddings_dict_s = convert_sentence_word_embeddings(x_train_sample)
# print("\n X_train_Glove_s \n ", X_train_Glove_s)
print("\n Word index of the word testing is : ", word_index_s["industry"])
print("\n Embedding for the word want \n \n", embeddings_dict_s["want"])
len(embeddings_dict_s["want"])

word_index_to_embedding = {}

# print(X_train_Glove_s[0])

# for i in embeddings_dict_s.keys():
#     print(i)
#     print(word_index_s[i])
#     word_index_to_embedding[word_index_s[i]] =  embeddings_dict_s[i]

# print(word_index_to_embedding[0])

# print(train_labels)

train_labels = list(map(lambda a: int(a), train_labels))

# print(train_labels)
    
# more stuff to save

def converge_to_binary(x):
    x = np.where(x < .5, -1, 1)
    return x

def converge_to_prob(x):
    x = np.where(x < 0., 0., 1.)
    return x

# Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h, n_out, hidden_layers, activation_function = 50, 10, 1, 1, 'lstm'

x = torch.tensor(X_train_Glove_s).float()
y = torch.tensor(train_labels).float()

class MyModule(nn.Module):
    def __init__(self, 
            n_in, 
            neurons_per_hidden, 
            n_out,
            hidden_layers, 
            activation_function,
            embeddings
        ):

        super(MyModule, self).__init__()

        self.n_in = n_in
        self.n_h = neurons_per_hidden
        self.n_out = n_out
        self.h_l = hidden_layers

        self.a_f = activation_function

        self.embeddings = embeddings

        # Inputs to hidden layer linear transformation
        self.input = nn.Linear(n_in, self.n_h)

        # Embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=len(self.embeddings), embedding_dim=n_in)
        # Load weights into embedding layer
        self.embedding_layer.load_state_dict({'weight': self.embeddings})
        # Do not train embedding layer
        self.weight.requires_grad = False

        #probably need a variable for recurrent dropout and dropout layer dropout

        # Defaults to Relu if activation_function is improperly sp
        self.activation_layer = nn.ReLU()

        if activation_function == 'relu':
            self.activation_layer = nn.ReLU()
        elif activation_function == 'tanh':
            self.activation_layer = nn.Tanh()
        elif activation_function == 'sigmoid':
            self.activation_layer == nn.Sigmoid()
        elif activation_function == 'identity':
            self.activation_layer == nn.Identity()
        elif activation_function == 'lstm':
            self.activation_layer == nn.LSTM(bidirectional=True, dropout=0.2)
        else:
            print("Invalid activation function specified")
            sys.exit(1)

        # self.linears = nn.ModuleList([nn.Linear(self.n_h, self.n_h) for i in range(self.h_l - 1)])
        self.activation_layers = nn.ModuleList([self.activation_layer for i in range(self.h_l - 1)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(0.5) for i in range(self.h_l - 1)])
        
        # Output layer, 10 units - one for each digit
        # self.output = nn.Linear(self.n_h, n_out)
        self.output == nn.ReLU()
        # Define sigmoid output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.input(x)
        x = self.activation_layer(x)

        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            # x = self.linears[i // 2](x) + l(x)
            x = self.activation_layers[i // 2](x) + l(x)
            x = self.dropout_layers[i // 2](x) + l(x)

        x = self.output(x)
        x = self.sigmoid(x)

        return x

embeddings = torch.tensor(embeddings_dict_s)
model = nn.Sequential(MyModule(n_in, n_h, n_out, hidden_layers, activation_function, embeddings))
print(model)

# Construct the loss function
criterion = torch.nn.BCELoss()
# Construct the optimizer (Adam in this case)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.05)

# Optimization
for epoch in range(50):
   # Forward pass: Compute predicted y by passing x to the model
   y_pred = model(x)

   y_pred_numpy = y_pred.detach().numpy()
   y_pred_tanh_range = converge_to_binary(y_pred_numpy)
   y_numpy = y.detach().numpy()
#    y_sigmoid_range = converge_to_prob(y_numpy)

#    print(sklearn.metrics.accuracy_score(y_pred_numpy, y))

   # Compute and print loss
   loss = criterion(y_pred, torch.tensor(y_numpy).float())
#    print('epoch: ', epoch,' loss: ', loss.item())

   # Zero gradients, perform a backward pass, and update the weights.
   optimizer.zero_grad()

   # perform a backward pass (backpropagation)
   loss.backward()

   # Update the parameters
   optimizer.step()

# Plotting training data
# plot_data(X, Y)

y_pred_tanh_range_from_train = y_pred_tanh_range

y_pred = model(x_test)
y_pred_numpy = y_pred.detach().numpy()
y_pred_tanh_range = converge_to_binary(y_pred_numpy)

# print("Training Accuracy")
# print(sklearn.metrics.accuracy_score(y_pred_tanh_range_from_train, y))

# print("Testing Accuracy")
# print(sklearn.metrics.accuracy_score(y_pred_tanh_range, y_test))

# Plotting the test data
# plot_data(x_test, y_pred_tanh_range)

