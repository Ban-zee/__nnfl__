#Loading necessary packages and defining global variables
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from collections import defaultdict
import os
from torch.utils import data
import matplotlib.pyplot as plt
import pandas as pd
from sys import platform
import timeit
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cpu")
OUTPUT_DIR_TRAIN='data/train.dat'
OUTPUT_DIR_TEST='data/test1.dat'
NUM_RESTS = 5138
NUM_USERS = 3579


def get_sparse_mat(filename):
    '''

    Inputs:
        -filename: a string containing the name of the file from which we want
                    to extract the data. In our case it can be either train.dat
                    or test.dat

    Returns a python list of size 3579 (number of users) with each element of
    the list being a list of tuples (restaurantID, rating).

    '''

    data = pd.read_table(filename, sep=',', names=['index', '1', '2'], )
    sparse_mat = []
    for i in range(0, NUM_USERS):
        collect = data.loc[data['index'] == i]
        collect = collect.drop(['index'], axis=1)
        a = collect['1'].values.tolist()
        b = collect['2'].values.tolist()
        c = []
        for j in range(0, len(a)):
            c.append([a[j], b[j]])
        sparse_mat.append(c)


    return sparse_mat


s = timeit.timeit()
train_smat = get_sparse_mat(OUTPUT_DIR_TRAIN)
print(s-timeit.timeit())
test_smat = get_sparse_mat(OUTPUT_DIR_TEST)

print("Running Sample Test Case 1")
assert np.allclose(len(train_smat), 3579)
print("Sample Test Case 1 Passed")
print("Running Sample Test Case 2")
assert np.allclose(len(test_smat), 3579)
print("Sample Test Case 2 Passed")
print("Running Sample Test Case 3")
assert np.allclose(len(train_smat[5]), 234)
print("Sample Test Case 3 Passed")
print("Running Sample Test Case 4")
assert np.allclose(train_smat[5][:5],[(626, 4.0), (1177, 4.5), (976, 4.0), (3926, 4.0), (3274, 5.0)])
print("Sample Test Case 4 Passed")
print("Running Sample Test Case 5")
assert np.allclose(len(test_smat[5]),5)
print("Sample Test Case 5 Passed")
print("Running Sample Test Case 6")
assert np.allclose(test_smat[5][:5], [(574, 3.5), (3717, 4.0), (2303, 4.0), (863, 3.5), (1706, 1.0)])
print("Sample Test Case 6 Passed")
print("Running Sample Test Case 7")
assert ((type(train_smat[5][:5][0][0]) is int) and (type(train_smat[5][:5][0][1]) is float))
print("Sample Test Case 7 Passed")
print("Running Sample Test Case 8")
assert ((type(test_smat[5][:5][0][0]) is int) and (type(test_smat[5][:5][0][1]) is float))
print("Sample Test Case 8 Passed")


class Dataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X_sam = torch.zeros(5138)
        y_sam = torch.zeros(5138)
        for i in range(len(self.X[index])):
            X_sam[self.X[index][i][0]] = self.X[index][i][1]

        for i in range(len(self.y[index])):
            y_sam[self.y[index][i][0]] = self.y[index][i][1]

        return X_sam, y_sam


train_dataset = Dataset(train_smat,train_smat)
test_dataset = Dataset(train_smat, test_smat)

params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6 if platform == 'linux' else 0}
training_generator = data.DataLoader(train_dataset, **params)  # sampler = torch.utils.data.SequentialSampler(
# train_dataset))
validation_generator = data.DataLoader(test_dataset, **params) # sampler = torch.utils.data.SequentialSampler(
# train_dataset))

class threeLayerNet(nn.Module):

    def __init__(self):
        '''
        In constructor we define different layers we will use in our architecture.
        '''
        # Constructor call to the superclass
        super(threeLayerNet, self).__init__()
        # Defining the layers to be used in the network.
        # nn.Linear defines a fully connected layer and the first argument represents the input size and the second represents the output size.
        self.layer1 = nn.Linear(100, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 10)
        # Defining the activation function to be used in the network
        self.act = nn.ReLU()

    def forward(self, x):
        '''
        The forward function takes passes the input through the layers and returns the output.
        Inputs:
            -x : Input tensor of shape [N_batch, 100]

        Returns the output of neural network of shape [N_batch, 10]
        '''

        out = self.layer1(x)
        out = self.act(out)
        out = self.layer2(out)
        out = self.act(out)
        out = self.layer3(out)

        return out

#Once we have defined the network class we can create an instance for it.
net = threeLayerNet()

#To get the output of the network on data just call the network instance and feed the inputs

x = torch.rand(5, 100) #Just a random input
network_prediction = net(x)
network_prediction


class twolayerNet(nn.Module):

    def __init__(self):
        '''
        Define the layers and activation functions to be used in the network.
        '''
        super(twolayerNet, self).__init__()
        self.layer1 = nn.Linear(224, 128)
        self.layer2 = nn.Linear(128, 5)
        self.act_1 = nn.ReLU()
        self.act_2 = nn.Tanh()


    def forward(self, x):
        '''
        Implement the forward function which takes as input the tensor x and feeds it to the layers of the network
        and returns the output.

        Inputs:
            -x : Input tensor of shape [N_batch, 224]

        Returns the output of neural network of shape [N_batch, 5]
        '''

        out = torch.zeros(x.shape[0], 5)
        out = self.layer1(x)
        out = self.act_1(out)
        out = self.layer2(out)
        out = self.act_2(out)

        return out


net = twolayerNet()
params_shapes = [p.shape for p in net.parameters()]
params_shapes = sorted(params_shapes)
print("Running Sample Test Case")
assert params_shapes ==[torch.Size([5]),
 torch.Size([5, 128]),
 torch.Size([128]),
 torch.Size([128, 224])]
print("Sample Test Case Passed")

x = torch.rand(10, 224)
print(net(x))


class DAE(nn.Module):
    def __init__(self):
        '''
        Define the layers and activation functions to be used in the network.
        '''
        super(DAE, self).__init__()
        self.layer_1 = nn.Linear(5138, 32)
        self.layer_2 = nn.Linear(32, 5138)
        self.act = nn.Tanh()

    def forward(self, x):
        '''
        Implement the forward function which takes as input the tensor x and feeds it to the layers of the network
        and returns the output.

        Inputs:
            -x : Input tensor of shape [N_batch, 5138]

        Returns the output of neural network of shape [N_batch, 5138]
        '''

        out = torch.zeros(x.shape[0], 5138)
        out = self.layer_1(x)
        out = self.act(out)
        out = self.layer_2(out)

        return out

net = DAE()
"""Don't change code in this cell"""

### SAMPLE TEST CASE
params_shapes = [p.shape for p in net.parameters()]
params_shapes = sorted(params_shapes)
print("Running Sample Test Case")
assert params_shapes == [torch.Size([32]), torch.Size([32, 5138]), torch.Size([5138]), torch.Size([5138, 32])]
print("Sample Test Case Passed")

c = torch.rand(5, 5138)
print(net(c))


def masked_loss(preds, labels):
    '''
    Inputs:
        -preds: Model predictions [N_batch, 5138]
        -labels: User ratings [N_batch, 5138]

    Returns the masked loss as described above.
    '''

    ones = torch.ones_like(preds)
    zeros = torch.zeros_like(preds)
    mask = torch.where(labels == zeros, zeros, ones)
    layer_apply = mask*preds
    loss_mat = (layer_apply-labels)**2
    loss = (torch.sum(loss_mat))/(torch.nonzero(loss_mat).size(0))
    return loss

x = torch.zeros(3, 5138)
x[0][100] = 1
x[0][7] = 1
x[0][1009] = 1
x[1][101] = 1
x[1][8] = 1
x[1][1010] = 1
x[1][56] = 1
x[2][102] = 1
x[2][9] = 1


def train(net, criterion, opti, training_generator, validation_generator, max_epochs=10):
    '''
    Inputs:
        - net: The model instance
        - criterion: Loss function, in our case it is masked_loss function.
        - opti: Optimizer Instance
        - training_generator: For iterating through the training set
        - validation_generator: For iterating through the test set
        - max_epochs: Number of training epochs. One epoch is defined as one complete presentation of the data set.

    Outputs:
        - train_losses: a list of size max_epochs containing the average loss for each epoch of training set.
        - val_losses: a list of size max_epochs containing the average loss for each epoch of test set.

        Note: We compute the average loss in an epoch by summing the loss at each iteration of that epoch
        and then dividing the sum by the number of iterations in that epoch.
    '''

    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        running_loss = 0  # Accumulate the loss in each iteration of the epoch in this variable
        cnt = 0  # Increment it each time to find the number iterations in the epoch.
        # Training iterations
        for batch_X, batch_y in training_generator:
            opti.zero_grad()  # Clears the gradients of all variables.
            preds = net(batch_X)
            loss = masked_loss(preds, batch_y)
            running_loss = running_loss+loss.item()
            loss.backward()
            opti.step()
            cnt = cnt+1
        print("Epoch {}: Training Loss {}".format(epoch + 1, running_loss / cnt))
        train_losses.append(running_loss / cnt)

        # Now that we have trained the model for an epoch, we evaluate it on the test set
        running_loss = 0
        cnt = 0
        with torch.set_grad_enabled(False):
            for batch_X, batch_y in validation_generator:
                preds = net(batch_X)
                loss = masked_loss(preds, batch_y)
                opti.step()
                running_loss = running_loss+loss.item()
                cnt = cnt+1

        print("Epoch {}: Validation Loss {}".format(epoch + 1, running_loss / cnt))

        val_losses.append(running_loss / cnt)

    return train_losses, val_losses

net = DAE()
opti = optim.SGD(net.parameters(), lr=1e-1)
train_losses, val_losses = train(net, masked_loss, opti, training_generator, validation_generator, 20)

plt.plot(train_losses)
plt.plot(val_losses)
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epochs')
plt.show()
