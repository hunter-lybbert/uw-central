"""The model class for training FashionMNIST model."""
from torch import nn
from torch.nn import functional as F


#Define your (As Cool As It Gets) Fully Connected Neural Network 
class MyFirstCNN(nn.Module):
    #Initialize model layers, add additional arguments to adjust
    def __init__(self): 
        super(MyFirstCNN, self).__init__()

        #Define the network layer(s) and activation function(s)
        pass
 
    def forward(self, x):
        #Define how your model propagates the input through the network
        pass