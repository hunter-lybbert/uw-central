"""The model class for training FashionMNIST model."""
from torch import nn
from torch.nn import functional as F


#Define your (As Cool As It Gets) Fully Connected Neural Network 
class MyFirstCNN(nn.Module):
    #Initialize model layers, add additional arguments to adjust
    def __init__(self): 
        super(MyFirstCNN, self).__init__()

        #Define the network layer(s) and activation function(s)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.dropoout = nn.Dropout2d(0.5)
        self.final_layer = nn.Linear(64*7*7, 10)
        self.net = nn.Sequential(
            self.conv1,
            self.batch_norm1,
            nn.ReLU(),
            self.pool,
            self.conv2,
            self.batch_norm2,
            nn.ReLU(),
            self.pool,
            self.conv3,
            self.batch_norm3,
            nn.ReLU(),
            # self.pool,
            self.dropoout,
            nn.Flatten(),
            self.final_layer,
        )  
 
    def forward(self, x):
        #Define how your model propagates the input through the network
        return self.net(x)
    

class MySecondCNN(nn.Module):
    # This model is about 100k params
    def __init__(self, out_channels_list: list[int] = None): 
        super(MySecondCNN, self).__init__()

        if not out_channels_list:
            out_channels_list = [32, 64, 84]

        #Define the network layer(s) and activation function(s)
        conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels_list[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        batch_norm1 = nn.BatchNorm2d(out_channels_list[0])
        pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )
        conv2 = nn.Conv2d(
            in_channels=out_channels_list[0],
            out_channels=out_channels_list[1],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        batch_norm2 = nn.BatchNorm2d(out_channels_list[1])
        conv3 = nn.Conv2d(
            in_channels=out_channels_list[1],
            out_channels=out_channels_list[2],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        batch_norm3 = nn.BatchNorm2d(out_channels_list[2])
        dropoout = nn.Dropout2d(0.5)
        final_layer = nn.Linear(out_channels_list[2]*7*7, 10)
        self.net = nn.Sequential(
            conv1,
            batch_norm1,
            nn.ReLU(),
            pool,
            conv2,
            batch_norm2,
            nn.ReLU(),
            pool,
            conv3,
            batch_norm3,
            nn.ReLU(),
            # self.pool,
            dropoout,
            nn.Flatten(),
            final_layer,
        )
 
    def forward(self, x):
        #Define how your model propagates the input through the network
        return self.net(x)
