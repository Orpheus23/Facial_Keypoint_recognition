## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init 


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        '''
        LAYER 1
        Input (1,96,96)
        Convolution2d (32,93,93)
        Activation (32,93,93)
        Maxpooling2d (32,46,46)
        Dropout (32,46,46)
        '''
        self.conv1 = nn.Conv2d(1,32,4)
        #torch.nn.init.xavier_uniform(self.conv1.weight)
        self.mxpool1 = nn.MaxPool2d(2)
        self.drop1 =  nn.Dropout(0.1)
        '''
        LAYER 2
        Input (32,46,46)
        Convolution2d (64,44,44)
        Activation (64,44,44)
        Maxpooling2d (64,22,22)
        Dropout (64,22,22)
        '''
        self.conv2 = nn.Conv2d(32,64,3)
        #torch.nn.init.xavier_uniform(self.conv2.weight)
        self.mxpool2 = nn.MaxPool2d(2)
        self.drop2 =  nn.Dropout(0.2)
        '''
        LAYER 3
        Input (64,22,22)
        Convolution2d (128,21,21)
        Activation (128,21,21)
        Maxpooling2d (128,10,10)
        Dropout (128,10,10)
        '''
        self.conv3 = nn.Conv2d(64,128,2)
        #torch.nn.init.xavier_uniform(self.conv3.weight)
        self.mxpool3 = nn.MaxPool2d(2)
        self.drop3 =  nn.Dropout(0.3)
        '''
        LAYER 4
        Input (128,10,10)
        Convolution2d (256,10,10)
        Activation (256,10,10)
        Maxpooling2d (256,5,5)
        Dropout (256,5,5)
        '''
        self.conv4 = nn.Conv2d(128,256,1)
        #torch.nn.init.xavier_uniform(self.conv4.weight)
        self.mxpool4 = nn.MaxPool2d(2)
        self.drop4 =  nn.Dropout(0.4)
        '''
        LAYER 5
        Flatten (6400)
        Dense (1000)
        Activation (1000)
        Dropout (1000)
        '''
        self.flat = nn.Linear(256*13*13, 1000)
        #torch.nn.init.xavier_uniform(self.flat.weight)
        self.dense1 = nn.Linear(1000, 1000)
        torch.nn.init.xavier_uniform(self.dense1.weight)
        self.drop5 =  nn.Dropout(0.5)
        '''
        LAYER 6
        Dense (1000)
        Activation (1000)
        Dropout (1000)
        Dense (2)
        '''
        self.dense2 = nn.Linear(1000, 1000)
        torch.nn.init.xavier_uniform(self.dense2.weight)
        self.drop6 =  nn.Dropout(0.6)
        self.dense3 = nn.Linear(1000,136)
        torch.nn.init.xavier_uniform(self.dense3.weight)
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        #self.conv2 = nn.Conv2d
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(self.mxpool1(F.elu(self.conv1(x)))) 
        x = self.drop2(self.mxpool2(F.elu(self.conv2(x))))
        x = self.drop3(self.mxpool3(F.elu(self.conv3(x))))
        x = self.drop4(self.mxpool4(F.elu(self.conv4(x))))
        #print(x.shape)
        x = x.view(-1,256*13*13)
        x = self.drop5(F.elu(self.dense1(self.flat(x))))  
        x = self.dense3(self.drop5(F.tanh(self.dense2(x)))) 
        # a modified x, having gone through all the layers of your model, should be returned
        return x
