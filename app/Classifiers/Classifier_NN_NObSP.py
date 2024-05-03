import torch
from torch import nn
import torch.nn.functional as F

# Class to add an interpretation Layer on a Neural network Classification model.
#
# @Copyrigth:  Alexander Caicedo, April 2023

class Classifier_NN_NObSP(nn.Module):
    def __init__(self, model,W1,W2,b,p):
        super(Classifier_NN_NObSP, self).__init__()
        
        if not torch.is_tensor(W1):
            W1 = torch.from_numpy(W1)
        
        if not torch.is_tensor(W2):
            W2 = torch.from_numpy(W2)
        
        d, N = W1.size()
        
        self.previous= nn.Sequential(*list(model.children())[:-1])
        self.Linear_NObSP = nn.Linear(out_features = d, in_features = N) # Creting the linear layer that willintorduce interpretability in the model
        self.Linear_NObSP.weight = nn.Parameter(data=W1,requires_grad=False) # Assigning the weigths for the interpretability. These weigths were obtained using NObSP
        self.Linear_NObSP.bias = nn.parameter.Parameter(torch.zeros(1,d),requires_grad=False) # Setting th ebias to zero
        self.Linear_Final = nn.Linear(in_features = d*p, out_features = p) # Creating the output layer
        self.Linear_Final.weight = nn.Parameter(data=W2,requires_grad=False) # Assigning the lienar combination of the interpreatable neurons (ideally it will be  wigths with just values of 1)
        self.Linear_Final.bias = nn.parameter.Parameter(b,requires_grad=False) # Assigning the bias term to estimate the output
        self.prev_model = model # Assigning the previous model to a variable called prev_model. This will be used further.
        
    def forward(self,x):
        x1, x, y = self.prev_model(x) # Computing the transformation of the data, up to the last layer (previous to the output) in the original model
        # Evalauting the interpretable layer and computing the output
        xt = self.Linear_NObSP(x) # Computing the transformation done to the vector in the layer previous to the output
        x = F.softmax(self.Linear_Final(xt),dim=1) 
        return x, xt