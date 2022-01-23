import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt

#If GPU is available, use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyRNN(nn.Module):
    def __init__(self):
        super(MyRNN, self).__init__()
        self.input_size = 24000
        self.hidden_size = 512
        #self.rnn_layers = 512
        self.num_classes = 2
        self.rnn = nn.RNN(input_size = self.input_size,
                          hidden_size = self.hidden_size,
                          #rnn_layers = self.rnn_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        #batch_size = x.shape[0]
        x = x.to(device)
        x_rnn, hidden = self.rnn(x, None)
        x = self.fc(x_rnn[:, -1, :])
        x = self.softmax(x)
        return x

model = MyRNN().to(device)
