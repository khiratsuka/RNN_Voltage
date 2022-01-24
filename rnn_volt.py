import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset¥
import numpy as np
import matplotlib.pyplot as plt

#If GPU is available, use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['mage', 'nobashi']

class MyRNN(nn.Module):
    def __init__(self):
        super(MyRNN, self).__init__()
        self.input_size = 25000
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

class EMGDataset(Dataset):
    def __init__(self, emg_data_folder='./dataset',
                 class_name='hoge',
                 is_train=True):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.emg_data_paths = self._get_file_names(emg_data_folder)
        self.class_name = class_name

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

    def __len__(self):
        #_get_file_namesでファイル数取得してreturnする

    def _get_file_names(self):
        phase = 'train' if self.is_train else 'test'
        emg_data, correct_class = [], []

        emg_data_dir = os.path.join(self.emg_data_folder, self.class_name, phase)

        data_names = sorted([name for name in os.listdir('./dataset/mage/train') if name.endswith('csv')])
        


model = MyRNN().to(device)
