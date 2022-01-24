import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset¥
import numpy as np
import matplotlib.pyplot as plt

#If GPU is available, use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#setting classname
CLASS_NAMES_DICT = {'mage':0, 'nobashi':1}

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
        self.emg_data_folder = emg_data_folder
        self.class_name = class_name

        self.emg_data_path, self.correct_class = _get_file_names()

    def __getitem__(self, idx):
        emg_data_path, correct_class = self.emg_data_path[idx], self.correct_class[idx]
        max_data_sampling_num = 24000
        plus_data = 0.5

        #loading data
        emg_data = np.loadtxt(emg_data_path, delimiter='¥n')

        #EMGのデータ長を揃える
        #基準値に達していなかったら前padding, 超えていたら後ろのデータをカット
        over_data_num = max_data_sampling_num - len(emg_data)
        if over_data_num > 0:
            plus_data_list = []
            for i in range(plus_data_num):
                plus_data_list.append(plus_data)
            emg_data = [*plus_data, *emg_data]
        elif over_data_num < 0:
            del emg_data[over_data_num:]

        return emg_data, correct_class

    def __len__(self):
        return len(self.emg_data_path)

    def _get_file_names(self):
        phase = 'train' if self.is_train else 'test'
        emg_data_path, correct_class = [], []

        #set directory path
        emg_data_dir = os.path.join(self.emg_data_folder, self.class_name, phase)
        data_names = sorted([name for name in os.listdir(emg_data_dir) if name.endswith('csv')])

        #checking directory of csv data
        for data_name in data_names:
            data_name_path = os.path.join(emg_data_dir, data_name)
            if not os.path.join(data_name_path):
                continue
            emg_data_path.extend(data_name_path)
            correct_class.extend(CLASS_NAMES_DICT[self.class_name])

        assert len(emg_data_path) == len(correct_class), 'number of emg_data and class are not same.'

        return list(emg_data_path), list(correct_class)




model = MyRNN().to(device)
