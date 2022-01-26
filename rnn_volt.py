# coding: utf-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import numpy as np
#import matplotlib.pyplot as plt

#If GPU is available, use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#setting classname
CLASS_NAMES_DICT = {'mage':0, 'nobashi':1}
CLASS_ID = [0, 1]
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
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
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
        #assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.emg_data_folder = emg_data_folder
        self.class_name = class_name
        self.is_train = is_train

        self.emg_data_path, self.correct_class = self._get_file_names()

    def __getitem__(self, idx):
        emg_data_path, correct_class = self.emg_data_path[idx], self.correct_class[idx]
        list_correct_class = []
        list_correct_class.append(correct_class)
        max_data_sampling_num = 24000
        plus_data = 0.5

        #loading data
        emg_data = np.loadtxt(emg_data_path, delimiter='\n')

        #EMGのデータ長を揃える
        #基準値に達していなかったら前padding, 超えていたら後ろのデータをカット
        over_data_num = max_data_sampling_num - len(emg_data)
        if over_data_num > 0:
            plus_data_list = []
            for i in range(over_data_num):
                plus_data_list.append(plus_data)
            emg_data = [*plus_data_list, *emg_data]
        elif over_data_num < 0:
            del emg_data[over_data_num:]

        tensor_emg_data = torch.FloatTensor(emg_data)
        tensor_correct_class = torch.LongTensor(list_correct_class)
        return tensor_emg_data, tensor_correct_class

    def __len__(self):
        return len(self.emg_data_path)

    def _get_file_names(self):
        phase = 'train' if self.is_train else 'test'
        emg_data_path, correct_class, temp_class = [], [], [0]

        #set directory path
        for cname in self.class_name:
            emg_data_dir = os.path.join(self.emg_data_folder, cname, phase)
            data_names = sorted([name for name in os.listdir(emg_data_dir) if name.endswith('csv')])

            #checking directory of csv data
            for data_name in data_names:
                data_name_path = os.path.join(emg_data_dir, data_name)
                if not os.path.join(data_name_path):
                    continue

                emg_data_path.append(data_name_path)
                correct_class.append(CLASS_NAMES_DICT[cname])

        assert len(emg_data_path) == len(correct_class), 'number of emg_data and class are not same.'
        return list(emg_data_path), list(correct_class)



def main():
    dataset_folder = './dataset'
    Batch_size = 2
    Num_epochs = 100
    LearningRate = 0.001
    Train_EMG_Dataset = EMGDataset(emg_data_folder = dataset_folder,
                                   class_name = CLASS_NAMES,
                                   is_train=True)
    Test_EMG_Dataset  = EMGDataset(emg_data_folder = dataset_folder,
                                   class_name = CLASS_NAMES,
                                   is_train=False)

    Train_dataset_size = int(0.8 * len(Train_EMG_Dataset))
    Val_dataset_size = len(Train_EMG_Dataset) - Train_dataset_size
    Train_EMG_Dataset, Val_EMG_Dataset = torch.utils.data.random_split(Train_EMG_Dataset, [Train_dataset_size, Val_dataset_size])

    Train_DataLoader = DataLoader(Train_EMG_Dataset,
                                  batch_size=Batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True,
                                  pin_memory=True)
    Val_DataLoader   = DataLoader(Val_EMG_Dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True,
                                  pin_memory=True)
    Test_DataLoader  = DataLoader(Test_EMG_Dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True,
                                  pin_memory=True)

    net = MyRNN().to(device)
    history = {
        'train_loss':[],
        'val_loss':[],
        'val_acc':[]
    }
    criterion = F.cross_entropy
    optimizer = optim.Adam(params=net.parameters(), lr=LearningRate)

    for epoch in range(Num_epochs):
        #train phase
        net.train()
        temp_train_loss = 0
        for batch in Train_DataLoader:
            data, label = batch
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            preds = net(data)
            loss = criterion(preds, label)
            loss.backward()
            temp_train_loss += loss.item()
            optimizer.step()
        history['train_loss'].append(temp_train_loss/len(Train_DataLoader))

        #validation phase
        net.eval()
        temp_val_loss = 0
        temp_val_acc = 0
        with torch.no_grad():
            for batch in Val_DataLoader:
                data, label = batch

                preds = net(data)
                loss = criterion(preds, label)
                label_preds = torch.argmax(preds, dim=1)
                temp_val_loss += loss.item()
                temp_val_acc += torch.sum(labem_preds == label)
            history['val_loss'].append(temp_val_loss/len(Val_DataLoader))
            history['val_acc'].append(temp_val_acc/len(Val_DataLoader))
        print(history)
        
if __name__ == '__main__':
    main()
