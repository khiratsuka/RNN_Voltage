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
from tqdm import tqdm
import matplotlib.pyplot as plt

#If GPU is available, use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#setting classname
CLASS_NAMES_DICT = {'mage':0, 'nobashi':1}
CLASS_ID = [0, 1]
CLASS_NAMES = ['mage', 'nobashi']
num_class_num = 2

class MyRNN(nn.Module):
    def __init__(self):
        super(MyRNN, self).__init__()
        self.input_size = 100
        self.hidden_size = 512
        #self.rnn_layers = 512
        self.num_classes = num_class_num
        self.rnn = nn.RNN(input_size = self.input_size,
                          hidden_size = self.hidden_size,
                          #rnn_layers = self.rnn_layers,
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #batch_size = x.shape[0]
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
        max_data_sampling_num = 25000
        append_num_data = 0.5
        seq_batch_size = 100
        seq_batch_num = int(max_data_sampling_num / seq_batch_size)

        #loading data
        emg_data = np.loadtxt(emg_data_path, delimiter='\n')
        emg_data = np.reshape(emg_data, (emg_data.shape[0], 1))

        #EMGのデータ長を揃える
        #基準値に達していなかったら前padding, 超えていたら後ろのデータをカット
        over_data_num = max_data_sampling_num - len(emg_data)
        if over_data_num > 0:
            append_array = np.full((over_data_num, 1), append_num_data)
            emg_data = np.append(append_array, emg_data, axis=0)
        elif over_data_num < 0:
            emg_data = np.delete(emg_data, slice(over_data_num, None), 0)

        #RNNの入力は[シーケンシャルデータ(バッチ)の個数, サイズ, データ]であるから, reshape
        emg_data = np.reshape(emg_data, (seq_batch_num, seq_batch_size))
        #シーケンシャルデータの個数分の正解データを用意
        array_correct_class = np.full((num_class_num), 0.0)
        array_correct_class[correct_class] = 1.0

        tensor_emg_data = torch.FloatTensor(emg_data)
        tensor_correct_class = torch.FloatTensor(array_correct_class)
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
                                  num_workers=1,
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
        'train_acc':[],
        'val_loss':[],
        'val_acc':[]
    }
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=LearningRate)

    print('train start.\n')
    for epoch in range(Num_epochs):
        #train phase
        net.train()
        temp_train_loss = 0
        temp_train_acc = 0
        with tqdm(total=len(Train_DataLoader), unit='batch', desc='{}/{} epochs [Train]'.format(epoch+1, Num_epochs)) as progress_bar:
            for data, label in Train_DataLoader:
                data = data.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                preds = net(data)
                loss = criterion(preds, label)
                label_preds = torch.argmax(preds, dim=1)
                loss.backward()
                temp_train_loss += loss.item()
                temp_train_acc +=torch.sum(label_preds == label)
                optimizer.step()
                history['train_loss'].append(temp_train_loss/len(Train_DataLoader))
                history['train_acc'].append(temp_train_acc/len(Train_DataLoader))
                progress_bar.update(1)

        #validation phase
        net.eval()
        temp_val_loss = 0
        temp_val_acc = 0
        with tqdm(total=len(Val_DataLoader), unit='batch', desc='{}/{} epochs [Val]'.format(epoch+1, Num_epochs)) as progress_bar:
            with torch.no_grad():
                for data, label in Val_DataLoader:
                    data = data.to(device)
                    label = label.to(device)
                    preds = net(data)
                    loss = criterion(preds, label)
                    label_preds = torch.argmax(preds, dim=1)
                    temp_val_loss += loss.item()
                    temp_val_acc += torch.sum(label_preds == label)
                    history['val_loss'].append(temp_val_loss/len(Val_DataLoader))
                    history['val_acc'].append(temp_val_acc/len(Val_DataLoader))
                    progress_bar.update(1)

        print('----{} epochs----'.format(epoch))
        print('train_loss : ' + str(history['train_loss'][epoch]))
        print('train_acc : ' + str(history['train_loss'][epoch]))
        print('val_loss : ' + str(history['val_loss'][epoch]))
        print('val_acc : ' + str(history['val_acc'][epoch]))
        print('\n')

    net.eval()
    test_acc = 0
    with tqdm(total=len(Test_DataLoader), unit='batch', desc='{}/{} epochs [Val]'.format(epoch+1, Num_epochs)) as progress_bar:
        with torch.no_grad():
            for data, label in Test_DataLoader:
                data = data.to(device)
                label = label.to(device)
                preds = net(data)
                loss = criterion(preds, label)
                label_preds = torch.argmax(preds, dim=1)
                test_acc += torch.sum(label_preds == label)
                test_acc = test_acc/len(Test_DataLoader)
                progress_bar.update(1)
    print('test_acc = {}'.format(test_acc))

    #history = history.to('cpu').detach().numpy().copy()

    metrics = ['loss', 'acc']
    plt.figure(figsize=(10, 5))
    for i in range(len(metrics)):
        metric = metrics[i]

        plt.subplot(1, 2, i+1)
        plt.title(metric)
        plt_train = history['train_' + metric]
        plt_val   = history['val_'   + metric]
        plt.plot(plt_train, label='train')
        plt.plot(plt_val,   label='val')
        plt.legend()

    plt.show()
if __name__ == '__main__':
    main()
