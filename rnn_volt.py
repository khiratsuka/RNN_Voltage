# coding: utf-8
import os
import datetime
import math
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
#device = torch.device('cpu')

#setting classname
CLASS_NAMES_DICT = {'mage':0, 'nobashi':1}
CLASS_ID = [0, 1]
CLASS_NAMES = ['mage', 'nobashi']
num_class_num = 2
seq_batch_size = 248

class MyRNN(nn.Module):
    def __init__(self):
        super(MyRNN, self).__init__()
        self.input_size = seq_batch_size
        self.hidden_size = 100
        self.rnn_layers = 1
        self.num_classes = num_class_num
        self.rnn = nn.LSTM(input_size = self.input_size,
                           hidden_size = self.hidden_size,
                           num_layers = self.rnn_layers,
                           batch_first=True,
                           dropout=0.5)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

        """
        nn.init.normal_(self.rnn.weight_ih_l0, std=1/math.sqrt(self.input_size))
        nn.init.normal_(self.rnn.weight_hh_l0, std=1/math.sqrt(self.hidden_size))
        nn.init.zeros_(self.rnn.bias_ih_l0)
        nn.init.zeros_(self.rnn.bias_hh_l0)
        """
        """
        nn.init.normal_(self.rnn.weight_ih_l1, std=1/math.sqrt(self.input_size))
        nn.init.normal_(self.rnn.weight_hh_l1, std=1/math.sqrt(self.hidden_size))
        nn.init.zeros_(self.rnn.bias_ih_l1)
        nn.init.zeros_(self.rnn.bias_hh_l1)
        """
        #nn.init.normal_(self.fc.weight, std=0.01)
        #nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        #x_rnn, hidden = self.rnn(x, None)
        #x = self.fc2(x_rnn[:, -1, :])
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        x = self.relu(x)
        x = self.fc2(x)
        #print(x.shape)
        #x = torch.sigmoid(x)
        #x = self.softmax(x)
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
        seq_batch_num = 1

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


        temp_emg_data = 0.0
        normalize_emg_data = []
        for i in range(len(emg_data)):
            temp_emg_data += emg_data[i]
            if i % 100 == 0:
                normalize_emg_data.append(temp_emg_data/100)
                temp_emg_data = 0.0
        #print(normalize_emg_data)
        #plt.plot(normalize_emg_data)
        #plt.show()
        #print(len(normalize_emg_data))
        #RNNの入力は[シーケンシャルデータ(バッチ)の個数, サイズ, データ]であるから, reshape
        #print(emg_data.shape)
        emg_data = np.reshape(normalize_emg_data, (250))
        emg_data = np.fft.fft(emg_data)
        emg_data = emg_data[1:-1]
        emg_data = np.reshape(emg_data, (248))
        max_emg = np.max(emg_data)
        min_emg = np.min(emg_data)
        for i in range(len(emg_data)):
            emg_data[i] = (emg_data[i] - min_emg) / (max_emg - min_emg)
        #print(emg_data.shape)

        #plt.plot(emg_data)
        #plt.show()
        #plt.plot(test)
        #plt.show()

        #正解のクラス番号を入れるarrayを用意
        #array_correct_class = np.full((1), 0)
        #array_correct_class[0] = correct_class

        tensor_emg_data = torch.FloatTensor(emg_data)
        #tensor_correct_class = torch.LongTensor(array_correct_class)
        return tensor_emg_data, correct_class

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
    dataset_folder = './dataset/'
    result_folder = './result/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    Batch_size = 2
    Num_epochs = 100
    LearningRate = 0.01
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(), lr=LearningRate)

    print('train start.\n')
    for epoch in range(Num_epochs):
        #train phase
        net.train()
        temp_train_loss = 0.0
        temp_train_acc = 0.0
        with tqdm(total=len(Train_DataLoader), unit='batch', desc='{}/{} epochs [Train]'.format(epoch+1, Num_epochs)) as progress_bar:
            for data, label in Train_DataLoader:
                #plt.plot(data[0])
                #plt.show()
                #plt.plot(data[1])
                #plt.show()
                data = data.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                preds = net(data)
                #print('before:'+str(net.rnn.weight_ih_l0))
                #print('Preds:'+str(preds)+'\n')
                #print('label:'+str(label)+'\n')
                loss = criterion(preds, label)
                #print('loss:'+str(loss.item())+'\n')
                sm = nn.Softmax(dim=1)
                preds = sm(preds)
                label_preds = torch.argmax(preds, dim=1)
                for bn in range(len(label)):
                    if label_preds[bn] == label[bn]:
                        temp_train_acc += 1.0
                loss.backward()
                temp_train_loss += float(loss.item()) * float(data.size(0))
                optimizer.step()
                #print('after:'+str(net.rnn.weight_ih_l0))
                progress_bar.update(1)


            history['train_loss'].append(temp_train_loss/Train_dataset_size)
            history['train_acc'].append(temp_train_acc/Train_dataset_size)

        #validation phase

        net.eval()
        temp_val_loss = 0.0
        temp_val_acc = 0.0
        with tqdm(total=len(Val_DataLoader), unit='batch', desc='{}/{} epochs [Val]'.format(epoch+1, Num_epochs)) as progress_bar:
            with torch.no_grad():
                for data, label in Val_DataLoader:
                    #plt.plot(data[0])
                    #plt.show()
                    #plt.plot(data[1])
                    #plt.show()
                    data = data.to(device)
                    label = label.to(device)
                    preds = net(data)
                    loss = criterion(preds, label)
                    sm = nn.Softmax(dim=1)
                    preds = sm(preds)
                    #print('Preds:'+str(preds)+'\n')
                    #print('label:'+str(label)+'\n')
                    label_preds = torch.argmax(preds, dim=1)
                    #print(preds)
                    #print(label_preds)
                    #label_correct = torch.argmax(label, dim=1)
                    #print(loss.item())
                    temp_val_loss += float(loss.item())
                    for bn in range(len(label)):
                        if label_preds[bn] == label[bn]:
                            temp_val_acc += 1.0
                    progress_bar.update(1)
        history['val_loss'].append(temp_val_loss/Val_dataset_size)
        history['val_acc'].append(temp_val_acc/Val_dataset_size)

        print('----{} epochs----'.format(epoch))
        print('train_loss : ' + str(history['train_loss'][epoch]))
        print('train_acc : ' + str(history['train_acc'][epoch]))
        print('val_loss : ' + str(history['val_loss'][epoch]))
        print('val_acc : ' + str(history['val_acc'][epoch]))
        print('\n')

    net.eval()
    test_acc = 0.0
    with tqdm(total=len(Test_DataLoader), unit='batch', desc='{}/{} epochs [Test]'.format(epoch+1, Num_epochs)) as progress_bar:
        with torch.no_grad():
            for data, label in Test_DataLoader:
                data = data.to(device)
                label = label.to(device)
                preds = net(data)
                loss = criterion(preds, label)
                label_preds = torch.argmax(preds, dim=1)
                for bn in range(len(label)):
                    test_acc = 1.0 + test_acc if label_preds[bn] == label[bn] else test_acc
                progress_bar.update(1)
    test_acc = test_acc/len(Test_EMG_Dataset)
    print('test_acc = {}'.format(test_acc))

    #history = history.to('cpu').detach().numpy().copy()
    #print(history['train_loss'])
    #print(history['train_acc'])
    #print(history['val_loss'])
    #print(history['val_acc'])

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

    now = datetime.datetime.now()
    fname = now.strftime("%Y_%m_%d_%H%M%S") + '_eval.png'
    plt.savefig(result_folder+fname)
if __name__ == '__main__':
    main()
