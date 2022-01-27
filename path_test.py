import os
import numpy as np
import torch

"""
test = sorted([s for s in os.listdir('./dataset/mage/train') if s.endswith('csv')])


for t in test:
    data_name_path = os.path.join('./dataset/mage/train', t)
    if not os.path.join(data_name_path):
        continue
    print(data_name_path)
myarray = np.loadtxt(data_name_path, delimiter='¥n')

print(myarray)
"""
"""
a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
b = [4, 5, 6]
c = 777
n = []

array_a = np.array(a)
print(a)
print(array_a.shape)
array_a = np.reshape(array_a, (array_a.shape[0], 1))
print(array_a.shape)
print(array_a)
array_a = np.reshape(array_a, (3, 4))
print(array_a.shape)
print(array_a)
#print(np.delete(array_a, slice(3,None), 0))
"""
"""
num = 10 - len(a)
for i in range(num):
    n.append(c)
test = [*n, *a]
print(test)
print(len(test))

del test[-3:]
print(test)
print(len(test))

"""
"""
test0 = [c, *a]
test1 = [*a, *b]

print(test0)
print(test1)
"""
history = {
    'train_loss':[1, 2, 3],
    'train_acc':[1, 2, 3],
    'val_loss':[4, 5, 6],
    'val_acc':[4, 5, 6]
}

history = torch.from_numpy(history).clone()
history = history.to('cuda')
history = history.to('cpu').detach().numpy().copy()
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
