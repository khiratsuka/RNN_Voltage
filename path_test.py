import os
import numpy as np

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
