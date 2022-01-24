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

a = [1, 2, 3]
b = [4, 5, 6]
c = 777
n = []

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
test0 = [c, *a]
test1 = [*a, *b]

print(test0)
print(test1)
"""
