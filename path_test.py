import os

test = sorted([s for s in os.listdir('./dataset/mage/train') if s.endswith('csv')])

print(test)
