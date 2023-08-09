import torch
from torch.utils.data import DataLoader
import numpy as np

class mydataset:
    def __init__(self,data):
        self.data=data
    def __len__(self):#必须重写
        return len(self.data)
    def __getitem__(self,idx):#必须重写
        return self.data[idx]

a=np.random.rand(4,3)#4个数据，每一个数据是一个向量。
print(a)

dataset=mydataset(a)
print(len(dataset))
print(dataset[0])

dataloader=DataLoader(dataset,batch_size=2, collate_fn=lambda x: x)

for d in dataloader:
    print(type(d))