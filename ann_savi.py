import torch
import torch.nn as nn
from ann_base import ANNBase


class ANNSAVI(ANNBase):
    def __init__(self, train_ds, test_ds):
        super().__init__(train_ds, test_ds)
        self.linear = nn.Sequential(
            nn.Linear(1,5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        self.L = nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def forward(self,x):
        b4 = x[:,0:1]
        b8 = x[:,1:2]
        savi = ( (b8-b4) / (b8+b4+self.L) )*(1+self.L)
        return self.linear(savi)


    def verbose_after(self,ds):
        print(f" L = {self.L.item():.6f}", end="")