import torch.nn as nn
from ann_base import ANNBase


class ANNSimple(ANNBase):
    def __init__(self, train_ds, test_ds):
        super().__init__(train_ds, test_ds)
        self.linear = nn.Sequential(
            nn.Linear(train_ds.x.shape[1],10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )

    def forward(self,x):
        return self.linear(x)

