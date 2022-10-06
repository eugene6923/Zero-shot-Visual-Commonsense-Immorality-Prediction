import torch
import torch.nn as nn


class ClassificationHead3(nn.Module):
    
    def __init__(self,dropout):
        super(ClassificationHead3, self).__init__()
        self.dense = nn.Linear(512, 512)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(512, 1)

    def forward(self, features, **kwargs):
        x = features.float()
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ClassificationHead2(nn.Module):
    
    def __init__(self,dropout):
        super(ClassificationHead2, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(768, 1)

    def forward(self, features, **kwargs):
        x = features.float()
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


