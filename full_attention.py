import numpy as np
import torch as pt
from torch import nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init(self, input_dim, first_dim, out_dim, *args, **kwargs):
        super(FeedForward, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, first_dim)
        self.layer1 = nn.Linear(first_dim, out_dim)
        
    def forward(self, x):
        _, l, d = x.size()
        z = F.leaky_relu(self.layer1(x), negative_slope=0.6)
        z =  F.leaky_relu(self.layer2(x), negative_slope=0.6)
        return F.layer_norm(x + z, [l, d])
    


class SelfAttention(nn.module):
    def __init__(self, *args, **kwargs):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(**kwargs) #embed_dim, num_heads, batch_first
        self.ffn = FeedForward(*args)              #input_dim, first_dim, out_dim,
    
    def forward(self, x, **kwargs):
        _, l, d = x.size()
        
        attention_out = self.mha(x, **kwargs)
        x = F.layer_norm(x+attention_out, [l, d])
        
        x = F.layer_norm(x + self.ffn(x), [l, d])
        
        return x, attention_out
        
        