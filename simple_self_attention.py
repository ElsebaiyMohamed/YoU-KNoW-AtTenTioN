import torch as pt
from torch import nn
import torch.nn.functional as F

class SimpleSelfAttention(nn.Module):
    def __init__(self):
        super(SimpleSelfAttention, self).__init__()

    def forward(self, x):
        
        w = F.softmax(pt.matmul(x, x.mT), 2)
        y = pt.matmul(w, x)
        return y
        
        
class PosEmpedding(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(PosEmpedding, self).__init__()
        self.emppedding = nn.Embedding(*args, **kwargs)
        self.position   = nn.Embedding(*args, **kwargs)

    def forward(self, x):
        _, l, d = x.size()
        return F.layer_norm(self.emppedding(x) + self.position(x), [l, d])
    
    
if __name__ == '__main__':
    
    d = pt.randint(-10, 10, (1, 10, 3), dtype=pt.float)
    m = SimpleSelfAttention()
    
    print(m(d).size())
    # print(d.size())
    # print(pt.matmul(d, d.mT).size())
    #print(d.permute(0, 2, 1).size())
        
        