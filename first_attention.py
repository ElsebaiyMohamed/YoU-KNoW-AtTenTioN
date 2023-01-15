import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, taw=0.02, *args, **kwargs):
        super(Attention, self).__init__()
        self.taw = taw
        
    def forward(self, query, memory, *args, **kwargs):
        '''
        params:
                @query: current decoder hidden state
                @memory: all encoder hidden states
        return:
                weighted average above the memory based on current decoder hidden state
        '''
        
        
        if not pt.is_tensor(query): 
            query = pt.tensor(query)
        if not pt.is_tensor(memory): 
            memory = pt.tensor(memory)
            
        scores = pt.matmul(query, memory.mT)
    
        scores = F.softmax(scores/self.taw, -1)
        
        scores = pt.matmul(scores, memory)
        
        return scores
        
        
        
if __name__ == '__main__':
    np.random.seed(1)
    q      = np.random.randint(1, 5, (3, 1, 6)).astype(np.float32)
    memory = np.random.randint(1, 5, (3, 3, 6)).astype(np.float32)

    cal = Attention()
    
    cal(q, memory)