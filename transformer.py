import torch, math
import torch.nn as nn


class GlobalEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


class PostionalEmbedding(nn.Module):
    def __init__(self, max_length, d_model):
        super().__init__()
        self.pe = gen_pe(max_length, d_model)
    
    def forward(self, x):
        len = x.shape[1]
        return self.pe[:len, :]

    
def gen_pe(max_length, d_model):
    n = 10000
    pe = torch.zeros(max_length*d_model).reshape(max_length, d_model) 

    for k in torch.arange(max_length):
        for i in torch.arange(d_model//2):
            theta = k / (n ** ((2*i)/d_model))       
            pe[k, 2*i] = math.sin(theta) 
            pe[k, 2*i+1] = math.cos(theta)

    return pe


