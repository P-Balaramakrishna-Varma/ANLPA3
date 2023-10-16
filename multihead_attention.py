import torch
import torch.nn as nn
import math


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, batch_first=True):
        # batch_first always true
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = int(self.embed_dim / self.n_heads)
       
        # paramters
        self.query_matrix = nn.Linear(embed_dim, n_heads * self.head_dim, bias=False)  #W_q
        self.key_matrix = nn.Linear(embed_dim, n_heads * self.head_dim, bias=False)    #W_k
        self.value_matrix = nn.Linear(embed_dim, n_heads * self.head_dim, bias=False)  #W_v
        self.out = nn.Linear(n_heads * self.head_dim, embed_dim, bias=False)           #W_o
        self.softmax = nn.Softmax(dim=2)
        
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # attn_mask None or 3d tensor
        # key_padding_mask None or 2d tensor
        Query = self.query_matrix(query)
        Key = self.key_matrix(key)
        Value = self.value_matrix(value)
        
        attention_heads = []
        for i in range(self.n_heads):
            vals = Value[:, :, i * self.head_dim:(i+1) * self.head_dim]
            keys = Key[:, :, i * self.head_dim:(i+1) * self.head_dim]
            queries = Query[:, :, i * self.head_dim:(i+1) * self.head_dim]
            att_score = torch.einsum("hij,hkj->hik", queries, keys) / math.sqrt(self.head_dim)
            attn_weights = self.softmax(att_score)
            attention = torch.einsum("hij, hjk->hik", attn_weights, vals)
            attention_heads.append(attention)
        Attention = torch.cat(attention_heads, dim=2)
        return self.out(Attention)
    
    
    
if __name__ == "__main__":
    query = torch.rand(2, 4, 300)
    # print(query)
    model = MultiheadAttention(300, 10)
    model(query, query, query)