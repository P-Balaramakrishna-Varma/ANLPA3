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
        Query = self.query_matrix(query).reshape(query.shape[0], query.shape[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        Key = self.key_matrix(key).reshape(key.shape[0], key.shape[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        Value = self.value_matrix(value).reshape(value.shape[0], value.shape[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn_score = torch.einsum("bhij, bhkj->bhik", Query, Key) / math.sqrt(self.head_dim)
        attn_wights = self.softmax(attn_score)
        attention = torch.einsum("bhij, bhjk->bhik", attn_wights, Value)
        attention = attention.permute(0, 2, 1, 3).reshape(query.shape[0], query.shape[1], self.n_heads * self.head_dim)
        return self.out(attention)
    
    
 
if __name__ == "__main__":
    query = torch.rand(2, 4, 300)
    # print(query)
    model = MultiheadAttention(300, 10)
    model(query, query, query)