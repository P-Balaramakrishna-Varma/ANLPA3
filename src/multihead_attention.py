import torch
import torch.nn as nn
import math
# from transformer import make_trg_mask

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
        # projections
        Query = self.query_matrix(query).reshape(query.shape[0], query.shape[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        Key = self.key_matrix(key).reshape(key.shape[0], key.shape[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        Value = self.value_matrix(value).reshape(value.shape[0], value.shape[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # caliculating attentinon scores
        attn_score = torch.einsum("bhij, bhkj->bhik", Query, Key) / math.sqrt(self.head_dim)
        
        # Masking and calliculating attention weights
        mask = torch.zeros(attn_score.shape).bool().to('cuda')
        if attn_mask is not None:
            attn_mask = attn_mask.reshape(-1, self.n_heads, attn_mask.shape[1], attn_mask.shape[1])
            mask = torch.logical_or(mask, attn_mask)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1).expand(key_padding_mask.shape[0], self.n_heads, mask.shape[2], key_padding_mask.shape[1])
            mask = torch.logical_or(mask, key_padding_mask)
        
        attn_score_masked = attn_score.clone()
        attn_score_masked[mask] = 0
        attn_weights = self.softmax(attn_score_masked)
        attn_weights_masked = attn_weights.clone()
        attn_weights_masked[mask] = 0        
        
        # weighted summation
        attention = torch.einsum("bhij, bhjk->bhik", attn_weights_masked, Value)
        attention = attention.permute(0, 2, 1, 3).reshape(query.shape[0], query.shape[1], self.n_heads * self.head_dim)
        return self.out(attention), "Dummy for consistency"
    
    
 
if __name__ == "__main__":
    query = torch.rand(2, 4, 300)
    # print(query)
    model = MultiheadAttention(300, 10)
    # attn_mask = make_trg_mask(torch.rand(2, 4), 10)
    key_padding_mask = torch.zeros(2, 4).bool()
    hello, _ = model(query, query, query)
    a = hello.sum()
    with torch.autograd.set_detect_anomaly(True):
        a.backward()