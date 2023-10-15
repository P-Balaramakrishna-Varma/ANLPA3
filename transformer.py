import torch, math
import torch.nn as nn
from data import *
from tqdm import tqdm

# Embedding
class GlobalEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


class PostionalEmbedding(nn.Module):
    def __init__(self, d_model, max_length=800, device='cuda'):
        super().__init__()
        self.pe = gen_pe(max_length, d_model).to(device)
    
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


# Encoder
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim, expansion_factor * embed_dim),
                          nn.ReLU(),
                          nn.Linear(expansion_factor * embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, X):
        attention_out, _ = self.attention(X, X, X)  
        attention_residual_out = attention_out + X  
        norm1_out = self.dropout1(self.norm1(attention_residual_out)) 

        feed_fwd_out = self.feed_forward(norm1_out)
        feed_fwd_residual_out = feed_fwd_out + norm1_out 
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out)) 
        return norm2_out


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8):
        super().__init__()
        self.embedding_layer = GlobalEmbedding(vocab_size, embed_dim)
        self.positional_encoder = PostionalEmbedding(embed_dim)
        self.layers = nn.ModuleList([EncoderBlock(embed_dim, expansion_factor, n_heads) for _ in range(num_layers)])
    
    def forward(self, x):
        global_embed_out = self.embedding_layer(x)
        postional_embed_out = self.positional_encoder(x)
        embed_out = global_embed_out + postional_embed_out
        for layer in self.layers:
            out = layer(embed_out)
        return out  
    
    
# Decoder




if __name__ == '__main__':
    vocab_en = build_vocab_from_iterator(vocab_iterator("en"), specials=["<pad>", "<unk>"], min_freq=2)
    vocab_en.set_default_index(vocab_en["<unk>"])
    vocab_fr = build_vocab_from_iterator(vocab_iterator("fr"), specials=["<pad>", "<unk>", "<sot>", "<eot>"], min_freq=2)
    vocab_fr.set_default_index(vocab_fr["<unk>"])

    
    enocder = TransformerEncoder(len(vocab_en), 300, 2, 2, 3).to('cuda')    
    print(enocder)
    data = EN_Fr_Dataset('test', vocab_en, vocab_fr)
    dataloder = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False, collate_fn=custom_collate)
    for x1, x2, y in tqdm(dataloder):
        x1 = x1.to('cuda')
        out = enocder(x1)
    