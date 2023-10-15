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


class PositionalEmbedding(nn.Module):
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
    def __init__(self, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8, device='cuda'):
        super().__init__()
        self.embedding_layer = GlobalEmbedding(vocab_size, embed_dim)
        self.positional_encoder = PositionalEmbedding(embed_dim, device=device)
        self.layers = nn.ModuleList([EncoderBlock(embed_dim, expansion_factor, n_heads) for _ in range(num_layers)])
    
    def forward(self, x):
        global_embed_out = self.embedding_layer(x)
        postional_embed_out = self.positional_encoder(x)
        embed_out = global_embed_out + postional_embed_out
        for layer in self.layers:
            out = layer(embed_out)
        return out  
    

    
# Decoder
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=2, n_heads=8):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.2)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(0.2)
        
        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim, expansion_factor * embed_dim),
                          nn.ReLU(),
                          nn.Linear(expansion_factor * embed_dim, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(0.2)
     
    def forward(self, query, key, val, mask):
        self_attention_out, _ = self.self_attention(query, query, query, attn_mask=mask)  
        self_attention_residual_out = self_attention_out + query  
        norm1_out = self.dropout1(self.norm1(self_attention_residual_out)) 


        cross_attention_out, _ = self.cross_attention(norm1_out, key, val)
        corss_attention_residual_out = cross_attention_out + norm1_out
        norm2_out = self.dropout2(self.norm2(corss_attention_residual_out))

        feed_fwd_out = self.feed_forward(norm2_out)
        feed_fwd_residual_out = feed_fwd_out + norm2_out 
        norm3_out = self.dropout3(self.norm3(feed_fwd_residual_out)) 
        return norm3_out


class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, num_layers=2, expansion_factor=2, n_heads=8, device='cuda'):
        super().__init__()
        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(embed_dim, device=device)
        self.layers = nn.ModuleList([DecoderBlock(embed_dim, expansion_factor, n_heads) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.softmax = nn.Softmax(dim=2)
        # Possible to add a dropout layer here

    def forward(self, x, enc_out, mask):
        global_embd_out = self.word_embedding(x)
        position_embd_out = self.position_embedding(x)
        out = global_embd_out + position_embd_out
        
        for layer in self.layers:
            out = layer(out, enc_out, enc_out, mask) 
        
        pred = self.fc_out(out)
        # pred = self.softmax(pred)
        return pred


def make_trg_mask(trg, num_heads):
    batch_size, trg_len = trg.shape
    trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
        batch_size, trg_len, trg_len
    )
    result = torch.cat([trg_mask] * num_heads, dim=0)
    return ~result.bool() 



# Transformer
class Transformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size,num_layers=2, expansion_factor=4, n_heads=8, device='cuda'):
        super().__init__()
        self.num_heads = n_heads
        self.device = device
        self.target_vocab_size = target_vocab_size
        self.encoder = TransformerEncoder(src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, device=device)
        self.decoder = TransformerDecoder(target_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, device=device)
    
    def forward(self, src, trg):
        trg_mask = make_trg_mask(trg, self.num_heads).to(self.device)
        enc_out = self.encoder(src)
        outputs = self.decoder(trg, enc_out, trg_mask)
        return outputs






if __name__ == '__main__':
    vocab_en = build_vocab_from_iterator(vocab_iterator("en"), specials=["<pad>", "<unk>"], min_freq=2)
    vocab_en.set_default_index(vocab_en["<unk>"])
    vocab_fr = build_vocab_from_iterator(vocab_iterator("fr"), specials=["<pad>", "<unk>", "<sot>", "<eot>"], min_freq=2)
    vocab_fr.set_default_index(vocab_fr["<unk>"])

    model = Transformer(300, len(vocab_en), len(vocab_fr), num_layers=2, expansion_factor=4, n_heads=3).to('cuda')
    print(model)
    
    data = EN_Fr_Dataset('test', vocab_en, vocab_fr)
    dataloder = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, collate_fn=custom_collate)
    for x1, x2, y in tqdm(dataloder): 
        x1, x2, y = x1.to('cuda'), x2.to('cuda'), y.to('cuda')
        model(x1, x2)
        # mask = make_trg_mask(x2, 2)
        # print(mask)