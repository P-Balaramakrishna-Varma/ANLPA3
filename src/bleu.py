import torch
from data import *
from transformer import *
from torchtext.data.metrics import bleu_score


def get_trans_refer(gen_indices, y, fr_itos):
    translation = []
    for sent in gen_indices:
        sent = sent.tolist()[1:]  # remove <sot>
        sent = [fr_itos[i] for i in sent]
        trans = []
        for word in sent:
            if word != '<eot>':
                trans.append(word)
            else:
                break
        translation.append(trans)
    
    reference = []
    for sent in y:
        sent = sent.tolist()
        sent = [fr_itos[i] for i in sent]
        ref = []
        for word in sent:
            if word != '<eot>':
                ref.append(word)
            else:
                break
        reference.append([ref])
    return translation, reference



if __name__ == "__main__":
    vocab_en = build_vocab_from_iterator(vocab_iterator("en"), specials=["<pad>", "<unk>"], min_freq=2)
    vocab_en.set_default_index(vocab_en["<unk>"])
    vocab_fr = build_vocab_from_iterator(vocab_iterator("fr"), specials=["<pad>", "<unk>", "<sot>", "<eot>"], min_freq=2)
    vocab_fr.set_default_index(vocab_fr["<unk>"])
    assert vocab_en['<pad>'] == 0
    assert vocab_fr['<pad>'] == 0
    
    data = EN_Fr_Dataset('test', vocab_en, vocab_fr)
    subset = torch.utils.data.Subset(data, range(10)) # see size
    dataloder = torch.utils.data.DataLoader(subset, batch_size=2, shuffle=False, collate_fn=custom_collate)
    
    # load model in future
    model = Transformer(300, len(vocab_en), len(vocab_fr), num_layers=2, expansion_factor=4, n_heads=3).to('cuda')
    fr_itos = vocab_fr.get_itos()
    
    Translations = []
    References = []
    for x1, x2, y in tqdm(dataloder): 
        x1, x2, y = x1.to('cuda'), x2.to('cuda'), y.to('cuda')
        
        gen_indices = model.generate(x1, vocab_fr)
        translation, reference = get_trans_refer(gen_indices, y, fr_itos)
        
        Translations += translation
        References += reference


    for i in range(len(Translations)):
        score = bleu_score([Translations[i]], [References[i]])
        print(score)