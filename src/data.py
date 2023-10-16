import torchtext
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def vocab_iterator(language):
    if language == "en":
        model = "en_core_web_sm"
        extension = "en"
    elif language == "fr":
        model = "fr_core_news_sm"
        extension = "fr"
    else:
        assert False, "language must be 'en' or 'fr'"

    tokenizer = get_tokenizer("spacy", language=model)
    with open('../ted-talks-corpus/train.' + extension) as f:
        for line in f:
            tokens = tokenizer(line[:-1])    # remove the last character '\n'
            for token in tokens:
                yield [token]

def test_vocab_iterator():
    for token in vocab_iterator("en"):
        print(token)



def Gather_data(split):
    with open('../ted-talks-corpus/' + split + '.en') as f_en:
        with open('../ted-talks-corpus/' + split + '.fr') as f_fr:
            for line_en, line_fr in zip(f_en, f_fr):
                yield line_en, line_fr

def test_gather_data():
    for line_en, line_fr in Gather_data('test'):
        print(line_en, line_fr)

        

def generate_datases(split, en_vocab, fr_vocab):
    en_tokenizer = get_tokenizer("spacy", language='en_core_web_sm')
    fr_tokenizer = get_tokenizer("spacy", language='fr_core_news_sm')
    X = []
    for line_en, line_fr in Gather_data(split):
        en_tokens = en_tokenizer(line_en[:-1])    # remove the last character '\n'
        en_indices = [en_vocab[token] for token in en_tokens]
        fr_tokens = fr_tokenizer(line_fr[:-1])
        fr_indices = [fr_vocab[token] for token in fr_tokens]
        X.append((en_indices, fr_indices))
    return X

def test_generate_dataset():
    data = generate_datases('test')
    print(json.dumps(data, indent=4, ensure_ascii=False))
    
    
class EN_Fr_Dataset(Dataset):
    def __init__(self, split, vocab_en, vocab_fr):
        super().__init__()
        self.vocab_en = vocab_en
        self.vocab_fr = vocab_fr
        self.X = generate_datases(split, self.vocab_en, self.vocab_fr)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x1 = torch.tensor(self.X[idx][0])
        x2 = torch.tensor([self.vocab_fr['<sot>']] + self.X[idx][1])
        y = torch.tensor(self.X[idx][1] + [self.vocab_fr['<eot>']]) 
        return x1, x2, y
    
    
def custom_collate(batch):
    En = [sample[0] for sample in batch]
    Fr = [sample[1] for sample in batch]
    tr = [sample[2] for sample in batch]
    
    En = pad_sequence(En, batch_first=True)
    Fr = pad_sequence(Fr, batch_first=True)
    tr = pad_sequence(tr, batch_first=True)
    return En, Fr, tr


if __name__ == "__main__":
    vocab_en = build_vocab_from_iterator(vocab_iterator("en"), specials=["<pad>", "<unk>"], min_freq=2)
    vocab_en.set_default_index(vocab_en["<unk>"])
    vocab_fr = build_vocab_from_iterator(vocab_iterator("fr"), specials=["<pad>", "<unk>", "<sot>", "<eot>"], min_freq=2)
    vocab_fr.set_default_index(vocab_fr["<unk>"])
    print(vocab_fr['<sot>'], vocab_fr['<eot>'], vocab_fr['<unk>'])
    print(vocab_en['<pad>'], vocab_fr['<pad>'])
    
    data = EN_Fr_Dataset('test', vocab_en, vocab_fr)
    print(len(data))
    dataloder = torch.utils.data.DataLoader(data, batch_size=2, shuffle=False, collate_fn=custom_collate)
    for x1, x2, y in dataloder:
        print(x1)
        print(x2)
        print(y)
        print("\n\n\n")