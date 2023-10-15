import torch

from data import *
from transformer import *



def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for x1, x2, y in tqdm(dataloader):
        # data loading
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        
        # Compute prediction and loss
        pred = model(x1, x2)
        pred = pred.reshape(-1, pred.shape[-1])
        y = y.reshape(-1)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x1, x2, y in tqdm(dataloader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            pred = model(x1, x2)
            pred = pred.reshape(-1, pred.shape[-1])
            y = y.reshape(-1)
            test_loss += loss_fn(pred, y).item()

    num_batches = len(dataloader)
    test_loss /= num_batches
    return test_loss
    


if __name__ == "__main__":
    vocab_en = build_vocab_from_iterator(vocab_iterator("en"), specials=["<pad>", "<unk>"], min_freq=2)
    vocab_en.set_default_index(vocab_en["<unk>"])
    vocab_fr = build_vocab_from_iterator(vocab_iterator("fr"), specials=["<pad>", "<unk>", "<sot>", "<eot>"], min_freq=2)
    vocab_fr.set_default_index(vocab_fr["<unk>"])

    model = Transformer(300, len(vocab_en), len(vocab_fr), num_layers=2, expansion_factor=4, n_heads=3).to('cuda')
    print(model)
    
    data = EN_Fr_Dataset('test', vocab_en, vocab_fr)
    dataloader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=False, collate_fn=custom_collate)
    # for x1, x2, y in tqdm(dataloader): 
    #    x1, x2, y = x1.to('cuda'), x2.to('cuda'), y.to('cuda')
    #    model(x1, x2)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loop(dataloader, model, loss_fn, optimizer, 'cuda')
    loss = test_loop(dataloader, model, loss_fn, 'cuda')
    print(loss)