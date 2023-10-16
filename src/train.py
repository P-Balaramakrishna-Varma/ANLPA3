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
    ## Hyperparameters
    device = 'cuda'
    
    epochs = 2  # 10
    lr = 0.001  # for a setting see the plot and decide
    batch_size = 32  # fixed time optimality
    
    num_layers = 2  # fixed as assingment
    expansion_factor = 2  # strong enough no need to experiment
    n_heads = 4       # vairable
    embed_dim = 300   # variable
    
    
    ## creating vocabularies
    vocab_en = build_vocab_from_iterator(vocab_iterator("en"), specials=["<pad>", "<unk>"], min_freq=2)
    vocab_en.set_default_index(vocab_en["<unk>"])
    vocab_fr = build_vocab_from_iterator(vocab_iterator("fr"), specials=["<pad>", "<unk>", "<sot>", "<eot>"], min_freq=2)
    vocab_fr.set_default_index(vocab_fr["<unk>"])
    assert vocab_en['<pad>'] == 0
    assert vocab_fr['<pad>'] == 0
    print("Voabulary created")

    test_data = EN_Fr_Dataset('test', vocab_en, vocab_fr)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    
    dev_data = EN_Fr_Dataset('dev', vocab_en, vocab_fr)
    dev_dataloader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    
    train_data = EN_Fr_Dataset('train', vocab_en, vocab_fr)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    print("Data loaded")
    
    
    # model, looss
    model = Transformer(embed_dim, len(vocab_en), len(vocab_fr), num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   
   
    # training
    losses = []
    for _ in tqdm(range(epochs)):
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        loss = test_loop(dev_dataloader, model, loss_fn, device)
        losses.append(loss)
        print(loss)
    
    print("\n\n")
    print(losses)
    final_loss = test_loop(test_dataloader, model, loss_fn, device)
    print(final_loss)