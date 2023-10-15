import torch

from data import *
from transformer import *



def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    vocab_en = build_vocab_from_iterator(vocab_iterator("en"), specials=["<pad>", "<unk>"], min_freq=2)
    vocab_en.set_default_index(vocab_en["<unk>"])
    vocab_fr = build_vocab_from_iterator(vocab_iterator("fr"), specials=["<pad>", "<unk>", "<sot>", "<eot>"], min_freq=2)
    vocab_fr.set_default_index(vocab_fr["<unk>"])

    model = Transformer(300, len(vocab_en), len(vocab_fr), num_layers=2, expansion_factor=4, n_heads=3).to('cuda')
    print(model)
    
    data = EN_Fr_Dataset('test', vocab_en, vocab_fr)
    dataloder = torch.utils.data.DataLoader(data, batch_size=8, shuffle=False, collate_fn=custom_collate)
    for x1, x2, y in tqdm(dataloder): 
       x1, x2, y = x1.to('cuda'), x2.to('cuda'), y.to('cuda')
       model(x1, x2)