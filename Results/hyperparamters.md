# Experimentation

The below are all the hyperparameters for training

```py
## Hyperparameters    
epochs = 10           # 10
lr = 0.001            # acc to other parameters
batch_size = 32       # fixed time optimality
    
num_layers = 2        # fixed as assingment
expansion_factor = 2  # strong enough no need to experiment
n_heads = 4           # vairable (4, 8)
embed_dim = 300       # variable (100, 300)
```

We will experiment with ```n_head``` and ```embed_dim``` keeping all other hyperparamters constant as they are insignificant.    
- ```n_head``` takes either 4, 8   
- ```embed_dim``` takes either 100, 300  

<br>

## Results

| ```n_head``` | ```embed_dim``` |  ```Test Loss``` |
| -------------- | ---------------- | ---------- |