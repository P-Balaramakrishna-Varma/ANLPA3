The below are all the hyperparameters for training

```py
## Hyperparameters    
epochs = 2            # 10
r = 0.001             # acc to other
batch_size = 32       # fixed time optimality
    
num_layers = 2        # fixed as assingment
expansion_factor = 2  # strong enough no need to experiment
n_heads = 4           # vairable
embed_dim = 300       # variable
```

We will experiment with ```n_head``` and ```embed_dim```.    
- ```n_head``` takes either 4, 8   
- ```embed_dim``` takes either 100, 300  