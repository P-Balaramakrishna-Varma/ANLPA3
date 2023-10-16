## Code information
- Results
    - hyperparameters.md contains the hyperparameters experimentation report.
    - Loss Plots
        - heads_4_d_100.png loss plot when n_heads is 4 and embed_dim is 100.
        - heads_4_d_300.png loss plot when n_heads is 4 and embed_dim is 300.
        - heads_6_d_100.png loss plot when n_heads is 6 and embed_dim is 100.
        - heads_6_d_300.png loss plot when n_heads is 6 and embed_dim is 300.
    - BLEU score
        - train_bleu.txt contains the bleu scores of the training set per sentence.
        - test_bleu.txt contains the bleu scores of the test set per sentence.
        - dev_bleu.txt contains the bleu scores of the dev set per sentence. (done only for 10,000 sentences (time constraints))
- Src
    - bleu.py is used to generate the bleu scores of a translation.
    - data.py contains the code related to data processing, tokenization, dataset, collate_fn.
    - multihead_attention.py contains multihead attention code.
    - transformer.py contain the implementation of the transformer model.
    - train.py is used to train the transformer.
- environment.yaml contains the conda environment used.
- theory.md contains answers to the theory questions.

## References
- https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
- PyTorch documentation
- chatGPT