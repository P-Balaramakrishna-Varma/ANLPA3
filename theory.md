## What is the purpose of self-attention, and how does it facilitate capturing dependencies in sequences?
The role of self-attention is meaning aggregation from context. Which in turn enables us to create contextual embeddings.    
It plays the same role as an LSTM does in a Elmov. LSTM (RNNs) are traditionally used in NLP to aggregate meaning.   
There two drawbacks of LSTM which are both addressed by self-attention. 
- Time taken. LSTM works sequentially word by word which increases the time taken, this becomes more significant as we increase layers.
- LSTM assumes that language is linear is structure. 

Self-attention is able aggregate meaning for all words in a context at once. Which makes it time efficient.   
Self-attention does not assume any language structure it learns the language structure. Hence it can capture longer dependencies better.



