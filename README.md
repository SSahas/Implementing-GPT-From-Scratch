# Implementing Decoder only Model (GPT style) from scratch with Pytorch
- PreTraining a LLM the model for Text generation.
- Trained the model on SalesForce/wikitext.
- Trained WordPeice Tokenizer using transformers

## Model Details
- n_embd = 512
- n_layers = 6
- vocab size = 28144
- n_heads = 8
- block_size = 512 
- batch_size = 8
- model parameters - 47987712

# Loss Curve 
![cross_entropy_loss_curve](https://github.com/user-attachments/assets/70396741-6fab-4ca0-96b6-a1e32ca49826)

# References 

- [Andrej karpathy YouTube Series](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [t5-pytorch](https://github.com/conceptofmind/t5-pytorch)

