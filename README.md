# Implementing Decoder only Model (GPT style) from scratch with PyTorch
- Pretraining a LLM the model for Text generation, used Salesforce/wikitext for trainig. The model was trained for 30000 iterations with a batch size of 8 for ~2.5 hours on Tesla P100 (Kaggle Free gpu support). The training loss is around 3.5. Used adam optimizer with a learning rate of 5e-4. After Training the model is able to produce a little reasonable english, can be trained for more time with bigger n_embd and block size for better generation and finetuning for other downstream tasks.

## Model Details
- n_embd = 512
- n_layers = 6
- vocab size = 28144
- n_heads = 8
- block_size = 512 
- batch_size = 8
- model parameters - 48 Million
- optimizer - adam
- learning rate - 5e-4

# Loss Curve 
![cross_entropy_loss_curve](https://github.com/user-attachments/assets/70396741-6fab-4ca0-96b6-a1e32ca49826)

# References 

- [Andrej karpathy-nanoGPT](https://github.com/karpathy/nanoGPT)
- [t5-pytorch](https://github.com/conceptofmind/t5-pytorch)

