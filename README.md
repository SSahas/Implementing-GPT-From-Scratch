# Implementing Decoder only Model (GPT style) from scratch with Pytorch
- Pretraining a LLM the model for Text generation, used Salesforce/wikitext for trainig. The model was trained for 30000 iterations with batch size of 8 for ~2.5 hours on Tesla P100 (Kaggle Free gpu support). The training loss is around 3.5. Used adam optimizer with a learning rate of 5e-4. After Traning the model is able to produce little reasonable english, can be trained for more time with bigger n_embd and block size for better reuslts and finetuning for other downstream tasks.

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

