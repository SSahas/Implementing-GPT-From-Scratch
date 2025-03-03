# Implementing Decoder only LLM Model (GPT style) from scratch with PyTorch
- Pretraining a LLM model for Text generation, used Salesforce/wikitext for training. The model was trained for 30000 iterations with a batch size of 8 for ~2.5 hours on Tesla P100 (Kaggle Free gpu support). The training loss is around 3.5. Used adam optimizer with a learning rate of 5e-4. After training, the model is producing little reasonable english, can be trained for more time with bigger n_embd and block size for better generation.

## Model Details
```
n_embd = 512
vocab_size = 28144
n_layers = 6
n_heads = 8
block_size = 512 # number to previous tokens to attend to perform attention
batch_size = 8
learning rate = 5e-4
```

- To train the model , clone the repository 

```
git clone https://github.com/SSahas/Implementing-GPT-From-Scratch.git
```
## Training 
- To train the model
```
python train.py --config config/config.json
```

## Inference
- To generate text using a trained model
```
python sample.py --model_path path/to/saved/model --prompt "Your prompt here"
```

## Load Dataset for training

- To create data file and the tokenizer for traning the model, run [Tokenizer_Training notebook](https://github.com/SSahas/Implementing-LLM-From-Scratch/blob/main/Tokenizer_Training.ipynb). Added a special token "<|EOS|>" as bos and eos token.Used [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) for pretrainig the model and training the tokenizer.





# Train Loss Curve 
![cross_entropy_loss_curve](https://github.com/SSahas/Implementing-LLM-From-Scratch/blob/main/assets/loss_curve.png)

# References 
- [Andrej karpathy-nanoGPT](https://github.com/karpathy/nanoGPT)
- [t5-pytorch](https://github.com/conceptofmind/t5-pytorch)
- [nanoT5](https://github.com/PiotrNawrot/nanoT5)

