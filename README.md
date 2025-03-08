# Implementing Decoder only LLM Model (GPT style) from scratch with PyTorch
- Pretraining a LLM model for Text generation, used Salesforce/wikitext for training. The model was trained for 30000 iterations with a batch size of 8 for ~3 hours on a 16GB Tesla P100 (Kaggle Free gpu support). The training loss is around 3.7. Used adam optimizer with a learning rate of 5e-4. After training, the model is generating english with understandable grammer.


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

## Model Details
```
n_embd = 512
vocab_size = 50257
n_layers = 6
n_heads = 8
block_size = 512 # number to previous tokens to attend to perform attention
batch_size = 8
learning rate = 5e-4
```

# Loss curves 
- The x-axis represents iterations in hundreds. The model was trained for a total of 30,000 iterations.
  
Train Loss             |  Test loss
:-------------------------:|:-------------------------:
![](https://github.com/SSahas/Implementing-GPT-From-Scratch/blob/add_eval/assets/train.png)  |  ![](https://github.com/SSahas/Implementing-GPT-From-Scratch/blob/add_eval/assets/test.png)




# Sample Generations
> *This is used for its purpose . The castle has its most extensive military value , with its new weapons and the ability to draw guns against and destroy obstacles ,
but it has always been used for long - duration.*

> *Once there was no threat to the United States who are expecting asylum to the United States government . The National Hurricane Center issued the same day the agency requested them to the Washington National Weather Service agencies at any request . By 1997 , the agency also considered the agency had a $ 20 , 000 fine ( equivalent to $ 15 , 060 , 061 in 2016 ) for an upcoming hurricane.*

> *This is to be called the " great leader of all the major things and the most beautiful leader of all the time " he is " not so happy " if he and his co - workers will be able to accomplish the truth they are in vain when him to death .*






# References 
- [Andrej karpathy-nanoGPT](https://github.com/karpathy/nanoGPT)
- [t5-pytorch](https://github.com/conceptofmind/t5-pytorch)
- [nanoT5](https://github.com/PiotrNawrot/nanoT5)
