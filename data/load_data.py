from datasets import load_dataset, concatenate_datasets 
import os
from tqdm import tqdm
import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")

def remove_duplicates(dataset):
    """
    Remove duplicate entries from the dataset.
    
    Args:
        dataset: Input dataset
    
    Returns:
        dataset: Dataset with duplicates removed
    """
    unique_text = set()
    
    def dedup_func(x):
        if x['text'] in unique_text:
            return False
        unique_text.add(x['text'])
        return True

    return dataset.filter(dedup_func, load_from_cache_file=False, num_proc=1)




def clean_text(text):
        text = text.replace("@-@", "-")
        text = text.replace("@.@", ".")
        text = text.replace("@,@", ",")
        text = text.replace("<unk>", "[UNK]")
        # text = text.replace("\n", "")
        text = text.replace("\'", "")
        text = text.replace("\\", "")
        text = text.replace(" '", "'")

        return text

def remove_short_sequences(x):

        if len(x['text']) > 94:
            return True
        else:
            return False 
        

def process(example):
        ids = enc.encode_ordinary(example['text']) 
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

def create_bin(dataset, split:str):

        arr_len = np.sum(dataset['len'], dtype = np.uint64)

        file_name = os.path.join(os.path.dirname(__file__), f"{split}.bin")

        dtype = np.uint16

        arr = np.memmap(file_name, dtype=dtype, mode='w+', shape=(arr_len,))

        total_batches = 40

        idx = 0

        for batch_idx in tqdm(range(total_batches), desc=f'writing {file_name}'):
            # Batch together samples for faster write
            batch = dataset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()




def main():

        print("Creating .bin files for training the model")
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

        ds = ds.filter(lambda example: len(example['text']) > 0)

        ds = remove_duplicates(ds)

        ds = ds.map(lambda example: {'text': clean_text(example['text'])})

        ds = ds.filter(remove_short_sequences, load_from_cache_file= False, num_proc=1)

        num_proc = 1


        tokenized = ds.map(
            process,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        train_data = concatenate_datasets([tokenized['train'], tokenized['validation']]) #pretraining

        test_data =  tokenized['test']

        create_bin(train_data, split = "train")
        create_bin(test_data, split = 'test')


if __name__ == "__main__":
    main()










    


    







