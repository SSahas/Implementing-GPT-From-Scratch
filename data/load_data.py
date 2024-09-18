import torch
from datasets import load_dataset
from typing import Tuple

class DataLoader:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = self._load_data()

    def _load_data(self) -> torch.Tensor:
        ds = load_dataset("SSahas/llm_pretrain_dataset")
        return torch.tensor(ds['train']['input_ids'], dtype=torch.long, device=self.device)

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self.data) - self.config['model']['block_size'], (self.config['training']['batch_size'],))
        x = torch.stack([self.data[i:i+self.config['model']['block_size']] for i in ix])
        y = torch.stack([self.data[i+1:i+self.config['model']['block_size']+1] for i in ix])
        return x.to(self.device), y.to(self.device)
    




