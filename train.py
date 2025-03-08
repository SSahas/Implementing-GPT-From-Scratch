import torch
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import tiktoken

from model import DecoderOnlyModel
from data import load_data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = f"{os.getcwd()}/Implementing-GPT-From-Scratch/data"


def get_batch(split, config):

    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'),
                         dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'test.bin'),
                         dtype=np.uint16, mode='r')

    ix = torch.randint(
        len(data) - config['model']['block_size'], (config['training']['batch_size'],))
    x = torch.stack([torch.from_numpy(
        (data[i:i+config['model']['block_size']]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(
        (data[i+1:i+1+config['model']['block_size']]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(config, model):
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(config['training']['eval_iters'])
        for k in range(config['training']['eval_iters']):
            X, Y = get_batch(split, config)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(config: dict, model: DecoderOnlyModel):

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['training']['learning_rate'])

    max_iters = config['training']['max_iters']
    train_losses = []
    test_losses = []

    for iter in range(max_iters):
        xb, yb = get_batch('train', config=config)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (((iter + 1) % 100) == 0) or ((iter + 1) == max_iters):
            losses = estimate_loss(config, model)
            print(
                f"Step {iter + 1}: Train Loss = {losses['train']:.4f}, Eval loss = {losses['test']:.4f}")
            train_losses.append(losses['train'])
            test_losses.append(losses['test'])

        if (iter + 1) == max_iters:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'iter': iter,
            }, f"{os.getcwd()}/checkpoints/model_iter_{iter+1}.pt")

    return train_losses, test_losses


def plot_loss_curve(losses: list, split: str):

    save_path = f"{os.getcwd()}/plots/{split}.png"
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(f'{split} Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train the LLM model")
    parser.add_argument("--config", type=str, default="config/config.json",
                        help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    # Create directories
    Path("checkpoints").mkdir(exist_ok=True)
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    model = DecoderOnlyModel(config['model'])

    # Train the model
    train_losses, test_losses = train(config, model)

    # Plot and save the loss curve
    plot_loss_curve(train_losses, "train")
    plot_loss_curve(test_losses, "test")

    print("Training completed. Model saved in checkpoints directory, Loss curves saved in 'plots' directory.")


if __name__ == "__main__":
    main()
