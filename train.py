import torch
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from model import DecoderOnlyModel
from data import DataLoader

def train(config: dict, model: DecoderOnlyModel, data_loader: DataLoader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    max_iters = config['training']['max_iters']
    losses = []

    for iter in range(max_iters):
        xb, yb = data_loader.get_batch()
        _, loss = model(xb, yb)
        losses.append(loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (iter + 1) % 100 == 0:
          print(f"Step {iter + 1}: Loss = {loss.item()}")

        if (iter + 1) % 10000 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'iter': iter,
            }, f"checkpoints/model_iter_{iter+1}.pt")

    return losses

def plot_loss_curve(losses: list, save_path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train the LLM model")
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    # Create directories
    Path("checkpoints").mkdir(exist_ok=True)
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Initialize DataLoader and model
    data_loader = DataLoader(config)  # Create DataLoader instance with config
    model = DecoderOnlyModel(config['model'])

    # Train the model
    losses = train(config, model, data_loader)

    # Plot and save the loss curve
    plot_loss_curve(losses, plots_dir / "loss_curve.png")

    print("Training completed. Loss curve saved in 'plots' directory.")

if __name__ == "__main__":
    main()
