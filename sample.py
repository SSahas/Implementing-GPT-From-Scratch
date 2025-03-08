import torch
import json
import argparse
import tiktoken
from model import DecoderOnlyModel



def load_model(config_path: str, checkpoint_path: str) -> DecoderOnlyModel:
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = DecoderOnlyModel(config['model'])
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def generate_text(model: DecoderOnlyModel, tokenizer, prompt: str, max_new_tokens: int, temperature: float = 1.0) -> str:
    device = next(model.parameters()).device
    input_ids = tokenizer.encode_ordinary(prompt)
    input_ids = torch.tensor([input_ids])
    inputs_ids = input_ids.unsqueeze(0)
    input_ids = input_ids[:, :-1]
       
    with torch.no_grad():
        output_ids = model.generate(
            input_ids.to(device),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_k=50,
        )
    output_ids = output_ids.tolist()[0]
    
    generated_text = tokenizer.decode(output_ids)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained LLM model")
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to the configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to start generation")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (higher = more random)")
    args = parser.parse_args()

    # Load the model
    model = load_model(args.config, args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    enc = tiktoken.get_encoding("gpt2")


    # Generate text
    generated_text = generate_text(model, enc, args.prompt, args.max_tokens, args.temperature)

    print("Generated text:")
    print(generated_text)

if __name__ == "__main__":
    main()
