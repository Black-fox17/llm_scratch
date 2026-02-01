import torch
import torch.nn.functional as F
from models.stacked import StackedLSTM
from data.batchloader import CharLMDataLoader
import os
from utils import sample

def load_model(checkpoint_path, data):  
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 54
    hidden_size = 256
    num_layers = 2
    
    model = StackedLSTM(vocab_size, hidden_size, num_layers, device)
    
    model.embed = checkpoint['model_state']['embed']
    model.Wy = checkpoint['model_state']['Wy']
    model.by = checkpoint['model_state']['by']
    
    for i, (W, b) in enumerate(checkpoint['model_state']['layers']):
        model.layers[i].W = W
        model.layers[i].b = b
    
    return model, hidden_size, num_layers


# def generate_completion(model, data, prompt, max_length=100, temperature=0.8):
#     full_text = sample(model, data, prompt, length=max_length, temperature=temperature)
#     return full_text[len(prompt):]


if __name__ == '__main__':
    data = CharLMDataLoader(
        path='data/input.txt',
        batch_size=1,
        seq_len=1
    )
    
    
    checkpoint_path = "checkpoints/best.pt"
    model, hidden_size, num_layers = load_model(checkpoint_path, data)
    
    seed_texts = [
        "The ",
        "Once upon a time",
        "In the beginning",
        "Hello"
    ]
    
    print("SAMPLE GENERATIONS")
    
    for seed in seed_texts:
        print(f"Seed: '{seed}'")
        generated = sample(model, data, seed, length=200, temperature=0, greedy=True)
        print(generated)
        print("\n")
