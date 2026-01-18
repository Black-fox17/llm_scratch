import torch
import torch.nn.functional as F

def init_state(batch_size, hidden_size, num_layers, device):
    h = [torch.zeros(batch_size, hidden_size, device=device) for _ in range(num_layers)]
    c = [torch.zeros(batch_size, hidden_size, device=device) for _ in range(num_layers)]
    return (h, c)

@torch.no_grad()
def sample(model, data, seed, device = "cpu", length=120, temperature=1.0, greedy=False):
    state = init_state(1, model.hidden_size, model.num_layers, device)
    out = seed

    for ch in seed:
        idx = torch.tensor([data.stoi[ch]], device=device)
        _, state = model(idx, state)

    idx = torch.tensor([data.stoi[seed[-1]]], device=device)

    for _ in range(length):
        logits, state = model(idx, state)
        logits = logits / temperature

        if greedy:
            idx = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1)
        
        # Ensure idx is 1D for the next iteration to prevent dimension mismatch
        if idx.dim() > 1:
            idx = idx.view(-1)

        out += data.itos[idx.item()]


    return out

