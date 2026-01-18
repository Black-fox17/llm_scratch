import torch
import torch.nn.functional as F
from models.stacked import StackedLSTM
from data.batchloader import CharLMDataLoader
import os
from utils import init_state,sample

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    data = CharLMDataLoader(
        path="data/input.txt",
        batch_size=64,
        seq_len=64
    )

    model = StackedLSTM(
        vocab_size=data.vocab_size,
        hidden_size=256,
        num_layers=2,
        device=device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    epochs = 20
    grad_clip = 1.0

    prompts = ["The ", "Hello", "What ", "I "]

    for epoch in range(epochs):
        total_loss = 0.0
        steps = 1000

        for step in range(steps):
            x, y = data.next_train_batch()
            x = x.to(device)
            y = y.to(device)

            state = init_state(x.size(0), model.hidden_size, model.num_layers, device)

            loss = 0.0
            for t in range(x.size(1)):
                logits, state = model(x[:, t], state)
                loss += F.cross_entropy(logits, y[:, t])

            loss /= x.size(1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        avg_loss = total_loss / steps
        print(f"\nEPOCH {epoch} avg loss {avg_loss:.4f}\n")

        for p in prompts:
            print(sample(model, data, p, device, greedy=True))

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(
            {
                'epoch': epoch,
                'model_state': {
                    'embed': model.embed,
                    'Wy': model.Wy,
                    'by': model.by,
                    'layers': [(l.W, l.b) for l in model.layers]
                }
            },
            f"checkpoints/epoch_{epoch}.pt"
        )


if __name__ == "__main__":
    train()
