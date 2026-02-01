import torch
import torch.nn.functional as F
from models.GPT import GPTSmall
from data.dataloader import CharLMDataLoader

def loss_fn(logits, targets):
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.view(B*T, V),
        targets.view(B*T)
    )

def generate(model, dataloader, start_text, length=200, temperature = 0.8,top_k = None, top_p = None ):
    model.eval()
    device = next(model.parameters()).device
    idx = [dataloader.stoi[c] for c in start_text]
    generated = idx.copy()

    for _ in range(length):
        x = torch.tensor([generated[-dataloader.seq_len:]], device=device)
        logits = model(x)  # [1, seq_len, vocab_size]
        logits = logits[:, -1, :]  # take last token's logits: [1, vocab_size]
        
        if not temperature:
            next_id = torch.argmax(logits, dim=-1).item()
        else:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)  # softmax over vocab dimension
        
            # top-k
            if top_k is not None:
                v, _ = torch.topk(probs, top_k, dim=-1)
                cutoff = v[:, -1].unsqueeze(-1)
                probs = torch.where(probs < cutoff, torch.zeros_like(probs), probs)
        
            # top-p (nucleus)
            if top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum > top_p
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = False
                sorted_probs[mask] = 0.0
                probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
        
            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_id = torch.multinomial(probs, 1).item()
        
        generated.append(next_id)


    model.train()
    return "".join([dataloader.itos[i] for i in generated])


def train(d_model, n_heads, n_layers, block_size, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = CharLMDataLoader("data/input.txt", batch_size, block_size)
    vocab_size = data.vocab_size
    model = GPTSmall(vocab_size, d_model, n_heads, n_layers, block_size)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    grad_clip = 1.0

    epochs = 20
    train_steps = 1000
    val_steps = 200

    train_losses = []
    val_losses = []
    prompts = ["The ", "interesting", "But ", "It  "]

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for step in range(train_steps):
            x, y = data.next_train_batch()
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_train_loss += loss.item()

            if step % 100 == 0:
                print(f"epoch {epoch} step {step} train_loss {loss.item():.4f}")

        avg_train_loss = total_train_loss / train_steps
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for _ in range(val_steps):
                x, y = data.next_val_batch()
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = loss_fn(logits, y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / val_steps
        val_losses.append(avg_val_loss)

        print(f"\nEPOCH {epoch}")
        print(f"train_loss {avg_train_loss:.4f}")
        print(f"val_loss   {avg_val_loss:.4f}\n")

        for p in prompts:
            print(generate(model, data, p, length=150))

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            },
            f"checkpoints/gpt_epoch_{epoch}.pt"
        )
    plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train Loss", "Val Loss"])
    plt.title("GPT Training vs Validation Loss")
    plt.show()



train(d_model = 256, n_heads=8, n_layers=6, block_size=64, batch_size=64)