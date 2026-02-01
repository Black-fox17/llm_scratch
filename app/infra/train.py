def loss_fn(logits, targets):
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.view(B*T, V),
        targets.view(B*T)
    )

import os
import torch
import matplotlib.pyplot as plt


def train_lm(
    *,
    model_fn,              # () -> nn.Module
    dataloader_fn,         # () -> object with next_train_batch / next_val_batch / vocab_size
    loss_fn,               # (logits, targets) -> loss
    optimizer_fn,          # (model) -> optimizer
    generate_fn=None,      # optional text generation hook
    device=None,

    epochs=20,
    train_steps=1000,
    val_steps=200,
    grad_clip=1.0,

    prompts=None,
    checkpoint_dir="checkpoints",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    data = dataloader_fn()
    model = model_fn().to(device)
    optimizer = optimizer_fn(model)

    train_losses = []
    val_losses = []

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        # ---------- TRAIN ----------
        model.train()
        total_train_loss = 0.0

        for step in range(train_steps):
            x, y = data.next_train_batch()
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            total_train_loss += loss.item()

            if step % 100 == 0:
                print(f"epoch {epoch} step {step} train_loss {loss.item():.4f}")

        avg_train_loss = total_train_loss / train_steps
        train_losses.append(avg_train_loss)

        # ---------- VALIDATION ----------
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for _ in range(val_steps):
                x, y = data.next_val_batch()
                x, y = x.to(device), y.to(device)

                logits = model(x)
                loss = loss_fn(logits, y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / val_steps
        val_losses.append(avg_val_loss)

        print(f"\nEPOCH {epoch}")
        print(f"train_loss {avg_train_loss:.4f}")
        print(f"val_loss   {avg_val_loss:.4f}\n")

        # ---------- OPTIONAL GENERATION ----------
        if generate_fn and prompts:
            for p in prompts:
                print(generate_fn(model, data, p))

        # ---------- CHECKPOINT ----------
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            },
            os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"),
        )

    # ---------- PLOT ----------
    plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Val"])
    plt.title("Training vs Validation Loss")
    plt.show()

    return model, train_losses, val_losses
