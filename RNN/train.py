from models.stacked import StackedLSTM
from data.batchloader import CharLMDataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = CharLMDataLoader(
    path='data/input.txt',
    batch_size=50,
    seq_len=50
)

vocab_size = data.vocab_size
hidden_size = 128
num_layers = 2
seq_len = 50
batch_size = 50
lr = 2e-3
grad_clip = 5.0

model = StackedLSTM(vocab_size, hidden_size, num_layers)
params = model.parameters()

for epoch in range(20):
    state = init_state(batch_size, hidden_size, num_layers)

    for step in range(1000):
        x, y = data.next_train_batch()

        loss = 0.0
        states = [state]

        for t in range(seq_len):
            logits, state = model(x[:, t], state)
            loss += F.cross_entropy(logits, y[:, t])
            states.append(state)

        loss /= seq_len

        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        loss.backward()

        for p in params:
            p.grad.clamp_(-grad_clip, grad_clip)

        with torch.no_grad():
            for p in params:
                p -= lr * p.grad

        state = ([h.detach() for h in state[0]],
                 [c.detach() for c in state[1]])

        if step % 100 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
