import torch
import torch.nn as nn
from torch.nn import functional as F


# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "mps" if torch.backends.mps.is_available() else "cpu"
eval_iters = 200


torch.manual_seed(1337)


with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Set up vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Set up vocab encoder and decoder
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()
    model.train()

    return out


# Bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, context, targets=None):
        logits = self.token_embedding_table(context)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, context, max_new_tokens):
        # context is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, _ = self(context)
            # focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilites
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            next_char = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            context = torch.cat((context, next_char), dim=1)

        return context


model = BigramLanguageModel(vocab_size).to(device)

# create a Pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)[0].tolist()
decoded = decode(generated)
print(decoded)
