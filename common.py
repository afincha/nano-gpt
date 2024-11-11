import torch

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2


def get_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"using device: {device}")
    return device


# load training data file
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
