import torch
from model import GPT
from common import *

device = get_device()
model = GPT().to(device)

start = "\n"
max_new_tokens = 256

# load the saved parameters
model.load_state_dict(torch.load("model_parameters.pth", weights_only=True))

# set the model to eval mode
model.eval()

# create a starting context
context = torch.tensor(encode(start), dtype=torch.long, device=device)[None, ...]

while True:
    # generate from the model
    output = model.generate(context, max_new_tokens=max_new_tokens)

    # decode and print result
    generated = output[0].tolist()
    decoded = decode(generated)
    print(decoded)

    context = output
