import os
import sys
import pickle
import torch
import tiktoken
from model import Model


# Hyperparameters
batch_size = 12
context_length = 16
max_iters = 200
learning_rate = 1e-3
eval_interval = 20
eval_iters = 5
device  = 'cuda:1' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)
# epochs = 2

checkpoint = torch.load('checkpoint.pt')
model = Model().to(device)
model.load_state_dict(state_dict=checkpoint)
model.eval()
model.to(device)

tokenizer = tiktoken.get_encoding('cl100k_base')

start = '你是谁'
start_ids = tokenizer.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long,device=device).unsqueeze(0)).to(device)

with torch.no_grad():
    y = model.generate(x=x, max_new_tokens=100, temperature=1.0)
    print(tokenizer.decode(y[0].tolist()))