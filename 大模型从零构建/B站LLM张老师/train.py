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

with open('data.csv', 'r') as f:
    text = f.read()

tokenizer = tiktoken.get_encoding('cl100k_base')
tokenized_text = tokenizer.encode(text)
tokenized_text = torch.tensor(data=tokenized_text, dtype=torch.long, device=device)


p_size = (len(tokenized_text) * 0.9)
train_data = tokenized_text[:p_size]
valid_data = tokenized_text[p_size:]

model = Model().to(device)

def get_batch(split):
    data = train_data if split == 'train' else valid_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))    # (batch_size,)加逗号只是这是一个tuple
    x = torch.stack([data[idx:idx+context_length] for idx in idxs]) # batch_size, context_length
    y = torch.stack([data[idx+1:idx+context_length+1] for idx in idxs]) # batch_size, context_length

    return x, y

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}

    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out




optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for step in range(max_iters):

    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print('Step', step, 'Train Loss', round(losses['train'].item(), 3), 'Valid Loss', round(losses['valid'].item(), 3))
        

    x, y = get_batch('train')
    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


torch.save(model.state_dict(), 'model.bin')


