import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Hyperparameters
batch_size = 4
d_model = 512
context_length = 16
num_heads = 8
head_size = d_model // num_heads    # 每个头的维度
num_blocks = 12
dropout = 0.1
device  = 'cuda:1' if torch.cuda.is_available() else 'cpu'

class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    
    def forward(self, x):
        return self.ffn(x)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(d_model, head_size, bias=False)
        self.Wk = nn.Linear(d_model, head_size, bias=False)
        self.Wv = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(context_length, context_length)))
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        B ,T, D = x.shape
        q = self.Wq(x)  # q:[batch_size, context_length, head_size]
        k = self.Wk(x)
        v = self.Wv(x)

        output = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)   # [batch_size, context_length, context_length]
        output = output.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        output = F.softmax(output, dim=-1)  # [batch_size, context_length, context_length]
        output = self.Dropout(output)   # Optional:可选的
        output = output @ v # [batch_size, context_length, head_size]

        return output
    

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Attention() for _ in range(num_heads)])
        self.Wo = nn.Linear(d_model, d_model)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=1)  # [batch_size, context_length, d_model]
        output = self.Dropout(self.Wo(output))

        return output
    

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention()
        self.ffn = FeedForwardNetwork()

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x
    

class Model(nn.Module):
    def __init__(self, max_token_value=100256):
        super().__init__()
        self.vocab_linear = nn.Linear(d_model, max_token_value)
        self.te_lookup_table = nn.Embedding(max_token_value, d_model)
        ''' 
            *为一个解包功能，下面的代码等价于：
            TransformerBlock(),
            TransformerBlock(),
            ...
            TransformerBlock(),
            nn.LayerNorm(d_model)
        '''
        self.transformer_block = nn.Sequential(
            *([TransformerBlock() for _ in range(num_blocks)] + [nn.LayerNorm(d_model)])
        )

    def forward(self, x_batch, y_batch=None):   # x_batch:[batch_size, context_length]
        B, T, D = x_batch.shape

        pe_lookup_table =  torch.zeros(context_length, d_model, device=device)  # [context_length, d_model]
        position = torch.arange(0, context_length, dtype=torch.float, device=device).unsqueeze(1)  # [context_length, 1]
        
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, step=2).float() / d_model)
        pe_lookup_table[:, 0::2] = torch.sin(position * div_term)
        pe_lookup_table[:, 1::2] = torch.cos(position * div_term)

        output = self.te_lookup_table(x_batch) + pe_lookup_table
        output = self.transformer_block(output)
        logits = self.vocab_linear(output)

        if y_batch is not None:
            B, T, D = logits.shape
            logits_reshaped = logits.view(B*T, D)
            y_batch_reshaped = logits.view(B*T)

            loss = F.cross_entropy(input=logits_reshaped, target=y_batch_reshaped)
        else:
            loss = None
        return logits, loss
    

    def generate(self, x_batch, max_new_tokens=100,temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # x_batch: [batch_size, context_length]
            x_crop = x_batch[:, -context_length:]
            logits, _ = self(x_crop) # 或者self.forward(x_crop) [batch_size, context_length, vocab_size]
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probabilities = F.softmax(logits, dim=-1)
            predected_token = torch.multinomial(probabilities, num_samples=1)   # [batch_size, 1]
            x_batch = torch.cat((x_batch, predected_token),dim=-1)
        
        return x_batch
   


if __name__ == '__main__':
    model = Model()
    x = torch.randn(batch_size, context_length)
    # 前向传播
    output = model(x)

    # 打印输入和输出形状
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)