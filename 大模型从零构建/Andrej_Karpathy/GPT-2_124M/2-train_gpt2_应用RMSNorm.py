from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import os
import inspect
import time

# ------------------------------------------------------------------------------



def rotate_half(x):
    """将输入张量的后半部分旋转"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, sin, cos):
    """应用旋转位置编码到查询和键向量"""
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码模块（支持多头）"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def _get_sin_cos(self, block_size, device):
        pos = torch.arange(block_size, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.sin(), emb.cos()

    def forward(self, q, k):
        batch_size, block_size, n_head, _ = q.size()
        sin, cos = self._get_sin_cos(block_size, q.device) # sin.shape = cos.shape = (block_size, head_dim)
        
        # 扩展维度适配多头 [batch_size, block_size, n_head, head_dim]
        sin = sin.view(1, block_size, 1, -1).expand(batch_size, -1, n_head, -1)
        cos = cos.view(1, block_size, 1, -1).expand(batch_size, -1, n_head, -1)
        
        return apply_rotary_pos_emb(q, k, sin, cos)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # 旋转位置编码
        self.rotary = RotaryPositionEmbedding(config.n_embd // config.n_head)

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # # 旋转位置编码
        q, k = self.rotary(q, k)
        
        # 将上面的几个步骤改为下面的一个步骤，即使用了flash-attention
        y = F.scaled_dot_product_attention(q, k, v,is_causal=True)


        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x): 
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024   # 句子长度
    vocab_size: int = 50257  # 字典大小
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            # wpe=nn.Embedding(config.block_size, config.n_embd),
            # rotary = RotaryPositionEmbedding(config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 在GPT2中，embedding的参数和最后的linear的参数是共享的，及同一个
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # 多个残差网络会使得方差越来越大，因此需要除以根号下残差网络的格式，这里乘以2为每一个layer中有两次残差
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) # w初始化为均值为0，方差为0.02
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)   # b初始化为0
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # F.cross_entropy不接受B参数，所以得展平，(B, T, vocab_size) -> (B*T, vocab_size), target: (B, T) -> (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'/home/hpclp/disk/q/models/gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            '/home/hpclp/disk/q/models/gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    # 自定义优化器
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # print(f"using fused AdamW: {fused_available}")
        use_fused = fused_available and device_type != "cpu"
        print(f"using fused AdamW: {use_fused}")


        # if master_process:
        #     print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        #     print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # # Create AdamW optimizer and use the fused version if it is available
        # fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # use_fused = fused_available and device_type == "cuda"
        # if master_process:
        #     print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    def generate(self, idx, max_new_tokens=32, temperature=1.0, top_k=None, top_p=None):
        # 输入文本的大小为(B, T)
        for _ in range(max_new_tokens):
            # 限定文本最长输入长度为block_size
            idx_cond = idx[:, -self.config.block_size:]
            # get the predictions
            logits, _ = self(idx_cond)  # logits shape (B, T, C)
            # 只取最后一步token的预测值，该值即为下一个未知token的预测值
            logits = logits[:, -1, :] # becomes (B, C)

            # 使用temperature来调整预测值
            logits = logits / temperature
            # 使用top_k来调整预测值
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # v.shape (B, k)，从logits:(B, C)中取出前k个大的值
                # v[:, [-1]]为取v的最后一列，即取v中最小的值，shape为 (B, 1)，下面的代码即将logits中小于v[:, [-1]]的值置为负无穷
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 使用top_p进行采样
            if top_p is not None and top_p < 1.0:
                # 获取softmax后的概率分布
                probs = F.softmax(logits, dim=-1)  # (B, C)
                # 对概率分布进行排序（从大到小）
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                # 计算累积概率
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # (B, C)
                # 找到累积概率超过top_p的位置，即第一个为True的值
                mask = cumulative_probs > top_p

                # 如果top_p非常小，导致每一词的概率都超过了top_p，那么mask会为全为True，因此下面两步保证了第一个一定为False，可以被选中
                mask[..., 1:] = mask[..., :-1].clone()  # 保留第一个超出top_p的位置
                mask[..., 0] = False  # 确保至少保留一个值
                indices_to_remove = sorted_indices[mask]

                for i in range(probs.size(0)):  # 遍历批次中的每个样本
                    probs[i, indices_to_remove] = 0
                # 重新归一化概率分布
                probs = probs / probs.sum(dim=-1, keepdim=True)

            else:
                # 如果没有使用top_p，则直接对logits应用softmax
                probs = F.softmax(logits, dim=-1) # (B, C)
                # 从分布中随机采样一个token

            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx



# ----------------------------------------------------------------------------------

'''
# 测试生成功能
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

num_return_sequences = 5
max_length = 30

# model = GPT.from_pretrained('/home/hpclp/disk/q/models/gpt2') # 使用预训练模型
model = GPT(GPTConfig())    # 使用本地参数
model.eval()
model.to(device)


import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)     # shape (B, T)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:

    with torch.no_grad():
        logits, _ = model(x)    # logits shape (B, T, vocab_size)
        logits = logits[:, -1, :]   # last time step logits (B, vocab_size)
        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)  # topk_indices即为对于的token

        ix = torch.multinomial(topk_probs, num_samples=1)   # shape (B, 1)
        xcol = torch.gather(topk_indices, 1, ix)    # 1应该是维度, shape (B, 1)

        x = torch.cat((x, xcol), dim=-1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print('>', decoded)'
'''


# ----------------------------------------------------------------------------------

'''
# 测试训练功能
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

import tiktoken
enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)
B, T = 4, 32
buf = torch.tensor(tokens[:B*T+1])
buf = buf.to(device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

model = GPT(GPTConfig())
model.to(device)
# logits, loss = model(x)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f'step {i}, loss {loss.item():.4f}')
    

print(logits.shape)
print(loss)
'''


# -----------------------------------------------------------------------------
import tiktoken
import numpy as np

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        enc = tiktoken.get_encoding('gpt2')
        with open('input.txt', 'r') as f:
            text = f.read()
        # text = text[:1000]
        self.tokens = enc.encode(text)
        self.tokens = torch.tensor(self.tokens)

        self.current_position = 0


    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T

        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


# ----------------------------------------------------------------------------------
# 测试训练功能
import time
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=16, T=1024)

# 使用TF32精度
torch.set_float32_matmul_precision('high')

# 50304为2的整数幂，通过将原本词汇表大小改为2的整数幂，加速运算
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)


# 训练前生成测试
model.eval()
enc = tiktoken.get_encoding('gpt2')
text = 'I am a language model, '
# text = text[:1000]
text_tokens = enc.encode(text)
text_tensor = torch.tensor(text_tokens).unsqueeze(0).to(device)
output = model.generate(text_tensor, max_new_tokens=32) # , top_k=50, top_p=0.9
print(enc.decode(output[0].tolist()))
model.train()
# import sys; sys.exit()

''''
 一行代码就能加速明显！但是运行前需要等待一段时间
 1、原本python编译器为一行一行进行运行，torch.compile()将整个模型编译为C++，C++编译器比python编译器快很多'
 2、减少了不必要的 GPU 读/写操作，从而提高了整体运行效率
 3、利用硬件特性生成高效代码。
 4、避免重复计算，优化计算图。
 5、提供多种优化模式以适应不同场景。
 6、使用缓存机制提升后续运行速度。
'''
# model = torch.compile(model)


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    # 使用该句及可使用bfloat16精度，但不是所有部分都变成了Bfloat16，为混合精度
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
        # import code; code.interact(local=locals())
    loss.backward()
    # 通过对梯度的裁剪，避免梯度爆炸问题，从而提高模型训练的稳定性。
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()    # 因为CPU主要是进行调度任务，CPU将一堆任务交给GPU后，GPU任务还没有执行完，而CPU已经执行完了
    t1 = time.time()    # 因为time.time()记的是CPU的时间，所以需要同步一下，等待GPU任务执行完后再计算时间
    d1 = (t1-t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)

    print(f'step {i}, loss {loss.item():.4f}, time {d1:.2f}ms, norm {norm:.4f}, tokens/sec {tokens_per_sec:.2f}')
    

# 训练后生成测试
model.eval()
enc = tiktoken.get_encoding('gpt2')
text = 'I am a language model, '
# text = text[:1000]
text_tokens = enc.encode(text)
text_tensor = torch.tensor(text_tokens).unsqueeze(0).to(device)
output = model.generate(text_tensor, max_new_tokens=32)
print(enc.decode(output[0].tolist()))


# print(logits.shape)
# print(loss)


'''
# ----------------------------------------------------------------------------------
# 测试训练功能，加上动态学习率
import time
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=16, T=1024)

# 使用TF32精度
torch.set_float32_matmul_precision('high')

# 50304为2的整数幂，通过将原本词汇表大小改为2的整数幂，加速运算
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50 

# 动态学习率
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device)

for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    # 使用该句及可使用bfloat16精度，但不是所有部分都变成了Bfloat16，为混合精度
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
        # import code; code.interact(local=locals())
    loss.backward()
    # 通过对梯度的裁剪，避免梯度爆炸问题，从而提高模型训练的稳定性。
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize()    # 因为CPU主要是进行调度任务，CPU将一堆任务交给GPU后，GPU任务还没有执行完，而CPU已经执行完了
    t1 = time.time()    # 因为time.time()记的是CPU的时间，所以需要同步一下，等待GPU任务执行完后再计算时间
    d1 = (t1-t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)

    print(f'step {step}, loss {loss.item():.4f}, lr {lr:.4e} ,time {d1:.2f}ms, norm {norm:.4f}, tokens/sec {tokens_per_sec:.2f}')
    

# print(logits.shape)
# print(loss)
'''


'''
# ----------------------------------------------------------------------------------
# 测试训练功能，加上动态学习率，加上梯度累加
import time
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288
B = 16
T = 1024
assert total_batch_size % (B * T) == 0
grad_accum_step = total_batch_size // (B * T)
print(f'total_batch_size {total_batch_size}')
print(f'grad_accum_step {grad_accum_step}')


train_loader = DataLoaderLite(B=B, T=T)

# 使用TF32精度
torch.set_float32_matmul_precision('high')

# 50304为2的整数幂，通过将原本词汇表大小改为2的整数幂，加速运算
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50 

# 动态学习率
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for _ in range(grad_accum_step):
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)

        # 使用该句及可使用bfloat16精度，但不是所有部分都变成了Bfloat16，为混合精度
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            # import code; code.interact(local=locals())
        # 因为原本计算损失时是会除以整个batch大小的，这里进行backward()累加会少除以grad_accum_step次，视频2小时44分处
        loss = loss / grad_accum_step
        loss_accum += loss.detach()
        loss.backward()     # 梯度会不停地累加grad_accum_step次

    # 通过对梯度的裁剪，避免梯度爆炸问题，从而提高模型训练的稳定性。
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize()    # 因为CPU主要是进行调度任务，CPU将一堆任务交给GPU后，GPU任务还没有执行完，而CPU已经执行完了
    t1 = time.time()    # 因为time.time()记的是CPU的时间，所以需要同步一下，等待GPU任务执行完后再计算时间
    d1 = (t1-t0) * 1000
    tokens_processed = grad_accum_step * train_loader.B * train_loader.T
    tokens_per_sec = tokens_processed / (t1 - t0)

    print(f'step {step}, loss {loss_accum.item():.4f}, lr {lr:.4e} ,time {d1:.2f}ms, norm {norm:.4f}, tokens/sec {tokens_per_sec:.2f}')
    

# print(logits.shape)
# print(loss)


'''

# # -----------------------------------------------------------------------------
# import tiktoken
# import numpy as np


# class DataLoaderLite:
#     def __init__(self, B, T, process_rank, num_processes, split):
#         self.B = B
#         self.T = T
#         self.process_rank = process_rank
#         self.num_processes = num_processes
#         assert split in {'train', 'val'}
#         enc = tiktoken.get_encoding('gpt2')
#         with open('input.txt', 'r') as f:
#             text = f.read()
#         # text = text[:1000]
#         self.tokens = enc.encode(text)
#         self.tokens = torch.tensor(self.tokens)

#         self.current_position = self.B * self.T * self.process_rank

#         # # get the shard filenames
#         # data_root = "edu_fineweb10B"
#         # shards = os.listdir(data_root)
#         # shards = [s for s in shards if split in s]
#         # shards = sorted(shards)
#         # shards = [os.path.join(data_root, s) for s in shards]
#         # self.shards = shards
#         # assert len(shards) > 0, f"no shards found for split {split}"
#         # if master_process:
#         #     print(f"found {len(shards)} shards for split {split}")
#         # self.reset()


#     def next_batch(self):
#         B, T = self.B, self.T
#         buf = self.tokens[self.current_position : self.current_position+B*T+1]
#         x = (buf[:-1]).view(B, T) # inputs
#         y = (buf[1:]).view(B, T) # targets
#         # advance the position in the tensor
#         self.current_position += B * T * self.num_processes
#         # if loading the next batch would be out of bounds, advance to next shard
#         if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
#             # self.current_shard = (self.current_shard + 1) % len(self.shards)
#             # self.tokens = load_tokens(self.shards[self.current_shard])
#             self.current_position = B * T * self.process_rank
#         return x, y


# # -----------------------------------------------------------------------------
# # helper function for HellaSwag eval
# # takes tokens, mask, and logits, returns the index of the completion with the lowest loss

# def get_most_likely_row(tokens, mask, logits):
#     # evaluate the autoregressive loss at all positions
#     shift_logits = (logits[..., :-1, :]).contiguous()
#     shift_tokens = (tokens[..., 1:]).contiguous()
#     flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
#     flat_shift_tokens = shift_tokens.view(-1)
#     shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
#     shift_losses = shift_losses.view(tokens.size(0), -1)
#     # now get the average loss just for the completion region (where mask == 1), in each row
#     shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
#     masked_shift_losses = shift_losses * shift_mask
#     # sum and divide by the number of 1s in the mask
#     sum_loss = masked_shift_losses.sum(dim=1)
#     avg_loss = sum_loss / shift_mask.sum(dim=1)
#     # now we have a loss for each of the 4 completions
#     # the one with the lowest loss should be the most likely
#     pred_norm = avg_loss.argmin().item()
#     return pred_norm

# -----------------------------------------------------------------------------

# # -----------------------------------------------------------------------------
# # simple launch:
# # python train_gpt2.py
# # DDP launch for e.g. 8 GPUs:
# # torchrun --standalone --nproc_per_node=8 train_gpt2.py

# # run the training loop
# from torch.distributed import init_process_group, destroy_process_group
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist

# # set up DDP (distributed data parallel).
# # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
# ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

# if ddp:
#     # use of DDP atm demands CUDA, we set the device appropriately according to rank
#     assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
#     init_process_group(backend='nccl')
#     '''
#     例如，在两台机器的分布式训练中（每台机器 4 个 GPU），第一台机器的 RANK 为 0, 1, 2, 3，
#     第二台机器的 RANK 为 4, 5, 6, 7，但每台机器的 LOCAL_RANK 都是 0, 1, 2, 3
#     而 WORLD_SIZE 为 8，即总共有 8 个 GPU。
#     '''
#     ddp_rank = int(os.environ['RANK'])  # 第几个GPU
#     ddp_local_rank = int(os.environ['LOCAL_RANK'])  # 与多机器有关，单机器一般与RANL一样，多机器则每台机器的GPU都从0开始
#     ddp_world_size = int(os.environ['WORLD_SIZE'])  #进程总数，即GPU总个数
#     device = f'cuda:{ddp_local_rank}'
#     torch.cuda.set_device(device)
#     master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
# else:
#     # vanilla, non-DDP run
#     ddp_rank = 0
#     ddp_local_rank = 0
#     ddp_world_size = 1
#     master_process = True
#     # attempt to autodetect device
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda"
#     elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#         device = "mps"
#     print(f"using device: {device}")

# # added after video, pytorch can be serious about it's device vs. device_type distinction
# device_type = "cuda" if device.startswith("cuda") else "cpu"

# torch.manual_seed(1337)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(1337)

# enc = tiktoken.get_encoding("gpt2")

# total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
# B = 16 # micro batch size
# T = 1024 # sequence length
# assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
# grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
# if master_process:
#     print(f"total desired batch size: {total_batch_size}")
#     print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
#     print(f"rank {ddp_rank}, local_rank {ddp_local_rank}, world_size {ddp_world_size}, master_process {master_process}")
#     print(f'ddp:{ddp}')

# # print("I am GPU", ddp_rank)
# # print('bye')


# train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
# val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

# torch.set_float32_matmul_precision('high')

# # create model
# model = GPT(GPTConfig(vocab_size=50304))
# # model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
# model.to(device)
# use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
# if use_compile:
#     model = torch.compile(model)
# if ddp:
#     model = DDP(model, device_ids=[ddp_local_rank])
# raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# max_lr = 6e-4
# min_lr = max_lr * 0.1
# warmup_steps = 10
# max_steps = 50  # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
# def get_lr(it):
#     # 1) linear warmup for warmup_iters steps
#     if it < warmup_steps:
#         return max_lr * (it+1) / warmup_steps
#     # 2) if it > lr_decay_iters, return min learning rate
#     if it > max_steps:
#         return min_lr
#     # 3) in between, use cosine decay down to min learning rate
#     decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
#     assert 0 <= decay_ratio <= 1
#     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
#     return min_lr + coeff * (max_lr - min_lr)

# # optimize!
# optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)


# for step in range(max_steps):
#     t0 = time.time()
#     optimizer.zero_grad()
#     loss_accum = 0.0
#     for micro_step in range(grad_accum_steps):
#         x, y = train_loader.next_batch()
#         x = x.to(device)
#         y = y.to(device)

#         # 使用该句及可使用bfloat16精度，但不是所有部分都变成了Bfloat16，为混合精度
#         with torch.autocast(device_type=device, dtype=torch.bfloat16):
#             logits, loss = model(x, y)
#             # import code; code.interact(local=locals())
#         # 因为原本计算损失时是会除以整个batch大小的，这里进行backward()累加会少除以grad_accum_step次，视频2小时44分处
#         loss = loss / grad_accum_steps
#         loss_accum += loss.detach()
#         # require_backward_grad_sync为True才会同步梯度，为False则不同步，而是将梯度累加到本地缓冲区中。直到下一次设置为 True 时，才会执行梯度同步
#         if ddp:
#             model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
#         loss.backward()     # 梯度会不停地累加grad_accum_step次

#     if ddp:
#         dict.all_reduce(loss_accum, op=dict.ReduceOp.AVG)

#     # 通过对梯度的裁剪，避免梯度爆炸问题，从而提高模型训练的稳定性。
#     norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#     lr = get_lr(step)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

#     optimizer.step()
#     torch.cuda.synchronize()    # 因为CPU主要是进行调度任务，CPU将一堆任务交给GPU后，GPU任务还没有执行完，而CPU已经执行完了
#     t1 = time.time()    # 因为time.time()记的是CPU的时间，所以需要同步一下，等待GPU任务执行完后再计算时间
#     d1 = (t1-t0) * 1000
#     tokens_processed = grad_accum_steps * train_loader.B * train_loader.T * ddp_world_size
#     tokens_per_sec = tokens_processed / (t1 - t0)
#     if master_process:
#         print(f'step {step}, loss {loss_accum.item():.4f}, lr {lr:.4e} ,time {d1:.2f}ms, norm {norm:.4f}, tokens/sec {tokens_per_sec:.2f}')
    
# if ddp:
#     destroy_process_group()