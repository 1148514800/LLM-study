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

