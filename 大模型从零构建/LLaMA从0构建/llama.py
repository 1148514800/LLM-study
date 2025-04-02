import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
from collections import OrderedDict


MASTER_CONFIG = {
    'd_model': 128,
    'epochs': 1000,
    'log_interval': 10,      # 每10个batch打印一次log
    'batch_size': 32,
    'n_heads': 8,
    'n_layers': 4,
    'context_window': 16,      # 滑动窗口采样，设置采样大小
    'vocab_size':4325         # 咱们的西游记数据集，一共包含4325个不重复的汉字，标点符号
}


# 构建batch
def get_batches(data, split, batch_size, context_window, config=MASTER_CONFIG):
    # 切分训练集，验证集，测试集，比例为，训练80%，验证10%，测试10%
    train = data[:int(0.8 * len(data))]
    val = data[int(0.8 * len(data)): int(0.9 * len(data))]
    test = data[int(0.9 * len(data)):]

    # 将全部的训练数据作为batch，验证集，测试集也换个变量存储（单纯为了方便看）
    batch_data = train
    if split == 'val':
        batch_data = val
    if split == 'test':
        batch_data = test

    # print(batch_data.size(0))

    # 这里需要学习torch.randint，生成大小为batch_size，内部数值为随机整数的tensor。生成随机数数值域为[0,训练集字符数量-滑动窗口大小-1]之间的整数
    # 详情可以参考官方文档，或者这个博客：https://blog.csdn.net/qq_41813454/article/details/136326473
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    # print('ix输出:')
    # print(ix)

    # 这里需要学习torch.stack，执行操作类似于python的zip关键字，只不过操作对象是tensor张量，指定任意维度的张量进行组合
    # 详情参考官方文档，或者这个博客：https://blog.csdn.net/dongjinkun/article/details/132590205

    # 这里x作为特征，y作为预测值，因为文本生成任务是根据前n个字符，去推理后面的1个字符，因此y的构造会使窗口在保持原大小的基础上向后移一位
    # 通过滑动窗口，对batch_data中的训练数据，进行随机取样，相当于随机选择训练数据。
    # 在原65万多个字符中，随机选取一个字符作为开始，并以这个开始点，向后选取滑动窗口个数的字符，作为训练数据，向后移一位就是其目标值。  因此ix的构造不能超出index。
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()

    # 返回特征值，目标值
    return x, y

# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()

        # torch中register_parameter()功能为：向我们建立的网络module添加parameter
        # 因此，我们需要对pytorch官方封装好的RMSNorm功能模块添加一个可以训练参数的层，命名为scale，并初始化为形状为layer_shape，所有值为1的张量矩阵。
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))

    def forward(self, x):
        # 计算Frobenius范数（求某个矩阵中所有元素的平方和再开方得到，该范数用来衡量矩阵的大小，详情请百度）, RMS = 1/sqrt(N) * Frobenius
        # 具体来说，torch.linalg.norm(x, dim=(1, 2))计算了x在第1和第2维度上的范数。然后，将结果乘以x[0].numel() ** -.5。x[0].numel()表示x第一个元素（即x的第一行）的元素个数，** -.5表示求平方根的倒数。
        # print(f'x:{x.shape}')   # x:torch.Size([32, 16, 128])
        # print(f'x[0].numel():{x[0].numel()}')   # x[0].numel():2048
        ff_rms = torch.linalg.norm(x, dim=(1,2)) * x[0].numel() ** -.5  # torch.linalg.norm()为计算范数的函数，默认为2范数
        # print(f'ff_rms.shape:{ff_rms.shape}')   # torch.Size([32])

        # 将ff_rms算子应用于输入的张量x，依据公式，做除法，因为输入向量x是三维的，因此需要对ff_rms进行升两维，也变成三维的张量。这样可以进行元素之间的计算。
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        # print(raw.shape)

        # 返回缩放后归一化的张量
        # print((self.scale[:x.shape[1], :].unsqueeze(0) * raw).shape)    # torch.Size([32, 16, 128])
        # 将归一化后的张量 raw 与可训练的缩放参数 self.scale 逐元素相乘，以实现缩放归一化的效果。
        return self.scale[:x.shape[1], :].unsqueeze(0) * raw

# 旋转位置编码
def get_rotary_matrix(context_window, embedding_dim):
    # 初始化一个0填充，形状为（context_window, embedding_dim, embedding_dim）的张量矩阵，其中context_window为token数量，后面两个embedding_dim组成正方形矩阵，与后面的attention计算对齐格式
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
    
    # 遍历每一个位置的token
    for position in range(context_window):
        # 还记得我的上一篇文章中说的，对于特征，两两组合吗，因此需要循环的次数为embedding_dim除以2
        for i in range(embedding_dim // 2):
            # 设置θ值，采样频率，或者说旋转频率，旋转角都可以，除以embedding_dim防止梯度问题。
            theta = 10000. ** (-2. * (i - 1) / embedding_dim)
            # 根据欧拉公式，计算旋转的角度，分别有sin 和cos，将计算拉到复数空间，并将旋转角度应用在上面的0填充的矩阵
            m_theta = position * theta  # 旋转角度
            R[position, 2 * i, 2 * i] = np.cos(m_theta)
            R[position, 2 * i, 2 * i + 1] = -np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)
            # 得到的结果是旋转位置编码矩阵，到这里还没覆盖到attention
    return R

# 激活函数
class SwiGLU(nn.Module):
    
    def __init__(self, size):
        super().__init__()
        # 定义一个门控的线性层，输入输出都是门控结构的尺寸 
        self.linear_gate = nn.Linear(size, size) 
        # 门控结构主干线性层 
        self.linear = nn.Linear(size, size)
        # 初始化一个随机数作为beta系数  
        self.beta = torch.randn(1, requires_grad=True)  

        # nn.Parameter用于指定某一层参数为可学习的，即本来不能通过训练更改参数，现在变成了可以经过训练来更新的参数。
        self.beta = nn.Parameter(torch.ones(1))
        # 将随机数beta指定为一个名为beta的神经网络层
        self.register_parameter("beta", self.beta)

    def forward(self, x):
        # Swish门控但愿的计算：（从括号里开始）对于原始输入的数据张量，经过线性变换乘以beta系数，再经过sigmoid变换为0-1之间的值，再乘以原数据经过门控线性变换。总的来说，线型输出经过非线性变换，再应用到线性变换的结果，元素按位置相乘，修正原本数据张量，就是这个门控结构做的事情。
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        # 将门控结构输出的值再按位乘以线型输出的原数据张量
        # 为啥这么做，我不知道，但是论文复现的代码就是这样滴，有兴趣可以研究一下，我没研究过。
        out = swish_gate * self.linear(x)  
        return out

# 单头注意力机制
class RoPEMaskedAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 计算Q权重矩阵
        self.w_q = nn.Linear(config['d_model'], config['d_model'], bias=False)
        # 计算K权重矩阵
        self.w_k = nn.Linear(config['d_model'], config['d_model'], bias=False)
        # 计算V权重矩阵
        self.w_v = nn.Linear(config['d_model'], config['d_model'], bias=False)
        # 获得旋转位置编码矩阵，接下来会覆盖Q和K权重矩阵
        self.R = get_rotary_matrix(config['context_window'], config['d_model'])

    def get_rotary_matrix(context_window, embedding_dim):
        # 初始化一个0填充，形状为（context_window, embedding_dim, embedding_dim）的张量矩阵，其中context_window为token数量，后面两个embedding_dim组成正方形矩阵，与后面的attention计算对齐格式
        R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
        
        # 遍历每一个位置的token
        for position in range(context_window):
            # 还记得我的上一篇文章中说的，对于特征，两两组合吗，因此需要循环的次数为embedding_dim除以2
            for i in range(embedding_dim // 2):
                # 设置θ值，采样频率，或者说旋转频率，旋转角都可以，除以embedding_dim防止梯度问题。
                theta = 10000. ** (-2. * (i - 1) / embedding_dim)
                # 根据欧拉公式，计算旋转的角度，分别有sin 和cos，将计算拉到复数空间，并将旋转角度应用在上面的0填充的矩阵
                m_theta = position * theta
                R[position, 2 * i, 2 * i] = np.cos(m_theta)
                R[position, 2 * i, 2 * i + 1] = -np.sin(m_theta)
                R[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
                R[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)
                # 得到的结果是旋转位置编码矩阵，到这里还没覆盖到attention
        return R

    def forward(self, x, return_attn_weights=False):
        # 前向传播时，输入矩阵的形状为(batch, sequence length, dimension)

        b, m, d = x.shape  # batch size, sequence length, dimension

        # 线性变换Q,K,V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # print(f'q.transpose(0, 1):{q.transpose(0, 1).shape}')
        # print(f'self.R[:m]:{self.R[:m].shape}')
        # print(f'self.R:{self.R.shape}')

        # 将旋转位置编码应用于Q和K，其中torch.bmm为矩阵做批量外积，transpose是转置，对Q矩阵转置，并与旋转位置编码做外积，再转置回原状，Q便应用了旋转位置编码。
        # 考虑到输入文本的长度，因此对位置编码矩阵在第一维度做截断，因为长了也没用，与文本长度一样。
        q_rotated = (torch.bmm(q.transpose(0, 1), self.R[:m])).transpose(0, 1)
        # print(f'q_rotated:{q_rotated.shape}')

        # 同理对K也应用旋转位置编码进行覆盖
        k_rotated = (torch.bmm(k.transpose(0, 1), self.R[:m])).transpose(0, 1)

        # 对注意力机制点积进行等比例缩放，防止attention张量过长引发梯度爆炸，对应
        activations = F.scaled_dot_product_attention(
            q_rotated, k_rotated, v, dropout_p=0.1, is_causal=True
        )

        # print(f'activations:{activations.shape}')   # torch.Size([32, 16, 128])

        # 如果return_attn_weights参数置为1，则需要对attention进行掩码，因为在学习的时候，希望模型能依据前n个token去预测token，而不是开卷考试。
        if return_attn_weights:
            # 创建注意力掩码矩阵，其中torch.tril函数为：对于矩阵，取左下三角，剩下的都置0
            attn_mask = torch.tril(torch.ones((m, m)), diagonal=0)
            # 计算注意力机制的权重矩阵，并对最后一维度做归一化，（突击检查）为什么是最后一维！因为最后一维度是每个token的特征向量！
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1, 2)) / np.sqrt(d) + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights

        return activations

# 多头注意力机制
class RoPEMaskedMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 一个注意力机制头对象构建完毕了，多头的，首先多次创建这个对象。生成多个注意力机制头，塞到一个列表里。
        self.heads = nn.ModuleList([
            RoPEMaskedAttentionHead(config) for _ in range(config['n_heads'])
        ])
        # 在模型结构上，创建一个线性层（隐藏层），用于线型输出注意力机制头输出的张量矩阵，寻找多头之间的特征，但是更主要的是，x经过多头计算后形状改变了，创建线性层，让张量矩阵变回原来输入的形状。
        # 同时为了防止过拟合，使用随机神经元失活，比率0.1
        # 线性层输入形状：注意力机制的头数，乘以矩阵的维度，关联到俺的上一篇文章，就是key矩阵，在多头之间共享权重，减少计算的思维。 输出为：模型的embedding维度数
        self.linear = nn.Linear(config['n_heads'] * config['d_model'], config['d_model'])  
        self.dropout = nn.Dropout(0.1)  

    def forward(self, x):
        # 输入矩阵形状x： (batch, sequence length, dimension)

        # print(f'self.config:{self.config}')
        # print(f'self.heads:{self.heads}')

        # 每一个注意力机制头，都传入X进行计算。（这个地方开启并行执行会不会快一些，但是不知道pytorch是不是自动调用并行）
        heads = [h(x) for h in self.heads]
        # 输入张量x经过多个头计算attention（同时，attention是已经覆盖了RoPE的），重新拼接成新的矩阵，重新放入变量x。到这里你应该觉得：那矩阵形状不就变了吗
        x = torch.cat(heads, dim=-1)
        
        # 这不，线性层的作用来了
        x = self.linear(x)
        
        # 随机失活一下，防止过拟合
        x = self.dropout(x)
        return x


# LlaMa的功能块
class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rms = RMSNorm((config['context_window'], config['d_model']))
        self.attention = RoPEMaskedMultiheadAttention(config)
        self.feedforward = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
        )

    def forward(self, x):

        x = self.rms(x) 
        x = x + self.attention(x)
        x = self.rms(x) 
        x = x + self.feedforward(x)
        return x

# 现在，我们组装LlaMa
class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Embedding不解释
        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        # 根据传入的堆叠层数，创建Llama功能块，注意OrderedDict为一种特殊类型的字典数据，保留字典写入的顺序，先插入的数据在前，后插入的数据在后。
        # 这里，我们将llama的功能块堆叠4层
        self.llama_blocks = nn.Sequential(
            OrderedDict([(f"llama_{i}", LlamaBlock(config)) for i in range(config['n_layers'])])
        )
        # FFN层，包含：线性层、激活函数非线性变换、再用线性层输出最终解码数值。
        self.ffn = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        # 看看咱们的大模型多少参数！
        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        # embedding嵌入
        x = self.embeddings(idx)
        # Llama模型计算
        x = self.llama_blocks(x)
        # FFN计算，得到logits
        logits = self.ffn(x)

        # 推理阶段没有目标值，只输出结果
        if targets is None:
            return logits
        # 训练阶段，有目标值，需要输出结果，以及loss，用于反向传播更新权重！
        else:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss


# 训练函数
def train(model, optimizer, scheduler=None, config=MASTER_CONFIG, print_logs=False):
    # loss存储
    losses = []

    # 训练时间记录开始时间
    start_time = time.time()

    # 循环训练指定epoch的轮数
    for epoch in range(config['epochs']):
        # 优化器要初始化啊，否则每次训练都是基于上一次训练结果进行优化，效果甚微
        optimizer.zero_grad()

        # 获取训练数据
        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'])

        # print(f'xs:{xs.shape}')
        # print(f'ys:{ys.shape}')
        # import sys;sys.exit()

        # 前向传播计算概率矩阵与loss
        logits, loss = model(xs, targets=ys)

        # 反向传播更新权重参数，更新学习率优化器
        loss.backward()
        optimizer.step()

        # 如果提供学习率调度器，那么学习率会通过调度器进行修改，比如学习率周期性变化，或者梯度减小，增加，具体策略需要综合考虑进行设置，详情自行查询，关键字：lr_scheduler
        if scheduler:
            scheduler.step()

        # 打印log
        if epoch % config['log_interval'] == 0:
            # 训练时间
            batch_time = time.time() - start_time

            # 执行评估函数，在训练集和验证集上计算loss
            x = evaluate_loss(model)

            # Store the validation loss
            losses += [x]

            # 打印进度日志
            if print_logs:
                print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch)/config['log_interval'] :.3f}")

            # 重置开始时间，用于计算下一轮的训练时间
            start_time = time.time()

            # 打印下一轮的学习率，如果使用了lr_scheduler
            if scheduler:
                print("lr: ", scheduler.get_lr())

    # 上面所有epoch训练结束，打印最终的结果
    print("Validation loss: ", losses[-1]['val'])

    # 返还每一步loss值的列表，因为我们要画图，返还的是loss迭代的图像
    return pd.DataFrame(losses).plot()

# 评估函数
@torch.no_grad()
def evaluate_loss(model, config=MASTER_CONFIG):
    # 评估结果存储变量
    out = {}

    # 将模型置为评估模式
    model.eval()

    # 分别会在训练集和验证集里通过get_batchs()函数取评估数据
    for split in ["train", "val"]:

        losses = []

        # 评估10个batch
        for _ in range(10):
            # 拿到特征值（输入数据），以及目标值（输出数据）
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'])

            # 把拿到的数据丢进模型，得到loss值
            _, loss = model(xb, yb)

            # 更新loss存储
            losses.append(loss.item())

        # 这里就是大家经常在控制台看到的 "train_loss"  "valid_loss"由来
        out[split] = np.mean(losses)

    # 评估完了，别忘了把模型再置回训练状态，下一个epoch还要继续训练呢
    model.train()

    return out

# 生成函数
def generate(model, config=MASTER_CONFIG, max_new_tokens=20):
    # 生成5个0，作为输入数据,5行一列，代表输入5个字符。 这个地方可以自行替换其他随机数测试。
    idx = torch.zeros(5, 1).long()
    print(idx[:, -config['context_window']:])
    for _ in range(max_new_tokens):
        # 因为推理的时候，依赖后面的n个token，所以滑动窗口要从后往前选择输入数据的倒数几个token，这个是超过字符数量会对输入进行截断，只选取最后几个token：idx[:, -config['context_window']:]
        logits = model(idx[:, -config['context_window']:])
        # print(logits.size())
        # 得到模型输出的结果，进行解码，这里logits[:, -1, :]挺抽象的，实际上第一维度是输入的字符数，第二维度是时间步，第三维度是词表
        # 即，对每一步的解码结果，取最后一个时间步的数据，作为输出的数据。解码的过程是第一次解码，输入5个token，第二次解码依赖的是原来5个token的最后4个，加上上一步解码生成的一个，也是5个token，如此循环。
        last_time_step_logits = logits[:, -1, :]
        # print('last_time_step_logits')
        # print(last_time_step_logits.shape)
        # 计算概率分布
        p = F.softmax(last_time_step_logits, dim=-1)
        # print('p_shape')
        # print(p.shape)
        # 根据概率分布计算下一个token，这里使用 torch.multinomial做的是随机采样
        idx_next = torch.multinomial(p, num_samples=1)
        # print('idx_next_shape')
        # print(idx_next.shape)
        # 将新的idx通过张量拼接写入到解码序列中
        idx = torch.cat([idx, idx_next], dim=-1)
    # 使用之前定义的解码函数，将ID转换为汉字，我们得到的5行21列的数据，来源于每一个输入字符作为开始位置，生成20个字符。 因为5个输入都是0，在词表中编号为0的数据是'\n'。
    print(idx.shape)
    return [decode(x) for x in idx.tolist()]




# 读数据
lines = open("xiyouji.txt", 'r').read()

# 创建简易版词表（字符级）
vocab = sorted(list(set(lines)))


# 将词表编码成为数字，普通的整数，如："1":"孙"，"2":"悟" ，"3":"空"，下面那个是键和值对调
itos = {i: ch for i, ch in enumerate(vocab)}
# 双向映射
stoi = {ch: i for i, ch in enumerate(vocab)}

# 接下来我们创建一个简易的编码器和解码器。用于输入进模型之前，将文本转换为数字，输出之前将数字转换为文本：
# 编码器（青春版）
def encode(s):
    return [stoi[ch] for ch in s]

# 解码器（青春版）
def decode(l):
    return ''.join([itos[i] for i in l])

# 对全文进行编码，并映射成为tensor
dataset = torch.tensor(encode(lines), dtype=torch.int16)





# 学习率优化器选择余弦退火
llama_with_cosine = Llama(MASTER_CONFIG)

llama_optimizer = torch.optim.Adam(
    llama_with_cosine.parameters(),
    betas=(.9, .95),
    weight_decay=.1,
    eps=1e-9,
    lr=1e-3
)
# 余弦退火学习率优化器，让学习率逐渐减小，在结束时达到最低值。 详细可以百度，这种文章很多。
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(llama_optimizer, 300, eta_min=1e-5)

# 跑一下！
train(llama_with_cosine, llama_optimizer, scheduler=scheduler)

