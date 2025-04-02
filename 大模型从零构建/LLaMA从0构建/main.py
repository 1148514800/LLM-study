'''

（徒手搓LLM）逐行代码从0构造一个LLM——LlaMa篇
https://zhuanlan.zhihu.com/p/1674261485

'''

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd


MASTER_CONFIG = {
    # 参数放这里
}


# 读数据
lines = open("xiyouji.txt", 'r').read()

# 创建简易版词表（字符级）
vocab = sorted(list(set(lines)))

# 查看词表前n个字符
head_num=50
print('词表前{}个:'.format(head_num), vocab[:head_num])

print('词表大小:', len(vocab))

# 输出：
# 词表前50个: ['\n', ' ', '!', '"', '#', '*', ',', '.', '—', '‘', '’', '“', '”', '□', '、', '。', '《', '》', '一', '丁', '七', '万', '丈', '三', '上', '下', '不', '与', '丑', '专', '且', '丕', '世', '丘', '丙', '业', '丛', '东', '丝', '丞', '丢', '两', '严', '丧', '个', '丫', '中', '丰', '串', '临']
# 词表大小: 4325


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

# 来试一下这个“高端”的编解码器
decode(encode("悟空"))
encode("悟空")

# 输出：
# [1318, 2691]


# 对全文进行编码，并映射成为tensor
dataset = torch.tensor(encode(lines), dtype=torch.int16)

# 看一下形状，实际上就是多少个字符，一共65万个字符
print(dataset.shape)
print(dataset)

# 输出：
# torch.Size([658298])
# tensor([   0, 4319, 1694,  ...,   12,    0,    0], dtype=torch.int16)



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


# 根据上面构造的get_batchs()函数，更新参数字典。
MASTER_CONFIG.update({
    'batch_size': 8,          # 不解释
    'context_window': 16,      # 滑动窗口采样，设置采样大小
    'vocab_size':4325         # 咱们的西游记数据集，一共包含4325个不重复的汉字，标点符号
})


# 获取训练数据
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

# print(f'xs:{xs.shape}')
# import sys;sys.exit()


# 因为是随机生成的采样，我们可以看一下数据，其中每个采样数据，来自于原文随机的起始点，每个元组为一个（x,y），可以观察每个x和y的首位去直观感受一下滑动窗口执行的操作
decoded_samples = [(decode(xs[i].tolist()), decode(ys[i].tolist())) for i in range(len(xs))]

print(decoded_samples)

# 输出：
# [('姿娇且嫩’ ！”那女子笑\n而悄答', '娇且嫩’ ！”那女子笑\n而悄答道'), ('泼猢狲，打杀我也！”沙僧、八戒问', '猢狲，打杀我也！”沙僧、八戒问道'), ('人家，不惯骑马。”唐僧叫八戒驮着', '家，不惯骑马。”唐僧叫八戒驮着，'), ('著一幅“圯桥进履”的\n画儿。行者', '一幅“圯桥进履”的\n画儿。行者道'), ('从何来？这匹马，他在此久住，必知', '何来？这匹马，他在此久住，必知水'), ('声去，唿哨一声，寂然不见。那一国', '去，唿哨一声，寂然不见。那一国君'), ('刀轮剑砍怎伤怀！\n火烧雷打只如此', '轮剑砍怎伤怀！\n火烧雷打只如此，'), ('鲜。紫竹几竿鹦鹉歇，青松数簇鹧鸪', '。紫竹几竿鹦鹉歇，青松数簇鹧鸪\n')]


# 构造一个评估函数
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




# 网络搭建
class StupidModel(nn.Module):
    def __init__(self, config=MASTER_CONFIG):
        super().__init__()
        self.config = config

        # embedding层，输入：词表大小，输出：维度大小
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])

        # 创建线性层用于捕捉特征关系
        # 下面突击检查：这玩意是不是隐藏层！线性层堆叠越多是不是越好！堆叠越多是不是更计算开销越大！
        # LlaMa使用的激活函数是SwiGLU，目前在这个斯丢匹德模型架构里面先用Relu
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        # 这个命令可以背一下，或者复制粘贴到自己的学习笔记。 因为这行命令会直接帮你查看模型的参数量。
        # 否则要么自己手算，要么就是听别人讲某某模型 7B  20B  108B   有了这个命令，你就能直接查看你创建的模型参数量多少
        print("模型参数量：", sum([m.numel() for m in self.parameters()]))


class SimpleBrokenModel(nn.Module):
    # init里的跟上面一样，没变化
    def __init__(self, config=MASTER_CONFIG):
      super().__init__()
      self.config = config
      self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
      self.linear = nn.Sequential(
          nn.Linear(config['d_model'], config['d_model']),
          nn.ReLU(),
          nn.Linear(config['d_model'], config['vocab_size']),
      )



      # 添加前向传播函数
    def forward(self, idx, targets=None):
        # print(f'idx.shape:{idx.shape}')     # idx.shape:torch.Size([8, 16])

        # 实例化embedding层，输入映射为id的数据，输出嵌入后的数据
        x = self.embedding(idx)

        # print(f'x.shape:{x.shape}')     # x.shape:torch.Size([8, 16, 128])
        # import sys;sys.exit()

        # 线性层承接embedding层输出的数据
        a = self.linear(x)

        # 对线性层输出的数据在最后一个维度，做softmax，得到概率分布
        logits = F.softmax(a, dim=-1)

        # 如果有目标值（也就是我们前面的y），则计算通过交叉熵损失计算loss结果。给输出的概率矩阵变个形状，再给目标值变个形状。  统一一下输入输出，然后计算loss。其中最后一维代表着一条数据。
        # 此处需要了解tensor.view()函数，带上几何空间想象力去想一下矩阵的形状。
        if targets is not None:

            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        # 如果没有目标值，则只返回概率分布的结果
        else:
            return logits

        # 查看参数量
        print("模型参数量：", sum([m.numel() for m in self.parameters()]))


# 这里我们设置这个模型为128维的embedding
MASTER_CONFIG.update({
    'd_model': 128,
})

# 实例化模型，传参
model = SimpleBrokenModel(MASTER_CONFIG)

# 再看看参数量
print("咱们的模型这么多参数量:", sum([m.numel() for m in model.parameters()]))
# 于是乎，我们创建了一个1128307个参数的模型，上面参数想怎么改，自己改！电脑不会爆炸！



# # 获取训练的特征数据与目标数据
# xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

# # 扔进模型获取概率分布矩阵与loss
# logits, loss = model(xs, ys)
# print(f'logits:{logits.shape}')
# print(f'loss:{loss}')

# 输出：
# tensor(8.3722, grad_fn=<NllLossBackward0>)


# # 获取每个位置上概率最高的索引
# predicted_indices = torch.argmax(logits, dim=-1)
# # 将预测的索引转换为对应的字符
# predicted_chars = [decode(predicted_indices[i].tolist()) for i in range(predicted_indices.shape[0])]
# # 输出预测结果
# for i, prediction in enumerate(predicted_chars):
#     print(f'样本 {i+1} 的预测结果: {prediction}')



# 更新参数，训练轮次，batch_size，log日志打印步长
MASTER_CONFIG.update({
    'epochs': 1000,
    'log_interval': 10,      # 每10个batch打印一次log
    'batch_size': 32,
})

# 实例化模型
model = SimpleBrokenModel(MASTER_CONFIG)

# 创建一个Adam优化器，基础知识，
optimizer = torch.optim.Adam(
    model.parameters(),      # 优化器执行优化全部的模型参数
)


# 训练函数
# 构建训练函数
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

# 启动训练
# train(model, optimizer)


# 拿掉softmax，logits改为获取最后一个线性层输出的结果，不进行softmax计算概率分布。
# 因此将这个架构取名为：不那么蠢的模型架构
class SimpleNotStupidModel(nn.Module):
    def __init__(self, config=MASTER_CONFIG):
      super().__init__()
      self.config = config
      self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
      self.linear = nn.Sequential(
          nn.Linear(config['d_model'], config['d_model']),
          nn.ReLU(),
          nn.Linear(config['d_model'], config['vocab_size']),
      )
      print("Model parameters:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embedding(idx)

        # 看这里，线性层直接输出结果，不转换为概率矩阵，只修改这里，其余不动。
        logits = self.linear(x)
        # print(logits.shape)

        if targets is not None:

            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss
        else:
            return logits
        print("Model parameters:", sum([m.numel() for m in self.parameters()]))


# # 再来一次实例化各种功能，再启动一次训练
# model = SimpleNotStupidModel(MASTER_CONFIG)
# xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
# logits, loss = model(xs, ys)
# optimizer = torch.optim.Adam(model.parameters())
# train(model, optimizer)

# # loss开窍了，下降了很多


# 推理函数（输出结果就别纠结其效果了，权重都没保存，就是根据模型初始化生成的随机数组成的矩阵做的推理）
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

# output = generate(model)
# print(f'output:{output}')



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
    

class SimpleNotStupidModel_RMS(nn.Module):
    def __init__(self, config=MASTER_CONFIG):
      super().__init__()
      self.config = config
      self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
      # 在这里，我们添加RMS层
      self.rms = RMSNorm((config['context_window'], config['d_model']))
      self.linear = nn.Sequential(
          nn.Linear(config['d_model'], config['d_model']),
          nn.ReLU(),
          nn.Linear(config['d_model'], config['vocab_size']),
      )
      print("Model parameters:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        # 在这里，添加实例化后的RMS层，承接Embedding层输出的张量
        x = self.rms(x)

        logits = self.linear(x)
        # print(logits.shape)

        if targets is not None:

            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss
        else:
            return logits
        print("Model parameters:", sum([m.numel() for m in self.parameters()]))


# # 好啦，这样我们对原来的NotStupidModel添加了RMSNorm，现在执行一下看看
# model = SimpleNotStupidModel_RMS(MASTER_CONFIG)

# xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

# logits, loss = model(xs, ys)

# optimizer = torch.optim.Adam(model.parameters())

# train(model, optimizer)

# # 在同样的训练超参数设置上，加入了RMSNorm的训练速度明显加快。



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



# 单头注意力机制
# 此为单头注意力机制
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


    # 这里将上一个代码块中实现的创建旋转位置编码的功能函数原封不动的拿过来
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
    


# 单头注意力机制实现完毕，下面实现多头注意力机制
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


MASTER_CONFIG.update({
    'n_heads': 8,
})



# 我们已经创建完了所需要的算子，  现在积木已创建完毕，将这些积木组合起来！！！！
class RopeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embedding层
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        
        # RMSNorm层
        self.rms = RMSNorm((config['context_window'], config['d_model']))
        
        # 旋转位置编码器+注意力机制
        self.rope_attention = RoPEMaskedMultiheadAttention(config)

        # 线性层+激活函数变为非线性输出！
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
        )

        # 最终的输出，因为需要解码，因为输出的维度与词表大小统一！！！
        self.last_linear = nn.Linear(config['d_model'], config['vocab_size'])

        print("model params:", sum([m.numel() for m in self.parameters()]))
    # 前向传播
    def forward(self, idx, targets=None):
        # embedding，不解释
        x = self.embedding(idx)
        # 归一化数值，不解释
        x = self.rms(x)  

        # print(f'rms(x):{x.shape}')  # torch.Size([32, 16, 128])

        # 相加，解释一下，因为attention是要覆盖到原矩阵的，想象两个形状一样的矩阵为两张纸，左手一张纸，右手一张纸，双手合十，啪！覆盖。 使用加算，就是将两个矩阵中的元素按位置相加！直接覆盖值！
        x = x + self.rope_attention(x)
        # 再归一化！
        x = self.rms(x)
        # 因为直接计算归一化的数值可能出现梯度问题，因此把归一化的值作为修正系数，再覆盖！  
        x = x + self.linear(x)
        # 到这里，才是最终输出vocab数量的神经元输出！！！！！！
        logits = self.last_linear(x)

        # 训练阶段有目标值
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss
        # 验证或者推理阶段，目标值y没有！只有结果，没有loss！
        else:
            return logits
        


# # 再跑一下！
# model = RopeModel(MASTER_CONFIG)
# xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
# logits, loss = model(xs, ys)
# optimizer = torch.optim.Adam(model.parameters())
# train(model, optimizer)



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


# 再将swiglu添加进上面的模型
class RopeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.rms = RMSNorm((config['context_window'], config['d_model']))
        self.rope_attention = RoPEMaskedMultiheadAttention(config)
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            # 在这里，增加了SwiGLU层
            SwiGLU(config['d_model']),  
        )
        self.last_linear = nn.Linear(config['d_model'], config['vocab_size'])
        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        x = self.rms(x)  
        x = x + self.rope_attention(x)
        x = self.rms(x)
        x = x + self.linear(x)
        logits = self.last_linear(x)

        if targets is not None:
            # Calculate cross-entropy loss if targets are provided
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        else:
            return logits


# # 一二三四！再来一次！
# model = RopeModel(MASTER_CONFIG)
# xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
# logits, loss = model(xs, ys)
# optimizer = torch.optim.Adam(model.parameters())
# train(model, optimizer)



# OK！ 现在我们更新一下，隐藏层维度堆叠多少层，我们先来4层尝尝咸淡！！！！
MASTER_CONFIG.update({
    'n_layers': 4,  
})


# 现在我们拥有了所有的算子，RMS，ROPE,SWIGLU，我们搭建我们的LlaMa！ 首先实现LlaMa的功能块，然后堆叠。
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


# 看一下我们的超参数字典
print(f'MASTER_CONFIG:{MASTER_CONFIG}')

# 输出：
#{'batch_size': 32,'context_window': 16, 'vocab_size': 4325,'d_model': 128,'epochs': 1000,'log_interval': 10,'n_heads': 8,'n_layers': 4}


# # 用config字典，创建llama的功能块
# block = LlamaBlock(MASTER_CONFIG)

# # 生成一条随机数据，丢到这个llama功能块里，看一下是不是有bug
# random_input = torch.randn(MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'], MASTER_CONFIG['d_model'])

# # 执行以下看看输出
# output = block(random_input)
# print(output.shape)



# 现在，我们组装LlaMa
from collections import OrderedDict
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


# 开始训练咱们的Llama
llama = Llama(MASTER_CONFIG)
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])
logits, loss = llama(xs, ys)
optimizer = torch.optim.Adam(llama.parameters())
train(llama, optimizer)


# 再看一下推理效果（实际上也没什么效果-。-）
# 别忘了generate里面的输入数据是咱们弄的5个0，如果替换为encode之后的数也是可以的！组成列表，转换tensor，这个应该没问题的吧~
generated_text = generate(llama, MASTER_CONFIG, 500)[0]
print(generated_text)


# 下面是测试集跑一下 
# 获取测试集的特征值和目标值
xs, ys = get_batches(dataset, 'test', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

# 丢进Llama获取loss
logits, loss = llama(xs, ys)

print(loss)
# 输出：
# tensor(4.7326, grad_fn=<NllLossBackward0>)




# 还有优化的点哦，别忘了optimizer！以及学习率调度器！
# 调整参数再来一次！

MASTER_CONFIG.update({
    "epochs": 1000
})

# 学习率优化器选择余弦退火
llama_with_cosine = Llama(MASTER_CONFIG)

llama_optimizer = torch.optim.Adam(
    llama.parameters(),
    betas=(.9, .95),
    weight_decay=.1,
    eps=1e-9,
    lr=1e-3
)
# 余弦退火学习率优化器，让学习率逐渐减小，在结束时达到最低值。 详细可以百度，这种文章很多。
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(llama_optimizer, 300, eta_min=1e-5)

# 跑一下！
train(llama_with_cosine, llama_optimizer, scheduler=scheduler)



# 保存模型
# 保存模型权重
model_save_path = "./hf_model_save/pytorch_model.bin"
torch.save(llama_with_cosine.state_dict(), model_save_path)

# 生成一个config文件
import json

config_save_path = "./hf_model_save/config.json"
with open(config_save_path, 'w') as f:
    json.dump(MASTER_CONFIG, f)

# 保存optimizer和学习率调度器的状态，方便继续微调
optimizer_save_path = "./hf_model_save/optimizer.pt"
torch.save(llama_optimizer.state_dict(), optimizer_save_path)

scheduler_save_path = "./hf_model_save/scheduler.pt"
torch.save(scheduler.state_dict(), scheduler_save_path)



# 加载模型
# 接下来是加载模型
llama_with_cosine = Llama(MASTER_CONFIG)  

# 加载模型权重
model_save_path = "./hf_model_save/pytorch_model.bin"
llama_with_cosine.load_state_dict(torch.load(model_save_path))

# 设置为评估模式
llama_with_cosine.eval()


# 加载优化器和学习率调度器，如果需要继续训练什么的。
llama_optimizer.load_state_dict(torch.load(optimizer_save_path))
scheduler.load_state_dict(torch.load(scheduler_save_path))

# 进行推理
output = generate(llama_with_cosine, MASTER_CONFIG)
print(output)