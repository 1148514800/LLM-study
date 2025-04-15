# python -m accelerate.commands.launch demo.py

'''
单卡直接运行：35.76 seconds
单卡accelerate：48.82 seconds
双卡accelerate：29.51 seconds
'''

from accelerate import Accelerator, DeepSpeedPlugin
import torch
from torch.utils.data import DataLoader, TensorDataset

import time

class SimpleNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)  #.to("cuda:0")
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim) #.to("cuda:1")

    def forward(self, x):
        # x.to("cuda:0")
        x = torch.relu(self.fc1(x))
        # x.to("cuda:1")
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    input_dim = 10
    hidden_dim = 20
    output_dim = 2
    batch_size = 64
    data_size = 10000

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    input_data = torch.randn(data_size, input_dim)
    labels = torch.randn(data_size, output_dim)

    dataset = TensorDataset(input_data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNet(input_dim, hidden_dim, output_dim)
    # model.to(device)
    
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_clipping=1.0)
    accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
    # accelerator = Accelerator()
    optimization = torch.optim.Adam(model.parameters(), lr=0.00015)
    crition = torch.nn.MSELoss()
    # print(f'len(dataloader):{len(dataloader)}')
    model, dataloader, optimization = accelerator.prepare(model, dataloader, optimization)
    # print(f'len(dataloader):{len(dataloader)}')
    start_time = time.time()
    for epoch in range(100):
        model.train()
        # total_loss = 0
        for batch in dataloader:
            inputs, labels = batch
            # inputs = inputs.to(device)
            # labels = labels.to(device)
            outputs = model(inputs)
            loss = crition(outputs, labels)
            # total_loss += loss.item()
            optimization.zero_grad()
            # loss.backward()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                # print("sync gradients")
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimization.step()
        
        # 这里的loss只是取了最后一次的loss，并不是所有batch的loss

        # 下面两个方法得到的结果是一样的
        gather_loss = accelerator.gather(loss).mean()  # 收集所有 GPU 的损失并取平均
        if accelerator.is_main_process:
            print(f"Epoch {epoch} loss: {gather_loss.item()}")

        # avg_loss = accelerator.reduce(loss, reduction="mean")
        # if accelerator.is_main_process:
        #     print(f"Epoch {epoch} loss: {avg_loss.item()}")

    end_time = time.time()  # 记录训练结束时间
    training_time = end_time - start_time  # 计算训练时间
    if accelerator.is_main_process:
        print(f"Training time: {training_time:.2f} seconds")

    # accelerator.save(model.state_dict(), "model.pth")
            
    
    