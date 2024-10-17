import torch
import numpy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error
from typing import List, Dict
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


@dataclass
class Train_Config:

    # 超参数
    input_size: int = 20      # LSTM输入特征的维度（每个时间步的数据特征数量）
    hidden_size: int = 64     # LSTM隐藏层的大小（隐藏单元的数量）
    num_layers: int = 2       # LSTM的层数（堆叠LSTM层数）
    output_size: int = 1      # 模型输出的维度（RUL回归任务中为1）
    num_epochs: int = 2     # 训练的迭代次数
    batch_size: int = 32      # 每次训练时使用的数据样本数
    learning_rate: float = 0.001  # 学习率，控制每次梯度更新的步长

    # 正则化和优化
    dropout_ratio: float = 0.1      # LSTM层的dropout概率，用于防止过拟合
    weight_decay: float = 1e-5  # 权重衰减（L2正则化系数），用于防止过拟合

    # 设备相关配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 选择设备，使用GPU或CPU进行计算

    # 打印和日志配置
    print_every: int = 10     # 每训练多少个epoch打印一次训练损失
    save_model_path: str = "./model.pt"  # 保存训练好模型的路径
    


class SensorDataset(Dataset):
    def __init__(self, X: List[torch.Tensor], y: List[torch.Tensor]):
        self.X = [torch.tensor(x, dtype=torch.float) for x in X]
        self.y = [torch.tensor(y_i, dtype=torch.float) for y_i in y]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def collate_fn(batch):
    X_batch, y_batch = zip(*batch)
    lengths = [x.shape[0] for x in X_batch]
        
    # (batch_size, seq_len, input_size)
    X_batch = pad_sequence(X_batch, batch_first=True, padding_value=0)
    # (batch_size, seq_len)
    y_batch = pad_sequence(y_batch, batch_first=True, padding_value=0)
    
    return X_batch, y_batch, lengths

    
class LSTMModel(nn.Module):
    def __init__(self, config: Train_Config):
        super(LSTMModel, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.lstm = nn.LSTM(config.input_size, config.hidden_size, config.num_layers, batch_first=True)
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.fc = nn.Linear(config.hidden_size, config.output_size)
    
    def forward(self, x, lengths):
        # 对输入的序列进行 pack 操作，忽略填充部分
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_x)  # LSTM的输出
        out, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        out = self.dropout(out)
        out = self.fc(out)     # 只取最后一个时间步的输出，进行全连接层预测
        
        return out
    


def train(
    model,
    train_dataset,
    val_dataset,
    config: Train_Config
):
    # 创建数据加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)  # 验证集无需打乱
    
    # 模型实例化
    model.to(config.device)
    model.train()
    
    # 损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差作为损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # 训练模型
    for epoch in range(config.num_epochs):
        for i, (X_batch, y_batch, lengths) in enumerate(train_loader):
            X_batch = X_batch.to(config.device)
            y_batch = y_batch.to(config.device)
            
            # 前向传播
            outputs = model(X_batch, lengths)
            loss = criterion(outputs.squeeze(), y_batch)
            
            # 反向传播和优化
            optimizer.zero_grad()  # 梯度清零
            loss.backward()        # 反向传播
            optimizer.step()       # 更新权重
        
        if (epoch+1) % config.print_every == 0:
            print(f'Epoch [{epoch+1}/{config.num_epochs}], Loss: {loss.item():.4f}')

    model.eval()
    print("训练完成！")
    print(f"loss: {loss.item()}")
    
    # 保存模型
    torch.save(model.state_dict(), config.save_model_path)
    print(f"模型已保存到 {config.save_model_path}")
    
    # 训练结束后在验证集上进行评估
    val_preds = []
    val_targets = []
    with torch.inference_mode():  # 验证阶段不需要计算梯度
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.unsqueeze(1).to(config.device)
            y_batch = y_batch.to(config.device)
            
            outputs = model(X_batch)
            val_preds.append(outputs.item())  # 将预测值保存
            val_targets.append(y_batch.item())  # 将真实值保存

    # 计算验证集上的MSE
    mse = mean_squared_error(val_targets, val_preds)
    rmse = mse**0.5
    print(f'Validation MSE: {mse:.4f}')
    print(f'Validation RMSE: {rmse:.4f}')
    
    return loss.item(), mse, rmse