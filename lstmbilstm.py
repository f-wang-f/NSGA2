import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import time
import warnings
import traceback
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')
# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 配置参数（严格匹配论文） ====================
# 时序参数
SEQUENCE_LENGTH = 168  # 论文中的weekly序列（1周小时级数据）
PREDICTION_STEPS = 1  # 预测步长（论文支持1/2/3，可调整）
# 模型参数 - 标准LSTM
LSTM_HIDDEN_SIZE = 32
LSTM_DROPOUT = 0.2
LSTM_RECURRENT_DROPOUT = 0.1
# 模型参数 - BiLSTM
BILSTM_HIDDEN_SIZE = 32
# 训练参数
LSTM_BATCH_SIZE = 12
LSTM_EPOCHS = 300
BILSTM_BATCH_SIZE = 46
BILSTM_EPOCHS = 200
LEARNING_RATE = 0.001
# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {DEVICE}')


# ==================== 2. 数据集类定义 ====================
class WindTimeSeriesDataset(Dataset):
    def __init__(self, features, target, sequence_length, prediction_steps):
        self.features = torch.FloatTensor(features)
        self.target = torch.FloatTensor(target)
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps

    def __len__(self):
        return len(self.features) - self.sequence_length - self.prediction_steps + 1

    def __getitem__(self, idx):
        # 输入序列：[sequence_length, feature_dim]
        x = self.features[idx:idx + self.sequence_length]
        # 目标值：预测未来prediction_steps步的风速
        y = self.target[idx + self.sequence_length:idx + self.sequence_length + self.prediction_steps]
        return x, y.squeeze()  # squeeze确保目标维度正确


# ==================== 3. 模型定义（严格匹配论文参数） ====================
class StandardLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, dropout=0.2, recurrent_dropout=0.1):
        super(StandardLSTM, self).__init__()
        self.hidden_size = hidden_size

        # 论文参数：1层LSTM，relu激活，sigmoid循环激活
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        # 论文中LSTM激活为relu（PyTorch需手动实现）
        self.activation = nn.ReLU()

        # 输出层：线性激活（回归任务）
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x: [batch_size, sequence_length, input_dim]
        lstm_out, (hidden, cell) = self.lstm(x)
        # 仅取最后时间步的隐藏状态（论文中return_sequences=False）
        out = self.activation(hidden[-1])  # hidden: [num_layers, batch_size, hidden_size]
        out = self.fc(out)
        return out


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, dropout=0.2, recurrent_dropout=0.1):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2  # 双向

        # 论文参数：1层双向LSTM，relu激活，sigmoid循环激活
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.activation = nn.ReLU()

        # 双向LSTM输出维度为2*hidden_size
        self.fc = nn.Linear(hidden_size * self.num_directions, output_dim)

    def forward(self, x):
        # x: [batch_size, sequence_length, input_dim]
        bilstm_out, (hidden, cell) = self.bilstm(x)
        # 拼接正反方向最后一层的隐藏状态
        hidden_concat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.activation(hidden_concat)
        out = self.fc(out)
        return out


# ==================== 4. 训练与验证函数 ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, model_name):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = f'{model_name}_best.pth'

    start_time = time.time()

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * x_batch.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

        # 每20轮打印进度
        if (epoch + 1) % 20 == 0:
            elapsed_time = time.time() - start_time
            print(
                f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Time: {elapsed_time:.2f}s')

    # 加载最优模型
    model.load_state_dict(torch.load(best_model_path))
    print(f'{model_name}训练完成，最优验证损失: {best_val_loss:.6f}')

    return model, train_losses, val_losses


# ==================== 5. 测试与评估函数 ====================
def evaluate_model(model, test_loader, scaler_target):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(x_batch)

            # 反归一化
            pred = outputs.cpu().numpy()
            actual = y_batch.cpu().numpy()

            # 扩展维度以匹配scaler的输入格式
            pred_reshaped = pred.reshape(-1, 1)
            actual_reshaped = actual.reshape(-1, 1)

            pred_inv = scaler_target.inverse_transform(pred_reshaped)
            actual_inv = scaler_target.inverse_transform(actual_reshaped)

            predictions.extend(pred_inv.flatten())
            actuals.extend(actual_inv.flatten())

    # 计算评估指标
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    pearson_corr, _ = pearsonr(actuals, predictions)

    print(f'测试集评估结果：')
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'R²: {r2:.4f}')
    print(f'Pearson相关系数: {pearson_corr:.4f}')

    return predictions, actuals, {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'Pearson': pearson_corr}


# ==================== 6. 可视化函数 ====================
def plot_results(actuals, lstm_preds, bilstm_preds, model_metrics):
    # 绘制预测vs实际
    plt.figure(figsize=(15, 10))

    # 预测结果对比（取前500个点更清晰）
    plt.subplot(2, 2, 1)
    plt.plot(actuals[:500], label='实际风速', color='blue')
    plt.plot(lstm_preds[:500], label='标准LSTM预测', color='red', alpha=0.7)
    plt.plot(bilstm_preds[:500], label='BiLSTM预测', color='green', alpha=0.7)
    plt.xlabel('时间步')
    plt.ylabel('风速 (m/s)')
    plt.title('风速预测结果对比')
    plt.legend()
    plt.grid(True)

    # 误差分布
    plt.subplot(2, 2, 2)
    lstm_error = lstm_preds - actuals
    bilstm_error = bilstm_preds - actuals
    plt.hist(lstm_error, bins=50, alpha=0.5, label='标准LSTM误差')
    plt.hist(bilstm_error, bins=50, alpha=0.5, label='BiLSTM误差')
    plt.xlabel('预测误差')
    plt.ylabel('频次')
    plt.title('预测误差分布')
    plt.legend()
    plt.grid(True)

    # 指标对比柱状图
    plt.subplot(2, 2, 3)
    metrics = ['MAE', 'RMSE', 'R²', 'Pearson']
    lstm_vals = [model_metrics['LSTM'][m] for m in metrics]
    bilstm_vals = [model_metrics['BiLSTM'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width / 2, lstm_vals, width, label='标准LSTM')
    plt.bar(x + width / 2, bilstm_vals, width, label='BiLSTM')
    plt.xlabel('评估指标')
    plt.ylabel('值')
    plt.title('模型评估指标对比')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, axis='y')

    # 散点图：预测vs实际
    plt.subplot(2, 2, 4)
    plt.scatter(actuals[:500], lstm_preds[:500], alpha=0.5, label='标准LSTM', s=10)
    plt.scatter(actuals[:500], bilstm_preds[:500], alpha=0.5, label='BiLSTM', s=10)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'k--', lw=2)
    plt.xlabel('实际风速')
    plt.ylabel('预测风速')
    plt.title('预测值vs实际值')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('wind_speed_prediction_results.png', dpi=300)
    plt.show()


# ==================== 7. 主流程执行 ====================
if __name__ == '__main__':
    try:
        # ==================== 数据读取与预处理 ====================
        print('正在读取风速数据...')
        filename = 'winddata.xlsx'
        data = pd.read_excel(filename)

        feature_columns = ['Wind Direction', 'Theoretical_Power_Curve (KWh)', 'LV ActivePower (kW)', 'Wind Speed (m/s)']
        target_column = 'Wind Speed (m/s)'

        # 检查列是否存在
        for col in feature_columns:
            if col not in data.columns:
                raise ValueError(f'未找到特征列: {col}')

        # 提取特征和目标
        features = data[feature_columns].values
        target = data[target_column].values.reshape(-1, 1)  # 目标需二维用于归一化

        # 归一化（论文用MinMaxScaler）
        scaler_features = MinMaxScaler(feature_range=(0, 1))
        scaler_target = MinMaxScaler(feature_range=(0, 1))

        features_scaled = scaler_features.fit_transform(features)
        target_scaled = scaler_target.fit_transform(target)

        # 保存scaler以便后续反归一化
        with open('scaler_features.pkl', 'wb') as f:
            pickle.dump(scaler_features, f)
        with open('scaler_target.pkl', 'wb') as f:
            pickle.dump(scaler_target, f)

        # ==================== 构建数据集和数据加载器 ====================
        full_dataset = WindTimeSeriesDataset(
            features_scaled,
            target_scaled,
            SEQUENCE_LENGTH,
            PREDICTION_STEPS
        )

        # 时序划分：训练70%，验证15%，测试15%（不打乱）
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        train_indices = list(range(train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, total_size))

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)

        # 数据加载器（严格匹配论文batch_size）
        train_loader_lstm = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False)
        val_loader_lstm = DataLoader(val_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 测试batch_size不影响

        train_loader_bilstm = DataLoader(train_dataset, batch_size=BILSTM_BATCH_SIZE, shuffle=False)
        val_loader_bilstm = DataLoader(val_dataset, batch_size=BILSTM_BATCH_SIZE, shuffle=False)

        print(f'数据集划分完成：训练集{len(train_dataset)}，验证集{len(val_dataset)}，测试集{len(test_dataset)}')

        # ==================== 初始化模型 ====================
        input_dim = len(feature_columns)
        output_dim = PREDICTION_STEPS

        # 标准LSTM（论文参数）
        lstm_model = StandardLSTM(
            input_dim=input_dim,
            hidden_size=LSTM_HIDDEN_SIZE,
            output_dim=output_dim,
            dropout=LSTM_DROPOUT,
            recurrent_dropout=LSTM_RECURRENT_DROPOUT
        ).to(DEVICE)

        # BiLSTM（论文参数）
        bilstm_model = BiLSTMModel(
            input_dim=input_dim,
            hidden_size=BILSTM_HIDDEN_SIZE,
            output_dim=output_dim,
            dropout=LSTM_DROPOUT,
            recurrent_dropout=LSTM_RECURRENT_DROPOUT
        ).to(DEVICE)

        # 损失函数和优化器（论文用MSE和Adam）
        criterion = nn.MSELoss()
        optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-07)
        optimizer_bilstm = optim.Adam(bilstm_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-07)

        # ==================== 训练模型 ====================
        print('\n开始训练标准LSTM模型...')
        lstm_model, lstm_train_losses, lstm_val_losses = train_model(
            lstm_model, train_loader_lstm, val_loader_lstm, criterion, optimizer_lstm, LSTM_EPOCHS, 'StandardLSTM'
        )

        print('\n开始训练BiLSTM模型...')
        bilstm_model, bilstm_train_losses, bilstm_val_losses = train_model(
            bilstm_model, train_loader_bilstm, val_loader_bilstm, criterion, optimizer_bilstm, BILSTM_EPOCHS, 'BiLSTM'
        )

        # ==================== 评估模型 ====================
        print('\n评估标准LSTM模型...')
        lstm_preds, actuals, lstm_metrics = evaluate_model(lstm_model, test_loader, scaler_target)

        print('\n评估BiLSTM模型...')
        bilstm_preds, _, bilstm_metrics = evaluate_model(bilstm_model, test_loader, scaler_target)

        # ==================== 可视化结果 ====================
        model_metrics = {
            'LSTM': lstm_metrics,
            'BiLSTM': bilstm_metrics
        }
        plot_results(actuals, lstm_preds, bilstm_preds, model_metrics)

        # ==================== 保存结果 ====================
        results = {
            'actuals': actuals,
            'lstm_predictions': lstm_preds,
            'bilstm_predictions': bilstm_preds,
            'metrics': model_metrics,
            'lstm_train_losses': lstm_train_losses,
            'lstm_val_losses': lstm_val_losses,
            'bilstm_train_losses': bilstm_train_losses,
            'bilstm_val_losses': bilstm_val_losses
        }

        with open('prediction_results.pkl', 'wb') as f:
            pickle.dump(results, f)

        print('\n所有结果已保存！')

    except Exception as e:
        print(f'程序执行出错: {e}')
        traceback.print_exc()