#!/usr/bin/env python3
"""
极端风速回归预测 - LSTM-Transformer模型
基于论文方法论，但仅保留回归分支
评估指标: RMSE, MAE, MAPE, R (相关系数)
损失函数: MSE
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
# ==================== 1. 数据读取与预处理 ====================
print('正在读取风速数据...')
filename = 'winddata.xlsx'
try:
    data = pd.read_excel(filename, index_col=0, parse_dates=True)
    required_columns = ['Wind Direction', 'Theoretical_Power_Curve (KWh)',
                        'LV ActivePower (kW)', 'Wind Speed (m/s)']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f'未找到特征列: {col}')

    print(f"数据加载成功! 形状: {data.shape}")
    print(f"时间范围: {data.index.min()} 到 {data.index.max()}")
    print(f"列名: {list(data.columns)}")

except Exception as e:
    print(f'无法读取 Excel 文件: {e}')
    raise
# ==================== 2. 特征工程 ====================
print('正在进行特征工程...')
# 构建最终特征矩阵
feature_columns = ['Wind Direction', 'Theoretical_Power_Curve (KWh)',
                   'LV ActivePower (kW)', 'Wind Speed (m/s)']
final_features = [col for col in feature_columns if col in data.columns]
print(f"使用的特征: {final_features}")
feature_data = data[final_features].values
wind_speed = data['Wind Speed (m/s)'].values
# ==================== 3. 数据准备 ====================
print('准备训练数据...')


class WindRegressionDataset(Dataset):
    """风速回归数据集类"""

    def __init__(self, features, targets, history_length=36, forecast_horizon=6):
        self.features = features
        self.targets = targets
        self.history_length = history_length
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.features) - self.history_length - self.forecast_horizon + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.history_length]
        y = self.targets[idx + self.history_length:
                         idx + self.history_length + self.forecast_horizon]

        return {
            'features': torch.FloatTensor(x),
            'target': torch.FloatTensor(y)
        }


def prepare_data(features, wind_speed, train_ratio=0.7, val_ratio=0.15):
    """准备训练和验证数据"""

    # Manual MinMax scaling for features
    feature_min = np.min(features, axis=0)
    feature_max = np.max(features, axis=0)
    features_scaled = (features - feature_min) / (feature_max - feature_min + 1e-10)

    wind_speed_reshaped = wind_speed.reshape(-1, 1)
    target_min = np.min(wind_speed_reshaped)
    target_max = np.max(wind_speed_reshaped)
    target_scaled = ((wind_speed_reshaped - target_min) / (target_max - target_min + 1e-10)).flatten()

    total_size = len(features_scaled)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_features = features_scaled[:train_size]
    val_features = features_scaled[train_size:train_size + val_size]
    test_features = features_scaled[train_size + val_size:]

    train_target = target_scaled[:train_size]
    val_target = target_scaled[train_size:train_size + val_size]
    test_target = target_scaled[train_size + val_size:]

    return (train_features, val_features, test_features,
            train_target, val_target, test_target,
            target_min, target_max)


# 准备数据
(train_features, val_features, test_features,
 train_target, val_target, test_target,
 target_min, target_max) = prepare_data(feature_data, wind_speed)
print(f"训练集大小: {len(train_features)}")
print(f"验证集大小: {len(val_features)}")
print(f"测试集大小: {len(test_features)}")
# ==================== 4. 模型定义 ====================
print('定义LSTM-Transformer回归模型...')


class LSTMTransformerRegression(nn.Module):
    """LSTM-Transformer回归模型"""

    def __init__(self, input_dim, hidden_dim=16, num_layers=2,
                 forecast_horizon=6, num_heads=8, dropout=0.1):
        super(LSTMTransformerRegression, self).__init__()

        self.forecast_horizon = forecast_horizon

        # LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 回归输出分支
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_horizon)
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """He Normal初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Transformer
        transformer_out = self.transformer(lstm_out)

        # 取最后一个时间步
        last_hidden = transformer_out[:, -1, :]

        # 回归输出
        reg_output = self.regression_head(last_hidden)

        return reg_output


# ==================== 5. 训练函数 ====================
def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    """训练模型"""

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print('开始训练...')
    for epoch in tqdm(range(epochs), desc="训练进度"):
        # 训练阶段
        model.train()
        train_loss = 0

        for batch in train_loader:
            features = batch['features'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()

            # 前向传播
            output = model(features)

            # 计算损失
            loss = criterion(output, target)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                target = batch['target'].to(device)

                output = model(features)
                loss = criterion(output, target)
                val_loss += loss.item()

        # 计算平均损失
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_regression_model.pth')

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    return model, train_losses, val_losses


# ==================== 6. 评估函数 ====================
def evaluate_model(model, test_loader, device, target_min, target_max):
    """评估模型性能 - 仅回归指标"""
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)

            predictions = model(features)

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

    # 合并所有批次
    predictions = torch.cat(all_predictions, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    # 反归一化
    predictions = predictions * (target_max - target_min) + target_min
    targets = targets * (target_max - target_min) + target_min

    # 计算评估指标
    mse = np.mean((targets.flatten() - predictions.flatten()) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(targets.flatten() - predictions.flatten()))

    # MAPE (处理零值)
    mask = targets.flatten() != 0
    mape = np.mean(np.abs((targets.flatten()[mask] - predictions.flatten()[mask]) /
                          targets.flatten()[mask])) * 100

    # 相关系数R
    correlation_matrix = np.corrcoef(targets.flatten(), predictions.flatten())
    r = correlation_matrix[0, 1]

    # 每步的指标
    rmse_per_step = []
    mae_per_step = []
    for i in range(predictions.shape[1]):
        step_mse = np.mean((targets[:, i] - predictions[:, i]) ** 2)
        rmse_per_step.append(np.sqrt(step_mse))
        mae_per_step.append(np.mean(np.abs(targets[:, i] - predictions[:, i])))

    results = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r': r,
        'rmse_per_step': rmse_per_step,
        'mae_per_step': mae_per_step,
        'predictions': predictions,
        'targets': targets
    }

    return results


# ==================== 7. 可视化函数 ====================
def plot_results(results, forecast_horizon=6):
    """可视化预测结果"""
    pred = results['predictions']
    true = results['targets']

    # 随机选择几个样本可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i in range(4):
        idx = np.random.randint(0, len(true))
        time_steps = np.arange(forecast_horizon) * 10  # 10分钟间隔

        axes[i].plot(time_steps, true[idx, :], 'b-o', label='真实值', linewidth=2)
        axes[i].plot(time_steps, pred[idx, :], 'r--*', label='预测值', linewidth=2)
        axes[i].set_xlabel('预测时间 (分钟)', fontsize=12)
        axes[i].set_ylabel('风速 (m/s)', fontsize=12)
        axes[i].set_title(f'样本 {idx + 1}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('prediction_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 误差分布图
    plt.figure(figsize=(10, 6))
    errors = (pred - true).flatten()
    plt.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('预测误差 (m/s)', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.title('预测误差分布', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(true.flatten(), pred.flatten(), alpha=0.6, s=10, color='steelblue')
    plt.plot([true.min(), true.max()], [true.min(), true.max()], 'r--', lw=2)
    plt.xlabel('真实风速 (m/s)', fontsize=12)
    plt.ylabel('预测风速 (m/s)', fontsize=12)
    plt.title('真实值 vs 预测值', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


# ==================== 8. 主执行代码 ====================
def main():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 超参数 (与论文一致)
    HISTORY_LENGTH = 36  # 36个时间步 (360分钟)
    FORECAST_HORIZON = 6  # 6步 = 60分钟
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 0.001

    # 创建数据集
    train_dataset = WindRegressionDataset(train_features, train_target,
                                          history_length=HISTORY_LENGTH,
                                          forecast_horizon=FORECAST_HORIZON)
    val_dataset = WindRegressionDataset(val_features, val_target,
                                        history_length=HISTORY_LENGTH,
                                        forecast_horizon=FORECAST_HORIZON)
    test_dataset = WindRegressionDataset(test_features, test_target,
                                         history_length=HISTORY_LENGTH,
                                         forecast_horizon=FORECAST_HORIZON)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}, 测试批次: {len(test_loader)}")

    # 创建模型
    input_dim = train_features.shape[1]
    model = LSTMTransformerRegression(
        input_dim=input_dim,
        hidden_dim=16,
        forecast_horizon=FORECAST_HORIZON
    ).to(device)

    # 训练模型
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, device,
        epochs=EPOCHS, lr=LEARNING_RATE
    )

    # 加载最佳模型
    model.load_state_dict(torch.load('best_regression_model.pth'))

    # 评估模型
    print('评估模型性能...')
    results = evaluate_model(model, test_loader, device, target_min, target_max)

    # 打印结果
    print("\n" + "=" * 50)
    print("回归模型性能评估结果")
    print("=" * 50)
    print(f"RMSE: {results['rmse']:.4f} m/s")
    print(f"MAE: {results['mae']:.4f} m/s")
    print(f"MAPE: {results['mape']:.2f}%")
    print(f"R: {results['r']:.4f}")
    print("\n各步RMSE:")
    for i, rmse in enumerate(results['rmse_per_step']):
        print(f" 第{i + 1}步 (10分钟): {rmse:.4f} m/s")
    print("\n各步MAE:")
    for i, mae in enumerate(results['mae_per_step']):
        print(f" 第{i + 1}步 (10分钟): {mae:.4f} m/s")

    # 可视化
    plot_results(results, forecast_horizon=FORECAST_HORIZON)

    # 保存训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失 (MSE)')
    plt.plot(val_losses, label='验证损失 (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE损失')
    plt.title('训练曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()