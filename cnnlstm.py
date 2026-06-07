import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import pickle
import time
import warnings
import traceback
from scipy.stats import pearsonr
from PyEMD import CEEMDAN

warnings.filterwarnings('ignore')
# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False


# ==================== 配置参数（与论文完全一致） ====================
class Config:
    # 数据参数
    SEQ_LEN = 24  # 过去3天数据（3h分辨率）
    PRED_STEPS = 4  # 4步预测（12小时）
    IMF_NUM = 6  # CEEMDAN分解数量
    TEST_RATIO = 0.1  # 测试集比例
    VAL_RATIO = 0.1  # 验证集比例

    # 模型参数
    CNN_FILTERS = 4
    CNN_KERNEL_SIZE = 2
    LSTM_UNITS1 = 16
    LSTM_UNITS2 = 16
    DROPOUT_RATE = 0.1
    FC_UNITS = 4
    ACTIVATION = 'tanh'

    # 训练参数
    BATCH_SIZE = 2
    EPOCHS = 300
    INITIAL_LR = 0.008
    PATIENCE_LR = 5  # 学习率衰减耐心值
    PATIENCE_ES = 15  # 早停耐心值
    OPTIMIZER = 'Adam'

    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 随机种子
    SEED = 42


# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


# ==================== 1. 数据读取与预处理 ====================
print('正在读取风速数据...')
filename = 'winddata.xlsx'
try:
    data = pd.read_excel(filename)
    feature_columns = ['Wind Direction', 'Theoretical_Power_Curve (KWh)', 'LV ActivePower (kW)', 'Wind Speed (m/s)']
    target_column = 'Wind Speed (m/s)'
    for col in feature_columns:
        if col not in data.columns:
            raise ValueError(f'未找到特征列: {col}')
    feature_data = data[feature_columns].values
    print(f'数据加载成功，共 {len(data)} 条记录')
except Exception as e:
    print(f'无法读取 Excel 文件: {e}')
    traceback.print_exc()
    raise


# ==================== 2. CEEMDAN分解 ====================
def perform_ceemdan_decomposition(data, max_imf=6):
    """执行CEEMDAN分解"""
    print('正在执行CEEMDAN分解...')
    ceemdan = CEEMDAN(trials=100, epsilon=0.005)
    imfs = ceemdan(data, max_imf=max_imf)
    if len(imfs) < max_imf:
        # 如果分解数量不足，补零
        padding = np.zeros((max_imf - len(imfs), len(data)))
        imfs = np.vstack([imfs, padding])
    residue = data - np.sum(imfs[:max_imf], axis=0)
    print(f'CEEMDAN分解完成，获得 {max_imf} 个IMF')
    return imfs[:max_imf], residue


# 提取风速序列进行分解
wind_speed = data[target_column].values
imfs, residue = perform_ceemdan_decomposition(wind_speed, max_imf=Config.IMF_NUM)


# 可视化分解结果
def plot_imfs(imfs, residue, original, save_path='imfs_decomposition.png'):
    plt.figure(figsize=(15, 12))
    plt.subplot(Config.IMF_NUM + 2, 1, 1)
    plt.plot(original, 'k')
    plt.title('Original Wind Speed (m/s)')
    plt.xlim(0, len(original))

    for i, imf in enumerate(imfs, 1):
        plt.subplot(Config.IMF_NUM + 2, 1, i + 1)
        plt.plot(imf)
        plt.title(f'IMF {i}')
        plt.xlim(0, len(imf))

    plt.subplot(Config.IMF_NUM + 2, 1, Config.IMF_NUM + 2)
    plt.plot(residue)
    plt.title('Residue')
    plt.xlim(0, len(residue))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


plot_imfs(imfs, residue, wind_speed)


# ==================== 3. 数据集构建 ====================
class CustomMinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit_transform(self, X):
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return (X - self.min_) / (self.max_ - self.min_ + 1e-8)

    def inverse_transform(self, X):
        return X * (self.max_ - self.min_ + 1e-8) + self.min_


class WindSpeedDataset(Dataset):
    """风速数据集"""

    def __init__(self, features, imfs, seq_len, pred_steps):
        self.features = features
        self.imfs = imfs
        self.seq_len = seq_len
        self.pred_steps = pred_steps

        # 对每个IMF进行归一化
        self.scalers = [CustomMinMaxScaler() for _ in range(imfs.shape[0] + 1)]
        self.normalized_imfs = []
        for i, imf in enumerate(imfs):
            imf_reshaped = imf.reshape(-1, 1)
            self.normalized_imfs.append(self.scalers[i].fit_transform(imf_reshaped).flatten())

        # 对特征进行归一化
        self.normalized_features = self.scalers[-1].fit_transform(features)

    def __len__(self):
        return len(self.features) - self.seq_len - self.pred_steps + 1

    def __getitem__(self, idx):
        # 输入特征: [seq_len, feature_dim]
        x_features = self.normalized_features[idx:idx + self.seq_len]

        # 每个IMF的目标值
        y_imfs = []
        for i in range(self.imfs.shape[0]):
            y_imf = self.normalized_imfs[i][idx + self.seq_len:idx + self.seq_len + self.pred_steps]
            y_imfs.append(y_imf)

        # 合并所有IMF目标
        y = np.stack(y_imfs, axis=1)  # [pred_steps, IMF_NUM]

        return {
            'x': torch.FloatTensor(x_features),
            'y': torch.FloatTensor(y)
        }


# 创建完整数据集
dataset = WindSpeedDataset(feature_data, imfs, Config.SEQ_LEN, Config.PRED_STEPS)
# 划分训练集、验证集、测试集
total_size = len(dataset)
test_size = int(total_size * Config.TEST_RATIO)
val_size = int(total_size * Config.VAL_RATIO)
train_size = total_size - test_size - val_size
indices = list(range(total_size))
np.random.shuffle(indices)
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]
train_loader = DataLoader(Subset(dataset, train_indices), batch_size=Config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_indices), batch_size=Config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(Subset(dataset, test_indices), batch_size=Config.BATCH_SIZE, shuffle=False)
print(f'数据集划分: 训练集 {train_size}, 验证集 {val_size}, 测试集 {test_size}')


# ==================== 4. CNN-LSTM模型定义 ====================
class CNN_LSTM_Model(nn.Module):
    """与论文完全一致的CNN-LSTM模型"""

    def __init__(self, input_dim, pred_steps, imf_num):
        super(CNN_LSTM_Model, self).__init__()
        self.pred_steps = pred_steps
        self.imf_num = imf_num

        # 1D CNN层
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=Config.CNN_FILTERS,
            kernel_size=Config.CNN_KERNEL_SIZE,
            stride=1,
            padding=0
        )

        # LSTM层1
        self.lstm1 = nn.LSTM(
            input_size=Config.CNN_FILTERS,
            hidden_size=Config.LSTM_UNITS1,
            batch_first=True,
            dropout=Config.DROPOUT_RATE
        )

        # LSTM层2
        self.lstm2 = nn.LSTM(
            input_size=Config.LSTM_UNITS1,
            hidden_size=Config.LSTM_UNITS2,
            batch_first=True
        )

        # 全连接层
        self.fc = nn.Linear(
            in_features=Config.LSTM_UNITS2,
            out_features=pred_steps * imf_num
        )

        # 激活函数
        self.activation = nn.Tanh()

    def forward(self, x):
        # x: [batch, seq_len, feature_dim]
        batch_size = x.size(0)

        # CNN需要输入 [batch, channels, seq_len]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.activation(x)

        # 转换回 [batch, seq_len, channels] 用于LSTM
        x = x.permute(0, 2, 1)

        # LSTM层
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # 取最后时间步
        x = x[:, -1, :]

        # 全连接层
        x = self.fc(x)

        # 重塑为 [batch, pred_steps, imf_num]
        x = x.view(batch_size, self.pred_steps, self.imf_num)

        return x


# ==================== 5. 训练函数 ====================
def train_model(model, train_loader, val_loader, imf_index):
    """训练单个IMF的模型"""
    model = model.to(Config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.INITIAL_LR)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=Config.PATIENCE_LR
    )

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    print(f'开始训练 IMF {imf_index + 1}...')

    for epoch in range(Config.EPOCHS):
        # 训练阶段
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch['x'].to(Config.DEVICE)
            y = batch['y'].to(Config.DEVICE)

            # 只预测当前IMF
            y_imf = y[:, :, imf_index].unsqueeze(-1)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs[:, :, imf_index].unsqueeze(-1), y_imf)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # 验证阶段
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(Config.DEVICE)
                y = batch['y'].to(Config.DEVICE)

                y_imf = y[:, :, imf_index].unsqueeze(-1)
                outputs = model(x)
                val_loss = criterion(outputs[:, :, imf_index].unsqueeze(-1), y_imf)
                val_losses.append(val_loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # 学习率调度
        scheduler.step(avg_val_loss)

        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f'best_model_imf_{imf_index}.pth')
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{Config.EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

        if patience_counter >= Config.PATIENCE_ES:
            print(f'提前停止于 epoch {epoch + 1}')
            break

    # 加载最佳模型
    model.load_state_dict(torch.load(f'best_model_imf_{imf_index}.pth'))
    return model, history


# ==================== 6. 训练所有IMF模型 ====================
def train_all_imfs():
    """训练所有IMF的模型"""
    models = []
    histories = []

    for i in range(Config.IMF_NUM):
        print(f'\n{"=" * 50}')
        print(f'训练第 {i + 1}/{Config.IMF_NUM} 个IMF模型')
        print(f'{"=" * 50}')

        model = CNN_LSTM_Model(
            input_dim=feature_data.shape[1],
            pred_steps=Config.PRED_STEPS,
            imf_num=Config.IMF_NUM
        )

        trained_model, history = train_model(model, train_loader, val_loader, i)
        models.append(trained_model)
        histories.append(history)

        # 可视化训练过程
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title(f'IMF {i + 1} Training History')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'training_history_imf_{i}.png', dpi=300, bbox_inches='tight')
        plt.show()

    return models


# ==================== 7. 多步预测与重建 ====================
def multi_step_forecast(models, data_loader):
    """执行多步预测"""
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in data_loader:
            x = batch['x'].to(Config.DEVICE)
            y = batch['y']

            batch_pred = []
            for i, model in enumerate(models):
                model.eval()
                pred = model(x)
                batch_pred.append(pred[:, :, i].cpu().numpy())

            # 合并所有IMF预测
            batch_pred = np.stack(batch_pred, axis=-1)  # [batch, pred_steps, imf_num]
            predictions.append(batch_pred)
            actuals.append(y.numpy())

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    return predictions, actuals


def reconstruct_wind_speed(predictions, actuals):
    """重建风速并反归一化"""
    # 对每个IMF反归一化并求和
    pred_reconstructed = np.zeros(predictions.shape[:2])  # [batch, pred_steps]
    actual_reconstructed = np.zeros(actuals.shape[:2])

    for i in range(Config.IMF_NUM):
        # 反归一化
        pred_imf = dataset.scalers[i].inverse_transform(
            predictions[:, :, i].reshape(-1, 1)
        ).reshape(predictions.shape[:2])
        actual_imf = dataset.scalers[i].inverse_transform(
            actuals[:, :, i].reshape(-1, 1)
        ).reshape(actuals.shape[:2])

        pred_reconstructed += pred_imf
        actual_reconstructed += actual_imf

    return pred_reconstructed, actual_reconstructed


# ==================== 8. 评估指标 ====================
def evaluate_performance(y_true, y_pred):
    """计算评估指标"""
    # 确保输入是扁平化的
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # 基础指标
    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100

    # 新增指标
    # R²
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # 相关系数
    corr, _ = pearsonr(y_true_flat, y_pred_flat)

    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2,
        'Correlation': corr
    }


def calculate_prediction_interval(y_true, y_pred, alpha=0.05):
    """计算预测区间"""
    residuals = y_true.flatten() - y_pred.flatten()
    std_error = np.std(residuals)
    z_score = 1.96  # 95%置信区间

    pi_width = 2 * z_score * std_error
    avg_pi = np.mean([pi_width] * len(y_pred.flatten()))

    return avg_pi


# ==================== 9. 可视化函数 ====================
def plot_forecast_results(pred_ws, actual_ws, save_path='forecast_results.png'):
    """基础预测结果可视化 - 随机3个样本"""
    plt.figure(figsize=(20, 10))

    # 随机选择几个样本可视化
    sample_indices = np.random.choice(len(pred_ws), 3, replace=False)

    for i, idx in enumerate(sample_indices):
        plt.subplot(3, 1, i + 1)
        plt.plot(actual_ws[idx], 'k-', label='真实值', linewidth=2)
        plt.plot(pred_ws[idx], 'r--', label='预测值', linewidth=1.5)
        plt.title(f'样本 {idx + 1} - 4步预测 (12小时)', fontsize=14)
        plt.xlabel('时间步 (3小时间隔)', fontsize=12)
        plt.ylabel('风速 (m/s)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_detailed_comparison(pred_ws, actual_ws, num_samples=5, save_path='detailed_comparison.png'):
    """详细对比图 - 展示多个样本的预测与真实值"""
    fig, axes = plt.subplots(num_samples, 1, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    sample_indices = np.random.choice(len(pred_ws), num_samples, replace=False)

    for idx, ax in zip(sample_indices, axes):
        # 绘制每个时间步
        time_steps = np.arange(Config.PRED_STEPS)
        ax.plot(time_steps, actual_ws[idx], 'k-o', label='真实值', linewidth=2, markersize=6)
        ax.plot(time_steps, pred_ws[idx], 'r-s', label='预测值', linewidth=1.5, markersize=4, alpha=0.8)

        # 添加误差带
        error = np.abs(actual_ws[idx] - pred_ws[idx])
        ax.fill_between(time_steps, pred_ws[idx] - error, pred_ws[idx] + error,
                        alpha=0.2, color='red', label='绝对误差带')

        ax.set_title(f'测试样本 {idx} - 风速多步预测对比', fontsize=13)
        ax.set_xlabel('预测时间步 (3h)', fontsize=11)
        ax.set_ylabel('风速 (m/s)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_scatter_comparison(pred_ws, actual_ws, save_path='scatter_comparison.png'):
    """散点对比图 - 所有预测点 vs 真实值"""
    plt.figure(figsize=(10, 10))

    # 扁平化所有数据
    all_actual = actual_ws.flatten()
    all_pred = pred_ws.flatten()

    # 绘制散点
    plt.scatter(all_actual, all_pred, alpha=0.6, s=20, c='blue', edgecolors='k', linewidth=0.5)

    # 绘制完美预测线
    min_val = min(all_actual.min(), all_pred.min())
    max_val = max(all_actual.max(), all_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测线')

    # 添加统计信息
    metrics = evaluate_performance(actual_ws, pred_ws)
    textstr = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.xlabel('真实风速 (m/s)', fontsize=12)
    plt.ylabel('预测风速 (m/s)', fontsize=12)
    plt.title('所有预测点：真实值 vs 预测值', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_residual_analysis(pred_ws, actual_ws, save_path='residual_analysis.png'):
    """残差分析图"""
    residuals = actual_ws.flatten() - pred_ws.flatten()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 残差直方图
    axes[0, 0].hist(residuals, bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('残差分布直方图', fontsize=12)
    axes[0, 0].set_xlabel('残差 (m/s)', fontsize=11)
    axes[0, 0].set_ylabel('频数', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 残差Q-Q图
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q图 (正态性检验)', fontsize=12)

    # 3. 残差时序图 (前200个点)
    axes[1, 0].plot(residuals[:200], 'b-', alpha=0.7)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1, 0].set_title('残差时序图 (前200点)', fontsize=12)
    axes[1, 0].set_xlabel('样本点', fontsize=11)
    axes[1, 0].set_ylabel('残差 (m/s)', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 残差 vs 预测值
    axes[1, 1].scatter(pred_ws.flatten(), residuals, alpha=0.5, s=15, c='green')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1, 1].set_title('残差 vs 预测值', fontsize=12)
    axes[1, 1].set_xlabel('预测风速 (m/s)', fontsize=11)
    axes[1, 1].set_ylabel('残差 (m/s)', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_prediction_intervals(pred_ws, actual_ws, save_path='prediction_intervals.png'):
    """预测区间可视化"""
    # 计算残差标准差
    residuals = actual_ws.flatten() - pred_ws.flatten()
    std_error = np.std(residuals)

    # 95%置信区间
    z_score = 1.96
    interval = z_score * std_error

    # 随机选择几个样本
    sample_indices = np.random.choice(len(pred_ws), 3, replace=False)

    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(14, 4 * len(sample_indices)))
    if len(sample_indices) == 1:
        axes = [axes]

    for idx, ax in zip(sample_indices, axes):
        time_steps = np.arange(Config.PRED_STEPS)
        pred_line = pred_ws[idx]
        actual_line = actual_ws[idx]

        # 绘制预测区间
        ax.fill_between(time_steps, pred_line - interval, pred_line + interval,
                        alpha=0.3, color='orange', label='95%预测区间')

        # 绘制真实值和预测值
        ax.plot(time_steps, actual_line, 'k-o', label='真实值', linewidth=2, markersize=6)
        ax.plot(time_steps, pred_line, 'r-s', label='预测值', linewidth=1.5, markersize=4)

        ax.set_title(f'样本 {idx} - 带预测区间的风速预测', fontsize=13)
        ax.set_xlabel('预测时间步 (3h)', fontsize=11)
        ax.set_ylabel('风速 (m/s)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ==================== 10. 主执行流程 ====================
def main():
    print(f'开始训练，设备: {Config.DEVICE}')
    print(f'超参数配置: {vars(Config)}')

    # 训练所有IMF模型
    models = train_all_imfs()

    # 在测试集上预测
    print('\n正在执行多步预测...')
    test_predictions, test_actuals = multi_step_forecast(models, test_loader)

    # 重建风速
    print('正在重建风速...')
    pred_ws, actual_ws = reconstruct_wind_speed(test_predictions, test_actuals)

    # 评估性能
    print('\n' + '=' * 60)
    print('模型评估结果 - 最终合成风速')
    print('=' * 60)

    overall_metrics = evaluate_performance(actual_ws, pred_ws)

    # 以表格形式打印指标
    print(f"{'指标':<15} {'值':<15} {'说明':<30}")
    print('-' * 60)
    print(f"{'MSE':<15} {overall_metrics['MSE']:<15.6f} {'均方误差':<30}")
    print(f"{'MAE':<15} {overall_metrics['MAE']:<15.6f} {'平均绝对误差':<30}")
    print(f"{'RMSE':<15} {overall_metrics['RMSE']:<15.6f} {'均方根误差':<30}")
    print(f"{'MAPE':<15} {overall_metrics['MAPE']:<15.4f}% {'平均绝对百分比误差':<30}")
    print(f"{'R²':<15} {overall_metrics['R²']:<15.6f} {'决定系数':<30}")
    print(f"{'Correlation':<15} {overall_metrics['Correlation']:<15.6f} {'相关系数':<30}")

    # 计算预测区间
    avg_pi = calculate_prediction_interval(actual_ws, pred_ws)
    print(f"{'Avg PI Width':<15} {avg_pi:<15.6f} {'平均预测区间宽度':<30}")
    print('=' * 60)

    # 保存结果
    results = {
        'predictions': pred_ws,
        'actuals': actual_ws,
        'metrics': overall_metrics,
        'prediction_interval': avg_pi
    }

    with open('forecast_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print('\n生成可视化图表...')

    # 1. 基础预测结果对比
    plot_forecast_results(pred_ws, actual_ws, save_path='forecast_results.png')

    # 2. 详细对比图
    plot_detailed_comparison(pred_ws, actual_ws, num_samples=5, save_path='detailed_comparison.png')

    # 3. 散点对比图
    plot_scatter_comparison(pred_ws, actual_ws, save_path='scatter_comparison.png')

    # 4. 残差分析
    plot_residual_analysis(pred_ws, actual_ws, save_path='residual_analysis.png')

    # 5. 预测区间可视化
    plot_prediction_intervals(pred_ws, actual_ws, save_path='prediction_intervals.png')

    # 保存IMF分量预测结果用于分析
    def save_imf_predictions(predictions, actuals):
        """保存每个IMF的预测结果"""
        imf_metrics = []
        for i in range(Config.IMF_NUM):
            pred_imf = dataset.scalers[i].inverse_transform(
                predictions[:, :, i].reshape(-1, 1)
            ).flatten()
            actual_imf = dataset.scalers[i].inverse_transform(
                actuals[:, :, i].reshape(-1, 1)
            ).flatten()

            metrics = evaluate_performance(actual_imf.reshape(-1, 1), pred_imf.reshape(-1, 1))
            metrics['IMF'] = i + 1
            imf_metrics.append(metrics)

            print(f'\nIMF {i + 1} 评估:')
            for key, value in metrics.items():
                if key != 'IMF':
                    print(f' {key}: {value:.4f}')

        # 保存到CSV
        imf_df = pd.DataFrame(imf_metrics)
        imf_df.to_csv('imf_metrics.csv', index=False)

    save_imf_predictions(test_predictions, test_actuals)

    print('\n' + '=' * 60)
    print('所有任务完成！可视化图表已保存为PNG文件')
    print('=' * 60)


if __name__ == '__main__':
    main()