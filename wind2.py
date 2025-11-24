import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from copy import deepcopy
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 数据读取与预处理
print('正在读取风速数据...')
filename = 'winddata.xlsx'

try:
    data = pd.read_excel(filename)
    feature_columns = ['Wind Direction', 'Theoretical_Power_Curve (KWh)',
                       'LV ActivePower (kW)', 'Wind Speed (m/s)']
    target_column = 'Wind Speed (m/s)'

    # 检查是否存在所有特征列
    found = True
    for col in feature_columns:
        if col not in data.columns:
            found = False
            break

    if not found:
        print('可用列名：')
        for col in data.columns:
            print(f'  {col}')
        raise ValueError('未找到所有特征列。')

    # 提取特征数据
    feature_data = data[feature_columns].values

except Exception as e:
    print(f'无法读取 Excel 文件: {e}')
    raise

# 数据趋势可视化
print('绘制数据总体趋势图...')
n_samples = feature_data.shape[0]
time_axis = np.arange(n_samples)

plt.figure(figsize=(12, 10))
plt.suptitle('各特征数据趋势', fontsize=16)

for i, col in enumerate(feature_columns):
    plt.subplot(len(feature_columns), 1, i + 1)
    plt.plot(time_axis, feature_data[:, i], linewidth=1.2)
    plt.title(f'{col} 随时间变化趋势', fontsize=12)
    plt.xlabel('时间（样本点）', fontsize=10)
    plt.ylabel(col, fontsize=10)
    plt.grid(True)
    plt.box(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 数据清洗
valid_rows = np.all(~np.isnan(feature_data) & ~np.isinf(feature_data), axis=1)
feature_data = feature_data[valid_rows, :]

if feature_data.shape[0] < 100:
    raise ValueError('数据不足（少于100个样本），请提供更多数据。')

# 数据归一化
scaler = MinMaxScaler()
feature_data_norm = scaler.fit_transform(feature_data)
min_vals = scaler.data_min_
max_vals = scaler.data_max_


# 创建时间序列数据集
class WindDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length, :]  # 前sequence_length个时间步作为输入
        y = self.data[idx + self.sequence_length, -1]  # 下一个时间步的风速作为目标
        return torch.FloatTensor(x).transpose(0, 1), torch.FloatTensor([y])


sequence_length = 5
dataset = WindDataset(feature_data_norm, sequence_length)

# 核心修改：按6:1:1比例划分（训练集6/8，验证集1/8，测试集1/8），保持时间顺序
train_ratio = 6/8  # 62.5%
val_ratio = 1/8    # 12.5%
test_ratio = 1/8   # 12.5%

num_samples = len(dataset)
num_train = int(train_ratio * num_samples)
num_val = int(val_ratio * num_samples)
num_test = num_samples - num_train - num_val  # 确保总和等于样本数

# 时序划分：严格按时间顺序切割（训练集→验证集→测试集）
train_dataset = Subset(dataset, range(num_train))
val_dataset = Subset(dataset, range(num_train, num_train + num_val))
test_dataset = Subset(dataset, range(num_train + num_val, num_samples))

# 反归一化参数（目标变量为风速，对应最后一列）
min_speed = min_vals[-1]
max_speed = max_vals[-1]

print(f'数据预处理完成。训练样本数：{num_train}，验证样本数：{num_val}，测试样本数：{num_test}')
print(f'划分方式：按时间顺序（6:1:1），无时间交叉')


# 2. 模型定义
class FeatureWeightedLayer(nn.Module):
    def __init__(self, num_features):
        super(FeatureWeightedLayer, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_features, 1))

    def forward(self, x):
        # x shape: (batch_size, num_features, seq_len)
        weighted = x * self.weights  # 特征加权
        return weighted


class CNNBiLSTM(nn.Module):
    def __init__(self, num_features, num_filters1, filter_size1, num_filters2,
                 filter_size2, lstm_units1, lstm_units2, pool_type, num_conv_layers,
                 dropout_prob1, dropout_prob2, conv_dropout, activation_function):
        super(CNNBiLSTM, self).__init__()

        self.feature_weight = FeatureWeightedLayer(num_features)

        # 卷积层
        self.conv1 = nn.Conv1d(num_features, num_filters1, filter_size1, padding='same')
        self.bn1 = nn.BatchNorm1d(num_filters1)

        # 激活函数
        if activation_function == 1:
            self.act1 = nn.ReLU()
        elif activation_function == 2:
            self.act1 = nn.LeakyReLU(0.01)
        elif activation_function == 3:
            self.act1 = nn.Tanh()
        else:  # 4
            self.act1 = nn.Sigmoid()

        self.drop_conv1 = nn.Dropout(conv_dropout) if conv_dropout > 0 else nn.Identity()

        # 池化层
        if pool_type == 1:
            self.pool = nn.MaxPool1d(2, stride=1, padding=1)
        else:  # 2
            self.pool = nn.AvgPool1d(2, stride=1, padding=1)

        # 第二卷积层（可选）
        self.num_conv_layers = num_conv_layers
        if num_conv_layers == 2:
            self.conv2 = nn.Conv1d(num_filters1, num_filters2, filter_size2, padding='same')
            self.bn2 = nn.BatchNorm1d(num_filters2)

            if activation_function == 1:
                self.act2 = nn.ReLU()
            elif activation_function == 2:
                self.act2 = nn.LeakyReLU(0.01)
            elif activation_function == 3:
                self.act2 = nn.Tanh()
            else:  # 4
                self.act2 = nn.Sigmoid()

            self.drop_conv2 = nn.Dropout(conv_dropout) if conv_dropout > 0 else nn.Identity()
            rnn_input_size = num_filters2
        else:
            rnn_input_size = num_filters1

        # BiLSTM层
        self.bilstm1 = nn.LSTM(rnn_input_size, lstm_units1, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(dropout_prob1)
        self.bilstm2 = nn.LSTM(lstm_units1 * 2, lstm_units2, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(dropout_prob2)

        # 输出层
        self.fc_out = nn.Linear(lstm_units2 * 2, 1)

    def forward(self, x):
        # x shape: (batch_size, num_features, seq_len)

        # 特征加权
        x = self.feature_weight(x)

        # 卷积块1
        x = self.conv1(x)  # (batch_size, num_filters1, seq_len)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop_conv1(x)

        # 可选的卷积块2
        if self.num_conv_layers == 2:
            x = self.conv2(x)  # (batch_size, num_filters2, seq_len)
            x = self.bn2(x)
            x = self.act2(x)
            x = self.drop_conv2(x)

        # 池化
        x = self.pool(x)  # (batch_size, filters, seq_len)

        # 转换为LSTM输入格式 (batch_size, seq_len, features)
        x = x.transpose(1, 2)

        # BiLSTM层
        x, _ = self.bilstm1(x)  # (batch_size, seq_len, lstm_units1*2)
        x = self.drop1(x)
        x, _ = self.bilstm2(x)  # (batch_size, seq_len, lstm_units2*2)
        x = self.drop2(x)

        # 取最后一个时间步的输出
        x = x[:, -1, :]

        # 输出层
        x = self.fc_out(x)  # (batch_size, 1)
        return x


# 基准模型 - BiLSTM
class BiLSTM(nn.Module):
    def __init__(self, num_features, lstm_units1, lstm_units2, dropout_prob1, dropout_prob2):
        super(BiLSTM, self).__init__()
        self.feature_weight = FeatureWeightedLayer(num_features)
        self.bilstm1 = nn.LSTM(num_features, lstm_units1, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(dropout_prob1)
        self.bilstm2 = nn.LSTM(lstm_units1 * 2, lstm_units2, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(dropout_prob2)
        self.fc_out = nn.Linear(lstm_units2 * 2, 1)

    def forward(self, x):
        # x shape: (batch_size, num_features, seq_len)
        x = self.feature_weight(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, num_features)

        x, _ = self.bilstm1(x)
        x = self.drop1(x)
        x, _ = self.bilstm2(x)
        x = self.drop2(x)

        x = x[:, -1, :]
        x = self.fc_out(x)
        return x


# 基准模型 - GRU
class GRU(nn.Module):
    def __init__(self, num_features, gru_units1, gru_units2, dropout_prob1, dropout_prob2):
        super(GRU, self).__init__()
        self.feature_weight = FeatureWeightedLayer(num_features)
        self.gru1 = nn.GRU(num_features, gru_units1, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(dropout_prob1)
        self.gru2 = nn.GRU(gru_units1 * 2, gru_units2, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(dropout_prob2)
        self.fc_out = nn.Linear(gru_units2 * 2, 1)

    def forward(self, x):
        x = self.feature_weight(x)
        x = x.transpose(1, 2)

        x, _ = self.gru1(x)
        x = self.drop1(x)
        x, _ = self.gru2(x)
        x = self.drop2(x)

        x = x[:, -1, :]
        x = self.fc_out(x)
        return x


# 基准模型 - CNN
class CNN(nn.Module):
    def __init__(self, num_features, num_filters1, filter_size1, num_filters2, filter_size2):
        super(CNN, self).__init__()
        self.feature_weight = FeatureWeightedLayer(num_features)
        self.conv1 = nn.Conv1d(num_features, num_filters1, filter_size1, padding='same')
        self.bn1 = nn.BatchNorm1d(num_filters1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2, stride=1, padding=1)

        self.conv2 = nn.Conv1d(num_filters1, num_filters2, filter_size2, padding='same')
        self.bn2 = nn.BatchNorm1d(num_filters2)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2, stride=1, padding=1)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_filters2, 32)
        self.act3 = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.feature_weight(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.global_pool(x).squeeze(-1)
        x = self.fc1(x)
        x = self.act3(x)
        x = self.drop(x)
        x = self.fc_out(x)
        return x


# 基准模型 - Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerModel(nn.Module):
    def __init__(self, num_features, embedding_dim, num_heads, ffn_dim, seq_len):
        super(TransformerModel, self).__init__()
        self.feature_weight = FeatureWeightedLayer(num_features)
        self.embedding_proj = nn.Linear(num_features, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, seq_len)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embedding_dim, 32)
        self.relu_final = nn.ReLU()
        self.drop_final = nn.Dropout(0.1)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch_size, num_features, seq_len)
        x = self.feature_weight(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, num_features)

        x = self.embedding_proj(x)  # (batch_size, seq_len, embedding_dim)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)  # 位置编码

        x = self.transformer_encoder(x)  # (batch_size, seq_len, embedding_dim)

        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        x = self.global_avg_pool(x).squeeze(-1)  # (batch_size, embedding_dim)

        x = self.fc1(x)
        x = self.relu_final(x)
        x = self.drop_final(x)
        x = self.fc_out(x)
        return x


# 3. NSGA-II 超参数优化
def initialize_population(pop_size, lb, ub, int_con):
    num_vars = len(lb)
    population = np.random.rand(pop_size, num_vars) * (np.array(ub) - np.array(lb)) + np.array(lb)

    # 处理整数变量
    for i in range(pop_size):
        for j in int_con:
            population[i, j] = round(population[i, j])

    return population


def evaluate_model(params, train_dataset, val_dataset, min_speed, max_speed, num_features, sequence_length,
                   batch_size=32):
    # 解析超参数
    batch_size = round(params[0])
    learn_rate = params[1]
    pool_type = round(params[2])
    num_filters1 = round(params[3])
    filter_size1 = round(params[4])
    filter_size2 = filter_size1
    lstm_units1 = round(params[5])
    lstm_units2 = round(params[6])
    reg_type = round(params[7])
    dropout_prob1 = params[8]
    dropout_prob2 = params[9]
    optimizer_type = round(params[10])
    num_conv_layers = round(params[11])
    conv_dropout = params[12]
    activation_function = round(params[13])

    # 参数合法性检查
    if (batch_size < 8 or batch_size > 2048 or
            learn_rate < 1e-6 or learn_rate > 5e-2 or
            pool_type not in [1, 2] or
            num_filters1 < 8 or num_filters1 > 512 or
            filter_size1 < 2 or filter_size1 > 5 or
            lstm_units1 < 8 or lstm_units1 > 512 or
            lstm_units2 < 8 or lstm_units2 > 512 or
            reg_type not in [1, 2, 3] or
            dropout_prob1 < 0 or dropout_prob1 > 0.6 or
            dropout_prob2 < 0 or dropout_prob2 > 0.6 or
            optimizer_type not in [1, 2, 3] or
            num_conv_layers not in [1, 2] or
            conv_dropout < 0 or conv_dropout > 0.6 or
            activation_function not in [1, 2, 3, 4]):
        return 100.0, 1e7, 100.0, 100.0, 100.0, -1.0  # 无效解的惩罚值

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 创建模型
    num_filters2 = max(8, num_filters1 // 2) if num_conv_layers == 2 else 0
    model = CNNBiLSTM(
        num_features, num_filters1, filter_size1, num_filters2, filter_size2,
        lstm_units1, lstm_units2, pool_type, num_conv_layers,
        dropout_prob1, dropout_prob2, conv_dropout, activation_function
    ).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()

    if optimizer_type == 1:
        optimizer = optim.Adam(model.parameters(), lr=learn_rate,
                               weight_decay=1e-4 if reg_type in [1, 3] else 0)
    elif optimizer_type == 2:
        optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9,
                              weight_decay=1e-4 if reg_type in [1, 3] else 0)
    else:  # 3
        optimizer = optim.RMSprop(model.parameters(), lr=learn_rate,
                                  weight_decay=1e-4 if reg_type in [1, 3] else 0)

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    # 训练模型
    best_val_loss = float('inf')
    patience = 5
    counter = 0

    for epoch in range(30):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        scheduler.step()

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model = deepcopy(model.state_dict())
        else:
            counter += 1
            if counter >= patience:
                break

    # 使用最佳模型进行评估
    model.load_state_dict(best_model)
    model.eval()

    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    # 反归一化
    all_targets = np.array(all_targets) * (max_speed - min_speed) + min_speed
    all_outputs = np.array(all_outputs) * (max_speed - min_speed) + min_speed

    # 计算评估指标
    mae = mean_absolute_error(all_targets, all_outputs)
    rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))
    mape = np.mean(np.abs((all_targets - all_outputs) / all_targets)) * 100
    corr_matrix = np.corrcoef(all_targets.flatten(), all_outputs.flatten())
    r = corr_matrix[0, 1] if corr_matrix.shape[0] >= 2 else 0

    # 计算模型复杂度（参数数量）
    complexity = sum(p.numel() for p in model.parameters())

    return rmse, complexity, rmse, mae, mape, r


def evaluate_population(population, train_dataset, val_dataset, min_speed, max_speed, sequence_length, num_features):
    pop_size = len(population)
    performance = np.zeros(pop_size)
    complexity = np.zeros(pop_size)
    rmse = np.zeros(pop_size)
    mae = np.zeros(pop_size)
    mape = np.zeros(pop_size)
    r = np.zeros(pop_size)

    for i in range(pop_size):
        perf, comp, rms, ma, map_, corr = evaluate_model(
            population[i], train_dataset, val_dataset, min_speed, max_speed,
            num_features, sequence_length
        )
        performance[i] = perf
        complexity[i] = comp
        rmse[i] = rms
        mae[i] = ma
        mape[i] = map_
        r[i] = corr
        print(f'个体 {i + 1}/{pop_size} - 性能指标: RMSE=%.4f (m/s), 复杂度=%d' % (rms, comp))

    return performance, complexity, rmse, mae, mape, r


def fast_non_dominated_sort(performance, complexity):
    pop_size = len(performance)
    fronts = []
    rank = np.zeros(pop_size, dtype=int)
    domination_count = np.zeros(pop_size, dtype=int)
    dominated_solutions = [[] for _ in range(pop_size)]

    # 计算支配关系
    for i in range(pop_size):
        for j in range(pop_size):
            if i == j:
                continue

            # 检查i是否支配j
            if (performance[i] <= performance[j] and complexity[i] <= complexity[j]) and \
                    (performance[i] < performance[j] or complexity[i] < complexity[j]):
                dominated_solutions[i].append(j)
            # 检查j是否支配i
            elif (performance[j] <= performance[i] and complexity[j] <= complexity[i]) and \
                    (performance[j] < performance[i] or complexity[j] < complexity[i]):
                domination_count[i] += 1

    # 第一前沿：被支配计数为0的解
    current_front = np.where(domination_count == 0)[0]
    fronts.append(current_front)
    rank[current_front] = 0  # 从0开始编号

    # 迭代生成后续前沿
    front_idx = 1
    while len(fronts[-1]) > 0:
        next_front = []
        for i in fronts[-1]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
                    rank[j] = front_idx

        if next_front:
            fronts.append(np.array(next_front))
        else:
            break

        front_idx += 1

    return fronts, rank


def crowding_distance(performance, complexity, fronts):
    pop_size = len(performance)
    distance = np.zeros(pop_size)

    for front in fronts:
        if len(front) <= 1:
            continue

        # 提取当前前沿的目标值
        perf = performance[front]
        comp = complexity[front]

        # 初始化当前前沿的拥挤度
        dist = np.zeros(len(front))
        dist[[0, -1]] = np.inf  # 边界解的拥挤度设为无穷大

        # 对第一个目标排序
        sorted_idx = np.argsort(perf)
        perf_sorted = perf[sorted_idx]
        perf_range = perf_sorted[-1] - perf_sorted[0] if perf_sorted[-1] != perf_sorted[0] else 1e-6

        # 计算第一个目标的贡献
        for i in range(1, len(front) - 1):
            dist[sorted_idx[i]] += (perf_sorted[i + 1] - perf_sorted[i - 1]) / perf_range

        # 对第二个目标排序
        sorted_idx = np.argsort(comp)
        comp_sorted = comp[sorted_idx]
        comp_range = comp_sorted[-1] - comp_sorted[0] if comp_sorted[-1] != comp_sorted[0] else 1e-6

        # 计算第二个目标的贡献
        for i in range(1, len(front) - 1):
            dist[sorted_idx[i]] += (comp_sorted[i + 1] - comp_sorted[i - 1]) / comp_range

        distance[front] = dist

    return distance


def tournament_selection(population, rank, distance, pop_size):
    mating_pool = []
    for _ in range(pop_size):
        # 随机选择两个个体
        idx1 = np.random.randint(len(population))
        idx2 = np.random.randint(len(population))

        # 选择rank小的；rank相同则选择拥挤度大的
        if rank[idx1] < rank[idx2] or (rank[idx1] == rank[idx2] and distance[idx1] > distance[idx2]):
            mating_pool.append(population[idx1])
        else:
            mating_pool.append(population[idx2])

    return np.array(mating_pool)


def sbx_crossover(parent_pool, lb, ub, int_con, pc=0.8, eta_c=20):
    pop_size, num_vars = parent_pool.shape
    offspring = np.zeros_like(parent_pool)

    # 确保父代数量为偶数
    if pop_size % 2 != 0:
        parent_pool = parent_pool[:-1]
        pop_size -= 1

    for i in range(0, pop_size, 2):
        p1 = parent_pool[i]
        p2 = parent_pool[i + 1]

        for j in range(num_vars):
            x1, x2 = p1[j], p2[j]

            if np.random.rand() < pc:
                if x1 != x2:
                    # 确保y1 < y2
                    y1, y2 = (x1, x2) if x1 < x2 else (x2, x1)

                    # 计算交叉因子
                    rand_val = np.random.rand()
                    if rand_val <= 0.5:
                        beta = (2 * rand_val) ** (1 / (eta_c + 1))
                    else:
                        beta = (1 / (2 * (1 - rand_val))) ** (1 / (eta_c + 1))

                    # 生成子代
                    c1 = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + beta * (y2 - y1))

                    # 边界处理
                    c1 = max(lb[j], min(ub[j], c1))
                    c2 = max(lb[j], min(ub[j], c2))

                    # 随机分配给子代
                    if np.random.rand() < 0.5:
                        offspring[i, j] = c1
                        offspring[i + 1, j] = c2
                    else:
                        offspring[i, j] = c2
                        offspring[i + 1, j] = c1
                else:
                    # 若父代值相同，直接继承
                    offspring[i, j] = x1
                    offspring[i + 1, j] = x2
            else:
                # 不交叉，直接继承
                offspring[i, j] = x1
                offspring[i + 1, j] = x2

    # 处理整数变量
    for i in range(pop_size):
        for j in int_con:
            offspring[i, j] = round(offspring[i, j])

    return offspring


def polynomial_mutation(offspring, lb, ub, int_con, pm=0.07, eta_m=20):
    pop_size, num_vars = offspring.shape

    for i in range(pop_size):
        for j in range(num_vars):
            if np.random.rand() < pm:
                x = offspring[i, j]
                xl, xu = lb[j], ub[j]

                if x > xl and x < xu:
                    delta1 = (x - xl) / (xu - xl)
                    delta2 = (xu - x) / (xu - xl)

                    rand_val = np.random.rand()
                    if rand_val <= 0.5:
                        mut_pow = 1 / (eta_m + 1)
                        delta = (2 * rand_val + (1 - 2 * rand_val) * (delta1 ** (eta_m + 1))) ** mut_pow - 1
                    else:
                        mut_pow = 1 / (eta_m + 1)
                        delta = 1 - (2 * (1 - rand_val) + 2 * (rand_val - 0.5) * (delta2 ** (eta_m + 1))) ** mut_pow

                    x += delta * (xu - xl)

                # 边界处理
                x = max(xl, min(xu, x))

                # 整数变量处理
                if j in int_con:
                    x = round(x)

                offspring[i, j] = x

    return offspring


def environmental_selection(combined_pop, combined_perf, combined_complex,
                            combined_rank, combined_dist, pop_size):
    # 按rank升序排序
    sorted_idx = np.argsort(combined_rank)
    combined_pop = combined_pop[sorted_idx]
    combined_perf = combined_perf[sorted_idx]
    combined_complex = combined_complex[sorted_idx]
    combined_rank = combined_rank[sorted_idx]
    combined_dist = combined_dist[sorted_idx]

    # 依次加入前沿直到超过种群大小
    current_size = 0
    new_pop = []
    new_perf = []
    new_complex = []

    unique_ranks = np.unique(combined_rank)

    for rank in unique_ranks:
        if current_size >= pop_size:
            break

        # 当前rank的所有个体
        mask = combined_rank == rank
        current_front = combined_pop[mask]
        current_front_perf = combined_perf[mask]
        current_front_complex = combined_complex[mask]
        current_front_dist = combined_dist[mask]

        front_size = len(current_front)

        if current_size + front_size <= pop_size:
            # 全部加入
            new_pop.extend(current_front)
            new_perf.extend(current_front_perf)
            new_complex.extend(current_front_complex)
            current_size += front_size
        else:
            # 部分加入（按拥挤度降序）
            remaining = pop_size - current_size
            sorted_dist_idx = np.argsort(current_front_dist)[::-1]  # 降序排列
            selected = sorted_dist_idx[:remaining]

            new_pop.extend(current_front[selected])
            new_perf.extend(current_front_perf[selected])
            new_complex.extend(current_front_complex[selected])
            current_size = pop_size

    return np.array(new_pop), np.array(new_perf), np.array(new_complex)


def find_pareto_front(population, performance, complexity, fronts):
    if not fronts:
        return {'params': [], 'performance': [], 'complexity': [], 'num_solutions': 0}

    # 第一前沿即为Pareto前沿
    pareto_indices = fronts[0]

    return {
        'params': population[pareto_indices],
        'performance': performance[pareto_indices],
        'complexity': complexity[pareto_indices],
        'num_solutions': len(pareto_indices)
    }


# 运行NSGA-II优化
print('开始 NSGA-II 超参数优化...')

# 超参数搜索范围（下界、上界）
lb = [32, 1e-6, 1, 32, 1, 32, 16, 1, 0.1, 0.1, 1, 1, 0.1, 1]
ub = [256, 5e-2, 2, 512, 7, 512, 256, 3, 0.55, 0.55, 3, 2, 0.55, 4]
int_con = [0, 2, 3, 4, 5, 6, 7, 10, 11, 13]  # 整数变量索引（0-based）

# NSGA-II 算法参数
population_size = 10
max_generations = 3
pc = 0.8
eta_c = 20
pm = 0.07
eta_m = 20

num_vars = len(lb)
population = initialize_population(population_size, lb, ub, int_con)

# 记录优化过程
best_params_history = np.zeros((max_generations, num_vars))
best_performance_history = np.zeros(max_generations)
best_complexity_history = np.zeros(max_generations)
all_pareto_fronts = []

specific_gens = [1, 10, 30, 70, 100, 150]
specific_gen_data = {}

# 绘制Pareto前沿的图形
plt.figure(figsize=(10, 8))
plt.xlabel('模型复杂度（越小越好）')
plt.ylabel('预测误差 (RMSE, m/s)（越小越好）')
plt.title('每一代的 Pareto 前沿')
plt.grid(True)

for generation in range(max_generations):
    print(f'第 {generation + 1} 代进化中...')

    # 评估种群性能
    performance, complexity, rmse, mae, mape, r = evaluate_population(
        population, train_dataset, val_dataset, min_speed, max_speed,
        sequence_length, feature_data.shape[1]
    )

    # 记录当前代最优解（按RMSE）
    min_idx = np.argmin(performance)
    best_params_history[generation] = population[min_idx]
    best_performance_history[generation] = performance[min_idx]
    best_complexity_history[generation] = complexity[min_idx]

    # 快速非支配排序
    fronts, rank = fast_non_dominated_sort(performance, complexity)

    # 拥挤度计算
    distance = crowding_distance(performance, complexity, fronts)

    # 记录Pareto前沿
    pareto_struct = find_pareto_front(population, performance, complexity, fronts)
    all_pareto_fronts.append(pareto_struct)

    # 保存特定代数据
    if (generation + 1) in specific_gens:  # generation是0-based
        gen_data = {
            'population': population,
            'performance': performance,
            'complexity': complexity,
            'pareto_front': pareto_struct,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r': r
        }
        specific_gen_data[generation + 1] = gen_data

    # 绘制当前代Pareto前沿
    if pareto_struct['num_solutions'] > 0:
        perf_vals = pareto_struct['performance']
        complexity_vals = pareto_struct['complexity']

        # 为重叠点添加微小扰动以便可视化
        unique_points = np.unique(np.column_stack((complexity_vals, perf_vals)), axis=0)
        perturbed_front = []

        for point in unique_points:
            matching = np.all(np.column_stack((complexity_vals, perf_vals)) == point, axis=1)
            num_duplicates = np.sum(matching)

            if num_duplicates > 1:
                jitter = 1e-4 * np.random.randn(num_duplicates, 2)
                perturbed = np.tile(point, (num_duplicates, 1)) + jitter
                perturbed_front.extend(perturbed)
            else:
                perturbed_front.append(point)

        perturbed_front = np.array(perturbed_front)
        plt.scatter(perturbed_front[:, 0], perturbed_front[:, 1], 36, alpha=0.6)
        plt.pause(0.1)  # 刷新图形

    print(f'第 {generation + 1} 代 Pareto 解的数量: {pareto_struct["num_solutions"]}')

    # 选择算子
    mating_pool = tournament_selection(population, rank, distance, population_size)

    # 交叉
    offspring = sbx_crossover(mating_pool, lb, ub, int_con, pc, eta_c)

    # 变异
    offspring = polynomial_mutation(offspring, lb, ub, int_con, pm, eta_m)

    # 评估子代
    offspring_performance, offspring_complexity, _, _, _, _ = evaluate_population(
        offspring, train_dataset, val_dataset, min_speed, max_speed,
        sequence_length, feature_data.shape[1]
    )

    # 合并父代和子代
    combined_population = np.vstack((population, offspring))
    combined_performance = np.hstack((performance, offspring_performance))
    combined_complexity = np.hstack((complexity, offspring_complexity))

    # 环境选择
    combined_fronts, combined_rank = fast_non_dominated_sort(combined_performance, combined_complexity)
    combined_distance = crowding_distance(combined_performance, combined_complexity, combined_fronts)

    population, performance, complexity = environmental_selection(
        combined_population, combined_performance, combined_complexity,
        combined_rank, combined_distance, population_size
    )

plt.legend([f'第 {i + 1} 代' for i in range(max_generations)], loc='best')
plt.tight_layout()
plt.show()

print('NSGA-II 优化完成。')

# 绘制特定代的Pareto前沿对比
specific_gens = [1, 5, 10, 20, 40, 60]
plt.figure(figsize=(10, 8))
plt.title('特定代的 Pareto 前沿对比')
plt.xlabel('模型复杂度（越小越好）')
plt.ylabel('预测误差 (RMSE, m/s)（越小越好）')
plt.grid(True)

colors = plt.cm.tab10(np.linspace(0, 1, len(specific_gens)))

for idx, gen in enumerate(specific_gens):
    if gen <= max_generations and (gen - 1) < len(all_pareto_fronts):
        pareto_struct = all_pareto_fronts[gen - 1]  # 转换为0-based索引

        if pareto_struct['num_solutions'] > 0:
            perf_vals = pareto_struct['performance']
            complexity_vals = pareto_struct['complexity']

            unique_points = np.unique(np.column_stack((complexity_vals, perf_vals)), axis=0)
            perturbed_front = []

            for point in unique_points:
                matching = np.all(np.column_stack((complexity_vals, perf_vals)) == point, axis=1)
                num_duplicates = np.sum(matching)

                if num_duplicates > 1:
                    jitter = 1e-4 * np.random.randn(num_duplicates, 2)
                    perturbed = np.tile(point, (num_duplicates, 1)) + jitter
                    perturbed_front.extend(perturbed)
                else:
                    perturbed_front.append(point)

            perturbed_front = np.array(perturbed_front)
            plt.scatter(perturbed_front[:, 0], perturbed_front[:, 1], 36,
                        color=colors[idx], alpha=0.6, label=f'第 {gen} 代')

plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 保存优化结果
import pickle

final_pareto_struct = all_pareto_fronts[-1] if all_pareto_fronts else {'num_solutions': 0}

with open('nsga2_optimization_results_extended.pkl', 'wb') as f:
    pickle.dump({
        'finalParetoStruct': final_pareto_struct,
        'allParetoFronts': all_pareto_fronts,
        'bestParamsHistory': best_params_history,
        'bestPerformanceHistory': best_performance_history,
        'bestComplexityHistory': best_complexity_history,
        'specificGenData': specific_gen_data
    }, f)

print('优化结果已保存至 nsga2_optimization_results_extended.pkl')


# 4. 最终模型训练
# 查找所有代中最优的Pareto解
all_valid_pareto = []
all_valid_perf = []
all_valid_comp = []

for front in all_pareto_fronts:
    if front['num_solutions'] > 0:
        all_valid_pareto.extend(front['params'])
        all_valid_perf.extend(front['performance'])
        all_valid_comp.extend(front['complexity'])

# 如果最终代没有，使用所有代的有效解
if final_pareto_struct['num_solutions'] == 0 and all_valid_pareto:
    final_pareto_struct = {
        'params': np.array(all_valid_pareto),
        'performance': np.array(all_valid_perf),
        'complexity': np.array(all_valid_comp),
        'num_solutions': len(all_valid_pareto)
    }
    print('警告：最后一代无Pareto解，使用所有代的有效Pareto解')

if final_pareto_struct['num_solutions'] > 0:
    perf_vals = final_pareto_struct['performance']
    complexity_vals = final_pareto_struct['complexity']

    # 选择折中最优解
    normalized_perf = (perf_vals - np.min(perf_vals)) / (np.max(perf_vals) - np.min(perf_vals) + 1e-10)
    normalized_comp = (complexity_vals - np.min(complexity_vals)) / (
                np.max(complexity_vals) - np.min(complexity_vals) + 1e-10)
    trade_off_scores = np.sqrt(normalized_perf ** 2 + normalized_comp ** 2)

    trade_off_idx = np.argmin(trade_off_scores)
    best_params = final_pareto_struct['params'][trade_off_idx]
    print(f'选取Pareto前沿折中最优解（trade-off index: {trade_off_idx + 1}）。')
else:
    raise ValueError('最终Pareto前沿无有效解，无法选取模型。')

# 解析最优超参数
batch_size = round(best_params[0])
learn_rate = best_params[1]
pool_type = round(best_params[2])
num_filters1 = round(best_params[3])
filter_size1 = round(best_params[4])
filter_size2 = filter_size1
lstm_units1 = round(best_params[5])
lstm_units2 = round(best_params[6])
reg_type = round(best_params[7])
dropout_prob1 = best_params[8]
dropout_prob2 = best_params[9]
optimizer_type = round(best_params[10])
num_conv_layers = round(best_params[11])
conv_dropout = best_params[12]
activation_function = round(best_params[13])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 构建最终网络
num_filters2 = max(8, num_filters1 // 2) if num_conv_layers == 2 else 0
model_opt = CNNBiLSTM(
    feature_data.shape[1], num_filters1, filter_size1, num_filters2, filter_size2,
    lstm_units1, lstm_units2, pool_type, num_conv_layers,
    dropout_prob1, dropout_prob2, conv_dropout, activation_function
).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()

if optimizer_type == 1:
    optimizer = optim.Adam(model_opt.parameters(), lr=learn_rate,
                           weight_decay=1e-4 if reg_type in [1, 3] else 0)
elif optimizer_type == 2:
    optimizer = optim.SGD(model_opt.parameters(), lr=learn_rate, momentum=0.9,
                          weight_decay=1e-4 if reg_type in [1, 3] else 0)
else:  # 3
    optimizer = optim.RMSprop(model_opt.parameters(), lr=learn_rate,
                              weight_decay=1e-4 if reg_type in [1, 3] else 0)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.15)

# 训练模型
best_val_loss = float('inf')
patience = 10
counter = 0
train_losses = []
val_losses = []

for epoch in range(50):
    model_opt.train()
    train_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model_opt(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # 验证
    model_opt.eval()
    val_loss = 0.0
    all_val_targets = []
    all_val_outputs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_opt(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

            all_val_targets.extend(targets.cpu().numpy())
            all_val_outputs.extend(outputs.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    scheduler.step()

    print(f'Epoch {epoch + 1}/{50}, 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}')

    # 早停机制
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        best_model_opt = deepcopy(model_opt.state_dict())
    else:
        counter += 1
        if counter >= patience:
            print(f'早停于第 {epoch + 1} 轮')
            break

# 加载最佳模型
model_opt.load_state_dict(best_model_opt)
model_opt.eval()

# 在验证集上评估
all_val_targets = []
all_val_outputs = []

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model_opt(inputs)

        all_val_targets.extend(targets.cpu().numpy())
        all_val_outputs.extend(outputs.cpu().numpy())

# 反归一化
all_val_targets = np.array(all_val_targets) * (max_speed - min_speed) + min_speed
all_val_outputs = np.array(all_val_outputs) * (max_speed - min_speed) + min_speed

# 计算评估指标
mae_val_opt = mean_absolute_error(all_val_targets, all_val_outputs)
rmse_val_opt = np.sqrt(mean_squared_error(all_val_targets, all_val_outputs))
mape_val_opt = np.mean(np.abs((all_val_targets - all_val_outputs) / all_val_targets)) * 100
corr_matrix_val_opt = np.corrcoef(all_val_targets.flatten(), all_val_outputs.flatten())
r_val_opt = corr_matrix_val_opt[0, 1] if corr_matrix_val_opt.shape[0] >= 2 else 0

print(f'优化模型验证集性能: MAE = {mae_val_opt:.4f} (m/s), RMSE = {rmse_val_opt:.4f} (m/s), '
      f'MAPE = {mape_val_opt:.2f}%, R = {r_val_opt:.4f}')

# 在测试集上评估
all_test_targets = []
all_test_outputs = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model_opt(inputs)

        all_test_targets.extend(targets.cpu().numpy())
        all_test_outputs.extend(outputs.cpu().numpy())

# 反归一化
all_test_targets = np.array(all_test_targets) * (max_speed - min_speed) + min_speed
all_test_outputs = np.array(all_test_outputs) * (max_speed - min_speed) + min_speed

# 计算评估指标
mae_test_opt = mean_absolute_error(all_test_targets, all_test_outputs)
rmse_test_opt = np.sqrt(mean_squared_error(all_test_targets, all_test_outputs))
mape_test_opt = np.mean(np.abs((all_test_targets - all_test_outputs) / all_test_targets)) * 100
corr_matrix_test_opt = np.corrcoef(all_test_targets.flatten(), all_test_outputs.flatten())
r_test_opt = corr_matrix_test_opt[0, 1] if corr_matrix_test_opt.shape[0] >= 2 else 0

print(f'优化模型测试集性能: MAE = {mae_test_opt:.4f} (m/s), RMSE = {rmse_test_opt:.4f} (m/s), '
      f'MAPE = {mape_test_opt:.2f}%, R = {r_test_opt:.4f}')

# 学习到的特征权重
feature_weights = model_opt.feature_weight.weights.detach().cpu().numpy()
print('\n学习到的特征权重：')
for i, col in enumerate(feature_columns):
    print(f'  {col}: {feature_weights[i, 0]:.4f}')


# 5. 基准模型对比
print('开始基准模型对比...')

# BiLSTM模型
model_lstm = BiLSTM(feature_data.shape[1], lstm_units1, lstm_units2, dropout_prob1, dropout_prob2).to(device)

criterion_lstm = nn.MSELoss()
optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=learn_rate, weight_decay=1e-4 if reg_type in [1, 3] else 0)
scheduler_lstm = optim.lr_scheduler.StepLR(optimizer_lstm, step_size=20, gamma=0.2)

best_val_loss_lstm = float('inf')
counter_lstm = 0

for epoch in range(50):
    model_lstm.train()
    train_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer_lstm.zero_grad()
        outputs = model_lstm(inputs)
        loss = criterion_lstm(outputs, targets)
        loss.backward()
        optimizer_lstm.step()

        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)

    # 验证
    model_lstm.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_lstm(inputs)
            loss = criterion_lstm(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    scheduler_lstm.step()

    if val_loss < best_val_loss_lstm:
        best_val_loss_lstm = val_loss
        counter_lstm = 0
        best_model_lstm = deepcopy(model_lstm.state_dict())
    else:
        counter_lstm += 1
        if counter_lstm >= patience:
            break

# 在测试集上评估BiLSTM
model_lstm.load_state_dict(best_model_lstm)
model_lstm.eval()

all_test_outputs_lstm = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model_lstm(inputs)
        all_test_outputs_lstm.extend(outputs.cpu().numpy())

# 反归一化
all_test_outputs_lstm = np.array(all_test_outputs_lstm) * (max_speed - min_speed) + min_speed

# 计算评估指标
mae_test_lstm = mean_absolute_error(all_test_targets, all_test_outputs_lstm)
rmse_test_lstm = np.sqrt(mean_squared_error(all_test_targets, all_test_outputs_lstm))
mape_test_lstm = np.mean(np.abs((all_test_targets - all_test_outputs_lstm) / all_test_targets)) * 100
corr_matrix_test_lstm = np.corrcoef(all_test_targets.flatten(), all_test_outputs_lstm.flatten())
r_test_lstm = corr_matrix_test_lstm[0, 1] if corr_matrix_test_lstm.shape[0] >= 2 else 0

print(f'BiLSTM模型测试集性能: MAE = {mae_test_lstm:.4f}, RMSE = {rmse_test_lstm:.4f}, '
      f'MAPE = {mae_test_lstm:.2f}%, R = {r_test_lstm:.4f}')

# GRU模型
model_gru = GRU(feature_data.shape[1], lstm_units1, lstm_units2, dropout_prob1, dropout_prob2).to(device)

criterion_gru = nn.MSELoss()
optimizer_gru = optim.Adam(model_gru.parameters(), lr=learn_rate, weight_decay=1e-4 if reg_type in [1, 3] else 0)
scheduler_gru = optim.lr_scheduler.StepLR(optimizer_gru, step_size=20, gamma=0.2)

best_val_loss_gru = float('inf')
counter_gru = 0

for epoch in range(50):
    model_gru.train()
    train_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer_gru.zero_grad()
        outputs = model_gru(inputs)
        loss = criterion_gru(outputs, targets)
        loss.backward()
        optimizer_gru.step()

        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)

    # 验证
    model_gru.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_gru(inputs)
            loss = criterion_gru(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    scheduler_gru.step()

    if val_loss < best_val_loss_gru:
        best_val_loss_gru = val_loss
        counter_gru = 0
        best_model_gru = deepcopy(model_gru.state_dict())
    else:
        counter_gru += 1
        if counter_gru >= patience:
            break

# 在测试集上评估GRU
model_gru.load_state_dict(best_model_gru)
model_gru.eval()

all_test_outputs_gru = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model_gru(inputs)
        all_test_outputs_gru.extend(outputs.cpu().numpy())

# 反归一化
all_test_outputs_gru = np.array(all_test_outputs_gru) * (max_speed - min_speed) + min_speed

# 计算评估指标
mae_test_gru = mean_absolute_error(all_test_targets, all_test_outputs_gru)
rmse_test_gru = np.sqrt(mean_squared_error(all_test_targets, all_test_outputs_gru))
mape_test_gru = np.mean(np.abs((all_test_targets - all_test_outputs_gru) / all_test_targets)) * 100
corr_matrix_test_gru = np.corrcoef(all_test_targets.flatten(), all_test_outputs_gru.flatten())
r_test_gru = corr_matrix_test_gru[0, 1] if corr_matrix_test_gru.shape[0] >= 2 else 0

print(f'GRU模型测试集性能: MAE = {mae_test_gru:.4f}, RMSE = {rmse_test_gru:.4f}, '
      f'MAPE = {mape_test_gru:.2f}%, R = {r_test_gru:.4f}')

# CNN模型
num_filters_cnn1 = max(8, num_filters1 // 2)
num_filters_cnn2 = max(8, num_filters_cnn1 // 2)
model_cnn = CNN(feature_data.shape[1], num_filters_cnn1, filter_size1, num_filters_cnn2, filter_size1).to(device)

criterion_cnn = nn.MSELoss()
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=learn_rate, weight_decay=1e-4 if reg_type in [1, 3] else 0)
scheduler_cnn = optim.lr_scheduler.StepLR(optimizer_cnn, step_size=20, gamma=0.15)

best_val_loss_cnn = float('inf')
counter_cnn = 0

for epoch in range(50):
    model_cnn.train()
    train_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer_cnn.zero_grad()
        outputs = model_cnn(inputs)
        loss = criterion_cnn(outputs, targets)
        loss.backward()
        optimizer_cnn.step()

        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)

    # 验证
    model_cnn.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_cnn(inputs)
            loss = criterion_cnn(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    scheduler_cnn.step()

    if val_loss < best_val_loss_cnn:
        best_val_loss_cnn = val_loss
        counter_cnn = 0
        best_model_cnn = deepcopy(model_cnn.state_dict())
    else:
        counter_cnn += 1
        if counter_cnn >= patience:
            break

# 在测试集上评估CNN
model_cnn.load_state_dict(best_model_cnn)
model_cnn.eval()

all_test_outputs_cnn = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model_cnn(inputs)
        all_test_outputs_cnn.extend(outputs.cpu().numpy())

# 反归一化
all_test_outputs_cnn = np.array(all_test_outputs_cnn) * (max_speed - min_speed) + min_speed

# 计算评估指标
mae_test_cnn = mean_absolute_error(all_test_targets, all_test_outputs_cnn)
rmse_test_cnn = np.sqrt(mean_squared_error(all_test_targets, all_test_outputs_cnn))
mape_test_cnn = np.mean(np.abs((all_test_targets - all_test_outputs_cnn) / all_test_targets)) * 100
corr_matrix_test_cnn = np.corrcoef(all_test_targets.flatten(), all_test_outputs_cnn.flatten())
r_test_cnn = corr_matrix_test_cnn[0, 1] if corr_matrix_test_cnn.shape[0] >= 2 else 0

print(f'CNN模型测试集性能: MAE = {mae_test_cnn:.4f}, RMSE = {rmse_test_cnn:.4f}, '
      f'MAPE = {mape_test_cnn:.2f}%, R = {r_test_cnn:.4f}')

# Transformer模型
embedding_dim = 64
num_heads = 4
ffn_dim = 128
seq_len = sequence_length

model_transformer = TransformerModel(feature_data.shape[1], embedding_dim, num_heads, ffn_dim, seq_len).to(device)

criterion_transformer = nn.MSELoss()
optimizer_transformer = optim.Adam(model_transformer.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler_transformer = optim.lr_scheduler.StepLR(optimizer_transformer, step_size=15, gamma=0.5)

best_val_loss_transformer = float('inf')
counter_transformer = 0

print('开始训练标准Transformer模型...')
for epoch in range(50):
    model_transformer.train()
    train_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer_transformer.zero_grad()
        outputs = model_transformer(inputs)
        loss = criterion_transformer(outputs, targets)
        loss.backward()
        optimizer_transformer.step()

        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)

    # 验证
    model_transformer.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_transformer(inputs)
            loss = criterion_transformer(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    scheduler_transformer.step()

    if val_loss < best_val_loss_transformer:
        best_val_loss_transformer = val_loss
        counter_transformer = 0
        best_model_transformer = deepcopy(model_transformer.state_dict())
    else:
        counter_transformer += 1
        if counter_transformer >= patience:
            break

# 在测试集上评估Transformer
model_transformer.load_state_dict(best_model_transformer)
model_transformer.eval()

all_test_outputs_transformer = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model_transformer(inputs)
        all_test_outputs_transformer.extend(outputs.cpu().numpy())

# 反归一化
all_test_outputs_transformer = np.array(all_test_outputs_transformer) * (max_speed - min_speed) + min_speed

# 计算评估指标
mae_test_transformer = mean_absolute_error(all_test_targets, all_test_outputs_transformer)
rmse_test_transformer = np.sqrt(mean_squared_error(all_test_targets, all_test_outputs_transformer))
mape_test_transformer = np.mean(np.abs((all_test_targets - all_test_outputs_transformer) / all_test_targets)) * 100
corr_matrix_test_transformer = np.corrcoef(all_test_targets.flatten(), all_test_outputs_transformer.flatten())
r_test_transformer = corr_matrix_test_transformer[0, 1] if corr_matrix_test_transformer.shape[0] >= 2 else 0

print(f'标准Transformer模型测试集性能: MAE = {mae_test_transformer:.4f}, RMSE = {rmse_test_transformer:.4f}, '
      f'MAPE = {mape_test_transformer:.2f}%, R = {r_test_transformer:.4f}')


# 6. 可视化分析
plt.figure(figsize=(16, 12))
plt.suptitle('最终模型 - 预测分析', fontsize=16)

# 验证集预测效果
plt.subplot(2, 3, 1)
plt.plot(all_val_targets, 'b-', linewidth=1.5, label='验证集真实值')
plt.plot(all_val_outputs, 'r--', linewidth=1.5, label='优化模型预测')
plt.xlabel('时间步')
plt.ylabel('风速 (m/s)')
plt.title(f'验证集预测效果 (RMSE={rmse_val_opt:.3f})')
plt.legend(loc='best')
plt.grid(True)

# 测试集预测效果
plt.subplot(2, 3, 2)
plt.plot(all_test_targets, 'b-', linewidth=1.5, label='测试集真实值')
plt.plot(all_test_outputs, 'r--', linewidth=1.5, label='优化模型预测')
plt.xlabel('时间步')
plt.ylabel('风速 (m/s)')
plt.title(f'测试集预测效果 (RMSE={rmse_test_opt:.3f})')
plt.legend(loc='best')
plt.grid(True)

# 测试集误差分布
plt.subplot(2, 3, 3)
errors_test_opt = all_test_targets - all_test_outputs
plt.hist(errors_test_opt, 30, density=True, alpha=0.7)
plt.xlabel('预测误差 (m/s)')
plt.ylabel('概率密度')
plt.title(f'测试集误差分布 (均值={np.mean(errors_test_opt):.3f})')
plt.grid(True)

# 预测 vs 真实
plt.subplot(2, 3, 4)
plt.scatter(all_test_targets, all_test_outputs, 50, alpha=0.6)
plt.plot([all_test_targets.min(), all_test_targets.max()],
         [all_test_targets.min(), all_test_targets.max()], 'k--', linewidth=2)
plt.xlabel('真实值 (m/s)')
plt.ylabel('预测值 (m/s)')
plt.title(f'预测 vs 真实 (R={r_test_opt:.3f})')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()

# 不同模型测试集性能对比
plt.subplot(2, 3, 5)
metrics = ['MAE', 'RMSE', 'MAPE', 'R']
opt_metrics = [mae_test_opt, rmse_test_opt, mape_test_opt, r_test_opt]
lstm_metrics = [mae_test_lstm, rmse_test_lstm, mape_test_lstm, r_test_lstm]
gru_metrics = [mae_test_gru, rmse_test_gru, mape_test_gru, r_test_gru]
cnn_metrics = [mae_test_cnn, rmse_test_cnn, mape_test_cnn, r_test_cnn]
transformer_metrics = [mae_test_transformer, rmse_test_transformer, mape_test_transformer, r_test_transformer]

x = np.arange(len(metrics))
width = 0.15

plt.bar(x - 2 * width, opt_metrics, width, label='优化CNN-BiLSTM')
plt.bar(x - width, lstm_metrics, width, label='BiLSTM')
plt.bar(x, gru_metrics, width, label='GRU')
plt.bar(x + width, cnn_metrics, width, label='CNN')
plt.bar(x + 2 * width, transformer_metrics, width, label='轻量级Transformer')

plt.xticks(x, metrics)
plt.ylabel('指标值')
plt.title('不同模型测试集性能对比')
plt.legend(loc='best')
plt.grid(True, axis='y')

# 优化模型测试集绝对误差时间序列
plt.subplot(2, 3, 6)
plt.plot(np.abs(errors_test_opt), 'g-', linewidth=1)
plt.xlabel('时间步')
plt.ylabel('绝对误差 (m/s)')
plt.title('优化模型测试集绝对误差时间序列')
plt.grid(True)

mean_abs_error = np.mean(np.abs(errors_test_opt))
std_abs_error = np.std(np.abs(errors_test_opt))
plt.axhline(mean_abs_error, color='r', linestyle='--',
            label=f'均值={mean_abs_error:.3f}')
plt.axhline(mean_abs_error + std_abs_error, color='r', linestyle=':',
            label=f'+1σ={mean_abs_error + std_abs_error:.3f}')
plt.axhline(max(0, mean_abs_error - std_abs_error), color='r', linestyle=':',
            label=f'-1σ={max(0, mean_abs_error - std_abs_error):.3f}')
plt.legend(loc='best')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 7. 特征权重可视化
plt.figure(figsize=(10, 6))
weights = feature_weights.flatten()
plt.bar(range(len(feature_columns)), weights, color=[0.2, 0.5, 0.8])
plt.xticks(range(len(feature_columns)), feature_columns, rotation=15)
plt.xlabel('输入特征')
plt.ylabel('学习到的权重（越大越重要）')
plt.title('优化模型中各输入特征的权重')
plt.grid(True, axis='y')

for i, w in enumerate(weights):
    plt.text(i, w + 0.02, f'{w:.4f}', ha='center', fontsize=9)

plt.tight_layout()
plt.show()

# 8. 模型预测对比
plt.figure(figsize=(16, 10))
plt.suptitle('各模型预测 vs 真实值', fontsize=16)

# 优化模型
plt.subplot(2, 3, 1)
plt.plot(all_test_targets, 'b-', linewidth=1.5, label='真实值')
plt.plot(all_test_outputs, 'r--', linewidth=1.5, label='优化模型')
plt.title('优化模型预测 vs 真实')
plt.xlabel('时间步')
plt.ylabel('风速 (m/s)')
plt.legend(loc='best')
plt.grid(True)

# BiLSTM
plt.subplot(2, 3, 2)
plt.plot(all_test_targets, 'b-', linewidth=1.5, label='真实值')
plt.plot(all_test_outputs_lstm, 'g--', linewidth=1.5, label='BiLSTM')
plt.title('BiLSTM预测 vs 真实')
plt.xlabel('时间步')
plt.ylabel('风速 (m/s)')
plt.legend(loc='best')
plt.grid(True)

# GRU
plt.subplot(2, 3, 3)
plt.plot(all_test_targets, 'b-', linewidth=1.5, label='真实值')
plt.plot(all_test_outputs_gru, 'm--', linewidth=1.5, label='GRU')
plt.title('GRU预测 vs 真实')
plt.xlabel('时间步')
plt.ylabel('风速 (m/s)')
plt.legend(loc='best')
plt.grid(True)

# CNN
plt.subplot(2, 3, 4)
plt.plot(all_test_targets, 'b-', linewidth=1.5, label='真实值')
plt.plot(all_test_outputs_cnn, 'c--', linewidth=1.5, label='CNN')
plt.title('CNN预测 vs 真实')
plt.xlabel('时间步')
plt.ylabel('风速 (m/s)')
plt.legend(loc='best')
plt.grid(True)

# Transformer
plt.subplot(2, 3, 5)
plt.plot(all_test_targets, 'b-', linewidth=1.5, label='真实值')
plt.plot(all_test_outputs_transformer, 'y--', linewidth=1.5, label='轻量级Transformer')
plt.title('轻量级Transformer预测 vs 真实')
plt.xlabel('时间步')
plt.ylabel('风速 (m/s)')
plt.legend(loc='best')
plt.grid(True)

# 所有模型对比
plt.subplot(2, 3, 6)
plt.plot(all_test_targets, 'k-', linewidth=2, label='真实值')
plt.plot(all_test_outputs, 'r--', linewidth=1, label='优化模型')
plt.plot(all_test_outputs_lstm, 'g--', linewidth=1, label='BiLSTM')
plt.plot(all_test_outputs_gru, 'm--', linewidth=1, label='GRU')
plt.plot(all_test_outputs_cnn, 'c--', linewidth=1, label='CNN')
plt.plot(all_test_outputs_transformer, 'y--', linewidth=1, label='轻量级Transformer')
plt.title('所有模型预测对比')
plt.xlabel('时间步')
plt.ylabel('风速 (m/s)')
plt.legend(loc='best')
plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 9. 打印优化后的超参数
param_names = ['Batch Size', 'Learn Rate', 'Pool Type',
               'Num Filters1', 'Filter Size1', 'LSTM Units1',
               'LSTM Units2', 'Reg Type', 'Dropout Prob1', 'Dropout Prob2',
               'OptimizerType', 'NumConvLayers', 'ConvDropout', 'ActivationFunction']

print('\n=== 最终优化的超参数 ===')
for i, name in enumerate(param_names):
    if i in int_con:
        print(f'{name}: {round(best_params[i])}')
    else:
        print(f'{name}: {best_params[i]:.6f}')

# 10. 生成优化摘要报告
print('\n=== NSGA-II 优化摘要报告 ===')
print(f'总进化代数: {max_generations}')
print(f'种群大小: {population_size}')
print(f'最终Pareto解数量: {final_pareto_struct["num_solutions"]}')
print(f'初始最优RMSE: {best_performance_history[0]:.4f} (m/s)')
print(f'最终最优RMSE: {best_performance_history[-1]:.4f} (m/s)')
if best_performance_history[0] > 0:
    improvement = (best_performance_history[0] - best_performance_history[-1]) / best_performance_history[0] * 100
    print(f'性能改进: {improvement:.2f}%')

print('\n=== 最终模型测试集性能 ===')
print(f'优化模型: MAE={mae_test_opt:.4f}, RMSE={rmse_test_opt:.4f}, MAPE={mape_test_opt:.2f}%, R={r_test_opt:.4f}')

print('\n=== 基准模型测试集性能对比 ===')
print(
    f'BiLSTM模型: MAE={mae_test_lstm:.4f}, RMSE={rmse_test_lstm:.4f}, MAPE={mae_test_lstm:.2f}%, R={r_test_lstm:.4f}')
print(f'GRU模型: MAE={mae_test_gru:.4f}, RMSE={rmse_test_gru:.4f}, MAPE={mape_test_gru:.2f}%, R={r_test_gru:.4f}')
print(f'CNN模型: MAE={mae_test_cnn:.4f}, RMSE={rmse_test_cnn:.4f}, MAPE={mape_test_cnn:.2f}%, R={r_test_cnn:.4f}')
print(
    f'轻量级Transformer模型: MAE={mae_test_transformer:.4f}, RMSE={rmse_test_transformer:.4f}, MAPE={mape_test_transformer:.2f}%, R={r_test_transformer:.4f}')

print('\n可视化图表已生成。')
print('优化完成！所有结果已保存至 nsga2_optimization_results_extended.pkl')