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
import pickle

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 1. 数据读取与预处理 ====================

print('正在读取风速数据...')
filename = 'winddata.xlsx'
try:
    data = pd.read_excel(filename)
    feature_columns = ['Wind Direction', 'Theoretical_Power_Curve (KWh)',
                       'LV ActivePower (kW)', 'Wind Speed (m/s)']
    target_column = 'Wind Speed (m/s)'

    for col in feature_columns:
        if col not in data.columns:
            raise ValueError(f'未找到特征列: {col}')

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
# plt.show() # 暂时注释，避免在脚本运行时弹出太多图表

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
        x = self.data[idx:idx + self.sequence_length, :]
        y = self.data[idx + self.sequence_length, -1]
        return torch.FloatTensor(x).transpose(0, 1), torch.FloatTensor([y])


sequence_length = 5
dataset = WindDataset(feature_data_norm, sequence_length)

# 按6:1:1比例划分
train_ratio = 6 / 8
val_ratio = 1 / 8
test_ratio = 1 / 8
num_samples = len(dataset)
num_train = int(train_ratio * num_samples)
num_val = int(val_ratio * num_samples)
num_test = num_samples - num_train - num_val

train_dataset = Subset(dataset, range(num_train))
val_dataset = Subset(dataset, range(num_train, num_train + num_val))
test_dataset = Subset(dataset, range(num_train + num_val, num_samples))

# 反归一化参数
min_speed = min_vals[-1]
max_speed = max_vals[-1]
print(f'数据预处理完成。训练样本数：{num_train}，验证样本数：{num_val}，测试样本数：{num_test}')
print(f'划分方式：按时间顺序（6:1:1），无时间交叉')


# ==================== 2. 模型定义 ====================

class FeatureWeightedLayer(nn.Module):
    def __init__(self, num_features):
        super(FeatureWeightedLayer, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_features, 1))

    def forward(self, x):
        return x * self.weights


class CNNBiLSTM(nn.Module):
    def __init__(self, num_features, num_filters1, filter_size1, num_filters2,
                 filter_size2, lstm_units1, lstm_units2, pool_type, num_conv_layers,
                 dropout_prob1, dropout_prob2, conv_dropout, activation_function):
        super(CNNBiLSTM, self).__init__()
        self.feature_weight = FeatureWeightedLayer(num_features)
        self.conv1 = nn.Conv1d(num_features, num_filters1, filter_size1, padding='same')
        self.bn1 = nn.BatchNorm1d(num_filters1)

        if activation_function == 1:
            self.act1 = nn.ReLU()
        elif activation_function == 2:
            self.act1 = nn.LeakyReLU(0.01)
        elif activation_function == 3:
            self.act1 = nn.Tanh()
        else:
            self.act1 = nn.Sigmoid()

        self.drop_conv1 = nn.Dropout(conv_dropout) if conv_dropout > 0 else nn.Identity()

        if pool_type == 1:
            self.pool = nn.MaxPool1d(2, stride=1, padding=1)
        else:
            self.pool = nn.AvgPool1d(2, stride=1, padding=1)

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
            else:
                self.act2 = nn.Sigmoid()
            self.drop_conv2 = nn.Dropout(conv_dropout) if conv_dropout > 0 else nn.Identity()
            rnn_input_size = num_filters2
        else:
            rnn_input_size = num_filters1

        self.bilstm1 = nn.LSTM(rnn_input_size, lstm_units1, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(dropout_prob1)
        self.bilstm2 = nn.LSTM(lstm_units1 * 2, lstm_units2, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(dropout_prob2)
        self.fc_out = nn.Linear(lstm_units2 * 2, 1)

    def forward(self, x):
        x = self.feature_weight(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop_conv1(x)

        if self.num_conv_layers == 2:
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act2(x)
            x = self.drop_conv2(x)

        x = self.pool(x)
        x = x.transpose(1, 2)

        x, _ = self.bilstm1(x)
        x = self.drop1(x)
        x, _ = self.bilstm2(x)
        x = self.drop2(x)

        x = x[:, -1, :]
        x = self.fc_out(x)
        return x


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
        x = self.feature_weight(x)
        x = x.transpose(1, 2)
        x, _ = self.bilstm1(x)
        x = self.drop1(x)
        x, _ = self.bilstm2(x)
        x = self.drop2(x)
        x = x[:, -1, :]
        x = self.fc_out(x)
        return x


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


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, seq_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(seq_len, embedding_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim // 2).float() *
                             (-math.log(10000.0) * 2 / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    def __init__(self, num_features, embedding_dim, num_heads, ffn_dim, seq_len, dropout_prob):
        super(TransformerModel, self).__init__()
        self.feature_weight = FeatureWeightedLayer(num_features)
        self.embedding_proj = nn.Linear(num_features, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, seq_len)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout_prob,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embedding_dim, 32)
        self.relu_final = nn.ReLU()
        self.drop_final = nn.Dropout(dropout_prob)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.feature_weight(x)
        x = x.transpose(1, 2)
        x = self.embedding_proj(x)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.fc1(x)
        x = self.relu_final(x)
        x = self.drop_final(x)
        x = self.fc_out(x)
        return x


# ==================== 3. NSGA-II 核心改进算法 ====================

def is_power_of_two(n):
    """检查是否为2的幂次"""
    return n > 0 and (n & (n - 1)) == 0


def filter_invalid_hyperparam(individual, lb, ub, model_type, int_con):
    """超参数合法性检查"""
    if not np.all((individual >= lb) & (individual <= ub)):
        return False

    batch_size = round(individual[0])
    if not is_power_of_two(batch_size):
        return False

    if model_type == 'CNNBiLSTM':
        if individual[1] > 0.1:  # 学习率
            return False
        if any(individual[i] > 0.6 for i in [8, 9, 12]):  # dropout
            return False
    elif model_type in ['BiLSTM', 'GRU']:
        if individual[1] > 0.1:
            return False
        if any(individual[i] > 0.6 for i in [5, 6]):
            return False
    elif model_type == 'CNN':
        if individual[1] > 0.1:
            return False
        if individual[7] > 0.6:
            return False
    # ********************* Transformer 简化参数检查 *********************
    elif model_type == 'Transformer':
        # 0: Batch Size, 1: Learn Rate, 2: Dropout Prob, 3: Optimizer Type
        if individual[1] > 0.1:  # Learn Rate
            return False
        if individual[2] > 0.6:  # Dropout Prob
            return False
    # ***************************************************************
    return True


def get_sensitivity_weights(num_vars, model_type):
    """超参数敏感性权重定义"""
    weights = np.ones(num_vars) * 0.8

    if model_type == 'CNNBiLSTM':
        weights[1] = 1.0  # 学习率
        weights[0] = 0.9  # batch size
        weights[8] = 0.85  # dropout1
        weights[9] = 0.85  # dropout2
        weights[10] = 0.4  # optimizer_type
        weights[13] = 0.3  # activation_function
    elif model_type in ['BiLSTM', 'GRU']:
        weights[1] = 1.0
        weights[0] = 0.9
        weights[5] = 0.85
        weights[6] = 0.85
        weights[7] = 0.4
    elif model_type == 'CNN':
        weights[1] = 1.0
        weights[0] = 0.9
        weights[7] = 0.8
        weights[8] = 0.4
        weights[9] = 0.3
    # ********************* Transformer 权重 *********************
    elif model_type == 'Transformer':
        # New indices: [Batch Size (0), Learn Rate (1), Dropout Prob (2), Optimizer Type (3)]
        if num_vars == 4:
            weights = np.ones(num_vars) * 0.8
            weights[1] = 1.0  # Learn Rate (new index 1)
            weights[0] = 0.9  # Batch size (new index 0)
            weights[2] = 0.85  # Dropout Prob (new index 2)
            weights[3] = 0.4  # Optimizer Type (new index 3)
    # ************************************************************
    return weights


def get_param_type(param_idx, int_con, model_type):
    """判断参数类型：0=连续型, 1=离散整数型, 2=类别型"""
    if model_type == 'CNNBiLSTM':
        cat_params = [2, 7, 10, 11, 13]
    elif model_type in ['BiLSTM', 'GRU']:
        cat_params = [4, 7]
    elif model_type == 'CNN':
        cat_params = [6, 8, 9]
    # ********************* Transformer 类型 *********************
    elif model_type == 'Transformer':
        # New indices: [Batch Size (0), Learn Rate (1), Dropout Prob (2), Optimizer Type (3)]
        cat_params = [3]  # Optimizer Type
    # ************************************************************
    else:
        cat_params = []

    if param_idx in cat_params:
        return 2
    elif param_idx in int_con:
        return 1
    else:
        return 0


def dynamic_eta_c(generation, max_generations, param_idx, param_type, sens_weight):
    """动态SBX分布指数"""
    if param_type == 2:
        return None

    base_min, base_max = 5, 25
    stage_coef = 0.3 if generation < max_generations * 0.3 else 0.7
    sens_coef = 0.4 if sens_weight > 0.85 else (0.7 if sens_weight > 0.6 else 1.0)

    eta_c = base_min + (base_max - base_min) * stage_coef * sens_coef
    return max(3, min(30, eta_c))


def dynamic_eta_m(generation, max_generations, param_idx, param_type, sens_weight):
    """动态变异分布指数"""
    if param_type == 0:
        base_min, base_max = 5, 12
    elif param_type == 1:
        base_min, base_max = 15, 22
    else:
        return None

    stage_coef = 1.0 - 0.5 * (generation / max_generations)
    sens_coef = 1.5 - sens_weight

    eta_m = base_min + (base_max - base_min) * stage_coef * sens_coef
    return max(3, min(25, eta_m))


def dynamic_pc(param_idx, param_type, performance=None, threshold=0.1):
    """差异化交叉概率"""
    base_pc = 0.15 if param_type == 2 else 0.85

    if performance is not None and performance < threshold:
        base_pc *= 0.6

    return max(0.05, min(0.95, base_pc))


def sbx_crossover_hyperparam(parent_pool, lb, ub, int_con, generation, max_generations,
                             sens_weights, model_type, pc_override=None, performance=None):
    """改进SBX交叉算子"""
    pop_size, num_vars = parent_pool.shape
    offspring = np.zeros_like(parent_pool)

    if pop_size % 2 != 0:
        parent_pool = parent_pool[:-1]
        pop_size -= 1

    for i in range(0, pop_size, 2):
        p1 = parent_pool[i]
        p2 = parent_pool[i + 1]

        for j in range(num_vars):
            x1, x2 = p1[j], p2[j]
            param_type = get_param_type(j, int_con, model_type)
            pc = pc_override if pc_override is not None else dynamic_pc(j, param_type, performance)

            if np.random.rand() < pc:
                if param_type == 2:
                    offspring[i, j] = x1
                    offspring[i + 1, j] = x1
                    if np.random.rand() < 0.05:
                        cat_range = np.arange(lb[j], ub[j] + 1)
                        offspring[i, j] = np.random.choice(cat_range)
                        offspring[i + 1, j] = np.random.choice(cat_range)

                elif param_type == 1:
                    eta_c = dynamic_eta_c(generation, max_generations, j, param_type, sens_weights[j])
                    if eta_c is None:
                        offspring[i, j] = x1
                        offspring[i + 1, j] = x2
                        continue

                    if x1 != x2:
                        y1, y2 = (x1, x2) if x1 < x2 else (x2, x1)
                        rand_val = np.random.rand()
                        beta = (2 * rand_val) ** (1 / (eta_c + 1)) if rand_val <= 0.5 else \
                            (1 / (2 * (1 - rand_val))) ** (1 / (eta_c + 1))

                        c1 = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                        c2 = 0.5 * ((y1 + y2) + beta * (y2 - y1))

                        c1 = max(lb[j], min(ub[j], c1))
                        c2 = max(lb[j], min(ub[j], c2))
                        offspring[i, j] = round(c1)
                        offspring[i + 1, j] = round(c2)
                    else:
                        offspring[i, j] = x1
                        offspring[i + 1, j] = x2

                else:
                    eta_c = dynamic_eta_c(generation, max_generations, j, param_type, sens_weights[j])
                    if x1 != x2:
                        y1, y2 = (x1, x2) if x1 < x2 else (x2, x1)
                        rand_val = np.random.rand()
                        beta = (2 * rand_val) ** (1 / (eta_c + 1)) if rand_val <= 0.5 else \
                            (1 / (2 * (1 - rand_val))) ** (1 / (eta_c + 1))

                        c1 = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                        c2 = 0.5 * ((y1 + y2) + beta * (y2 - y1))

                        offspring[i, j] = max(lb[j], min(ub[j], c1))
                        offspring[i + 1, j] = max(lb[j], min(ub[j], c2))
                    else:
                        offspring[i, j] = x1
                        offspring[i + 1, j] = x2
            else:
                offspring[i, j] = x1
                offspring[i + 1, j] = x2

    return offspring


def polynomial_mutation_hyperparam(offspring, lb, ub, int_con, generation, max_generations,
                                   sens_weights, model_type, pm=0.07):
    """改进多项式变异算子"""
    pop_size, num_vars = offspring.shape
    mutated_offspring = offspring.copy()

    for i in range(pop_size):
        valid = False
        attempts = 0

        while not valid and attempts < 5:
            individual = mutated_offspring[i].copy()

            for j in range(num_vars):
                # 修复 UnboundLocalError: local variable 'param_type' referenced before assignment
                param_type = get_param_type(j, int_con, model_type)

                if np.random.rand() < pm:
                    if param_type == 2:
                        cat_range = np.arange(lb[j], ub[j] + 1)
                        individual[j] = np.random.choice(cat_range)

                    elif param_type == 1:
                        eta_m = dynamic_eta_m(generation, max_generations, j, param_type, sens_weights[j])
                        # 仅对 Batch Size 增加较大的跳变步长
                        step = 8 if j == 0 and (
                                    model_type in ['CNNBiLSTM', 'BiLSTM', 'GRU', 'CNN', 'Transformer']) else 1
                        sign = np.random.choice([-1, 0, 1])
                        individual[j] += sign * step
                        individual[j] = round(individual[j])

                    else:
                        eta_m = dynamic_eta_m(generation, max_generations, j, param_type, sens_weights[j])
                        x = individual[j]
                        xl, xu = lb[j], ub[j]

                        if xl < x < xu:
                            max_delta = 0.3 * x
                            actual_xu = min(xu, x + max_delta)
                            actual_xl = max(xl, x - max_delta)

                            if actual_xu > actual_xl:
                                delta1 = (x - actual_xl) / (actual_xu - actual_xl)
                                delta2 = (actual_xu - x) / (actual_xu - actual_xl)

                                rand_val = np.random.rand()
                                mut_pow = 1 / (eta_m + 1)

                                if rand_val <= 0.5:
                                    delta = (2 * rand_val + (1 - 2 * rand_val) * (delta1 ** (eta_m + 1))) ** mut_pow - 1
                                else:
                                    delta = 1 - (2 * (1 - rand_val) + 2 * (rand_val - 0.5) * (
                                            delta2 ** (eta_m + 1))) ** mut_pow

                                individual[j] = x + delta * (actual_xu - actual_xl)

                individual[j] = max(lb[j], min(ub[j], individual[j]))

                if param_type == 1:
                    individual[j] = round(individual[j])

            if filter_invalid_hyperparam(individual, lb, ub, model_type, int_con):
                valid = True
                mutated_offspring[i] = individual
            else:
                attempts += 1

        if not valid:
            mutated_offspring[i] = offspring[i]

    return mutated_offspring


def weighted_crowding_distance(performance, complexity, fronts, population, sens_weights):
    """超参数优先级加权的拥挤度计算"""
    pop_size = len(performance)
    distance = np.zeros(pop_size)

    for front in fronts:
        if len(front) <= 2:
            distance[front] = np.inf if len(front) == 2 else 0
            continue

        current_pop = population[front]
        perf = performance[front]
        comp = complexity[front]
        num_vars = current_pop.shape[1]
        var_distances = np.zeros((len(front), num_vars))

        for j in range(num_vars):
            sorted_idx = np.argsort(current_pop[:, j])
            sorted_vals = current_pop[sorted_idx, j]
            var_range = sorted_vals[-1] - sorted_vals[0]
            if var_range < 1e-6:
                var_range = 1e-6

            for i in range(1, len(front) - 1):
                var_distances[sorted_idx[i], j] = (sorted_vals[i + 1] - sorted_vals[i - 1]) / var_range

        dist = np.sum(var_distances * sens_weights, axis=1)
        # 确保边界点具有无限距离 (NSGA-II标准)
        # 边界点距离应设为无穷大，避免其被选中
        dist[np.argmin(perf)] = np.inf
        dist[np.argmax(perf)] = np.inf
        dist[np.argmin(comp)] = np.inf
        dist[np.argmax(comp)] = np.inf

        distance[front] = dist

    return distance


def remove_duplicate_solutions(population, performance, complexity, threshold=1e-3):
    """精英存档去重"""
    unique_indices = []
    seen = []

    for i, ind in enumerate(population):
        is_duplicate = False

        # 遍历已存储的个体，检查是否相似
        for idx, prev_ind in enumerate(seen):
            diff = np.abs(ind - prev_ind)
            is_similar = np.all(diff < threshold)

            if is_similar:
                is_duplicate = True
                duplicate_idx = unique_indices[idx]  # 直接使用对应位置的索引
                # 保留性能更好的个体
                if performance[i] < performance[duplicate_idx]:
                    unique_indices[idx] = i  # 替换为更好的个体索引
                    seen[idx] = ind  # 更新对应位置的参数
                break

        if not is_duplicate:
            unique_indices.append(i)
            seen.append(ind)

    return unique_indices


def initialize_population(pop_size, lb, ub, int_con):
    num_vars = len(lb)
    population = np.random.rand(pop_size, num_vars) * (np.array(ub) - np.array(lb)) + np.array(lb)
    for i in range(pop_size):
        for j in int_con:
            population[i, j] = round(population[i, j])
    return population


def evaluate_model(params, train_dataset, val_dataset, min_speed, max_speed, num_features,
                   sequence_length, model_type, batch_size_default=32):
    if model_type == 'CNNBiLSTM':
        batch_size = round(params[0])
        learn_rate = params[1]
        pool_type = round(params[2])
        num_filters1 = round(params[3])
        filter_size1 = round(params[4])
        lstm_units1 = round(params[5])
        lstm_units2 = round(params[6])
        reg_type = round(params[7])
        dropout_prob1 = params[8]
        dropout_prob2 = params[9]
        optimizer_type = round(params[10])
        num_conv_layers = round(params[11])
        conv_dropout = params[12]
        activation_function = round(params[13])

        num_filters2 = max(8, num_filters1 // 2) if num_conv_layers == 2 else 0
        filter_size2 = filter_size1

        model = CNNBiLSTM(
            num_features, num_filters1, filter_size1, num_filters2, filter_size2,
            lstm_units1, lstm_units2, pool_type, num_conv_layers,
            dropout_prob1, dropout_prob2, conv_dropout, activation_function
        ).to(device)

    elif model_type == 'BiLSTM':
        batch_size = round(params[0])
        learn_rate = params[1]
        units1 = round(params[2])
        units2 = round(params[3])
        reg_type = round(params[4])
        dropout_prob1 = params[5]
        dropout_prob2 = params[6]
        optimizer_type = round(params[7])

        model = BiLSTM(num_features, units1, units2, dropout_prob1, dropout_prob2).to(device)

    elif model_type == 'GRU':
        batch_size = round(params[0])
        learn_rate = params[1]
        units1 = round(params[2])
        units2 = round(params[3])
        reg_type = round(params[4])
        dropout_prob1 = params[5]
        dropout_prob2 = params[6]
        optimizer_type = round(params[7])

        model = GRU(num_features, units1, units2, dropout_prob1, dropout_prob2).to(device)

    elif model_type == 'CNN':
        batch_size = round(params[0])
        learn_rate = params[1]
        num_filters1 = round(params[2])
        filter_size1 = round(params[3])
        num_filters2 = round(params[4])
        filter_size2 = round(params[5])
        reg_type = round(params[6])
        conv_dropout = params[7]
        optimizer_type = round(params[8])
        activation_function = round(params[9])

        model = CNN(num_features, num_filters1, filter_size1, num_filters2, filter_size2).to(device)

    # ********************* Transformer 模型实例化 (4个参数) *********************
    elif model_type == 'Transformer':
        # 优化参数 (4个):
        batch_size = round(params[0])
        learn_rate = params[1]
        dropout_prob = params[2]
        optimizer_type = round(params[3])

        # 固定的结构参数:
        embedding_dim = 64
        num_heads = 4
        ffn_dim = 128
        reg_type = 1  # 固定为 L2 正则化

        model = TransformerModel(num_features, embedding_dim, num_heads, ffn_dim,
                                 sequence_length, dropout_prob).to(device)
    # **************************************************************************
    else:
        raise ValueError(f'Unknown model_type: {model_type}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = nn.MSELoss()
    # reg_type 来自于固定的参数或优化的参数（Transformer 现在是固定的 1）
    weight_decay_val = 1e-4 if reg_type in [1, 3] else 0

    if optimizer_type == 1:
        optimizer = optim.Adam(model.parameters(), lr=learn_rate,
                               weight_decay=weight_decay_val)
    elif optimizer_type == 2:
        optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9,
                              weight_decay=weight_decay_val)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=learn_rate,
                                  weight_decay=weight_decay_val)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    best_val_loss = float('inf')
    counter = 0
    best_model = model.state_dict()  # 避免在没有训练时出现未定义变量

    for epoch in range(20):  # 修改为20
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model = deepcopy(model.state_dict())
        else:
            counter += 1
            if counter >= 5:
                break

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

    all_targets = np.array(all_targets) * (max_speed - min_speed) + min_speed
    all_outputs = np.array(all_outputs) * (max_speed - min_speed) + min_speed

    mae = mean_absolute_error(all_targets, all_outputs)
    rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))
    mape = np.mean(np.abs((all_targets - all_outputs) / all_targets)) * 100
    corr_matrix = np.corrcoef(all_targets.flatten(), all_outputs.flatten())
    r = corr_matrix[0, 1] if corr_matrix.shape[0] >= 2 else 0

    complexity = sum(p.numel() for p in model.parameters())
    return rmse, complexity, rmse, mae, mape, r


def evaluate_population(population, train_dataset, val_dataset, min_speed, max_speed,
                        sequence_length, num_features, model_type):
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
            num_features, sequence_length, model_type
        )
        performance[i] = perf
        complexity[i] = comp
        rmse[i] = rms
        mae[i] = ma
        mape[i] = map_
        r[i] = corr
        # print(f'个体 {i + 1}/{pop_size} - RMSE={rms:.4f}, 复杂度={comp}') # 减少打印，提高运行效率

    return performance, complexity, rmse, mae, mape, r


def fast_non_dominated_sort(performance, complexity):
    pop_size = len(performance)
    fronts = []
    rank = np.zeros(pop_size, dtype=int)
    domination_count = np.zeros(pop_size, dtype=int)
    dominated_solutions = [[] for _ in range(pop_size)]

    for i in range(pop_size):
        for j in range(pop_size):
            if i == j:
                continue
            if (performance[i] <= performance[j] and complexity[i] <= complexity[j]) and \
                    (performance[i] < performance[j] or complexity[i] < complexity[j]):
                dominated_solutions[i].append(j)
            elif (performance[j] <= performance[i] and complexity[j] <= complexity[i]) and \
                    (performance[j] < performance[i] or complexity[j] < complexity[i]):
                domination_count[i] += 1

    current_front = np.where(domination_count == 0)[0]
    fronts.append(current_front)
    rank[current_front] = 0

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


def tournament_selection(population, rank, distance, pop_size):
    mating_pool = []
    for _ in range(pop_size):
        idx1 = np.random.randint(len(population))
        idx2 = np.random.randint(len(population))
        if rank[idx1] < rank[idx2] or (rank[idx1] == rank[idx2] and distance[idx1] > distance[idx2]):
            mating_pool.append(population[idx1])
        else:
            mating_pool.append(population[idx2])
    return np.array(mating_pool)


def environmental_selection(combined_pop, combined_perf, combined_complex, combined_rank, combined_dist, pop_size):
    sorted_idx = np.argsort(combined_rank)
    combined_pop = combined_pop[sorted_idx]
    combined_perf = combined_perf[sorted_idx]
    combined_complex = combined_complex[sorted_idx]
    combined_rank = combined_rank[sorted_idx]
    combined_dist = combined_dist[sorted_idx]

    current_size = 0
    new_pop = []
    new_perf = []
    new_complex = []
    unique_ranks = np.unique(combined_rank)

    for rank in unique_ranks:
        if current_size >= pop_size:
            break

        mask = combined_rank == rank
        current_front = combined_pop[mask]
        current_front_perf = combined_perf[mask]
        current_front_complex = combined_complex[mask]  # 修复：将 current_complex 改为 combined_complex
        current_front_dist = combined_dist[mask]
        front_size = len(current_front)

        if current_size + front_size <= pop_size:
            new_pop.extend(current_front)
            new_perf.extend(current_front_perf)
            new_complex.extend(current_front_complex)
            current_size += front_size
        else:
            remaining = pop_size - current_size
            sorted_dist_idx = np.argsort(current_front_dist)[::-1]
            selected = sorted_dist_idx[:remaining]
            new_pop.extend(current_front[selected])
            new_perf.extend(current_front_perf[selected])
            new_complex.extend(current_front_complex[selected])
            current_size = pop_size

    return np.array(new_pop), np.array(new_perf), np.array(new_complex)


def find_pareto_front(population, performance, complexity, fronts):
    if not fronts:
        return {'params': [], 'performance': [], 'complexity': [], 'num_solutions': 0}

    pareto_indices = fronts[0]
    return {
        'params': population[pareto_indices],
        'performance': performance[pareto_indices],
        'complexity': complexity[pareto_indices],
        'num_solutions': len(pareto_indices)
    }


def run_nsga2_optimization(model_type, lb, ub, int_con, population_size, max_generations,
                           pc_base, eta_c_base, pm_base, eta_m_base, train_dataset,
                           val_dataset, min_speed, max_speed, sequence_length, num_features):
    print(f'开始 {model_type} 的 NSGA-II 超参数优化...')
    num_vars = len(lb)

    population = np.random.rand(population_size, num_vars) * (np.array(ub) - np.array(lb)) + np.array(lb)
    for i in range(population_size):
        for j in int_con:
            population[i, j] = round(population[i, j])

    sens_weights = get_sensitivity_weights(num_vars, model_type)

    best_params_history = np.zeros((max_generations, num_vars))
    best_performance_history = np.zeros(max_generations)
    best_complexity_history = np.zeros(max_generations)
    all_pareto_fronts = []

    # ==================== 新增：定义需要保存的特定代数 ====================
    save_generations = [1, 5, 10, 20, 40, 60]  # 要保存的特定代数
    saved_pareto_data = {gen: None for gen in save_generations}  # 存储这些代数的Pareto前沿
    # =========================================================================

    plt.figure(figsize=(10, 8))
    plt.xlabel('模型复杂度（越小越好）')
    plt.ylabel('预测误差 (RMSE, m/s)（越小越好）')
    plt.title(f'{model_type} Pareto前沿演化')
    plt.grid(True)
    plt.show(block=False)  # 非阻塞显示，允许后续绘图

    for generation in range(max_generations):
        print(f'第 {generation + 1}/{max_generations} 代进化中...')

        performance, complexity, rmse, mae, mape, r = evaluate_population(
            population, train_dataset, val_dataset, min_speed, max_speed,
            sequence_length, num_features, model_type
        )

        min_idx = np.argmin(performance)
        best_params_history[generation] = population[min_idx]
        best_performance_history[generation] = performance[min_idx]
        best_complexity_history[generation] = complexity[min_idx]

        fronts, rank = fast_non_dominated_sort(performance, complexity)
        distance = weighted_crowding_distance(performance, complexity, fronts, population, sens_weights)

        pareto_dict = find_pareto_front(population, performance, complexity, fronts)
        all_pareto_fronts.append(pareto_dict)

        # ==================== 新增：保存特定代数的Pareto前沿 ====================
        if generation + 1 in save_generations:
            saved_pareto_data[generation + 1] = {
                'complexity': pareto_dict['complexity'].copy(),
                'performance': pareto_dict['performance'].copy(),
                'num_solutions': pareto_dict['num_solutions']
            }
        # =========================================================================

        if pareto_dict['num_solutions'] > 0:
            plt.scatter(pareto_dict['complexity'], pareto_dict['performance'],
                        36, alpha=0.6, label=f'Gen {generation + 1}')
            # plt.pause(0.1) # 暂停可能导致运行时间过长，暂时注释掉

        print(f'  Pareto解数量: {pareto_dict["num_solutions"]}')

        mating_pool = tournament_selection(population, rank, distance, population_size)

        offspring = sbx_crossover_hyperparam(
            mating_pool, lb, ub, int_con, generation, max_generations,
            sens_weights, model_type, pc_override=pc_base
        )

        offspring = polynomial_mutation_hyperparam(
            offspring, lb, ub, int_con, generation, max_generations,
            sens_weights, model_type, pm=pm_base
        )

        offspring_performance, offspring_complexity, _, _, _, _ = evaluate_population(
            offspring, train_dataset, val_dataset, min_speed, max_speed,
            sequence_length, num_features, model_type
        )

        combined_pop = np.vstack((population, offspring))
        combined_perf = np.hstack((performance, offspring_performance))
        combined_comp = np.hstack((complexity, offspring_complexity))

        combined_fronts, combined_rank = fast_non_dominated_sort(combined_perf, combined_comp)
        combined_dist = weighted_crowding_distance(combined_perf, combined_comp, combined_fronts, combined_pop,
                                                   sens_weights)

        population, performance, complexity = environmental_selection(
            combined_pop, combined_perf, combined_comp, combined_rank, combined_dist, population_size
        )

        unique_idx = remove_duplicate_solutions(population, performance, complexity)
        population = population[unique_idx]
        performance = performance[unique_idx]
        complexity = complexity[unique_idx]

        if len(population) < population_size:
            num_missing = population_size - len(population)
            new_individuals = np.random.rand(num_missing, num_vars) * (np.array(ub) - np.array(lb)) + np.array(lb)
            for i in range(num_missing):
                for j in int_con:
                    new_individuals[i, j] = round(new_individuals[i, j])

            new_perf, new_comp, _, _, _, _ = evaluate_population(
                new_individuals, train_dataset, val_dataset, min_speed, max_speed,
                sequence_length, num_features, model_type
            )

            population = np.vstack((population, new_individuals))
            performance = np.hstack((performance, new_perf))
            complexity = np.hstack((complexity, new_comp))

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show(block=True)

    print(f'{model_type} NSGA-II 优化完成。')

    # ==================== 修改：增加返回保存的Pareto数据 ====================
    return all_pareto_fronts, best_params_history, best_performance_history, best_complexity_history, saved_pareto_data
    # =========================================================================


# ==================== 5. 参数配置与运行 ====================

population_size = 40
max_generations = 60
num_features = feature_data.shape[1]

# CNNBiLSTM搜索空间
lb_cnn_bilstm = [32, 1e-6, 1, 32, 1, 32, 16, 1, 0.1, 0.1, 1, 1, 0.1, 1]
ub_cnn_bilstm = [256, 5e-2, 2, 512, 7, 512, 256, 3, 0.55, 0.55, 3, 2, 0.55, 4]
int_con_cnn_bilstm = [0, 2, 3, 4, 5, 6, 7, 10, 11, 13]

# ==================== 修改：增加接收保存的Pareto数据 ====================
all_pareto_fronts_cnn_bilstm, best_params_history_cnn_bilstm, best_performance_history_cnn_bilstm, best_complexity_history_cnn_bilstm, saved_pareto_cnn_bilstm = run_nsga2_optimization(
    'CNNBiLSTM', lb_cnn_bilstm, ub_cnn_bilstm, int_con_cnn_bilstm,
    population_size, max_generations, 0.85, 20, 0.1, 20,
    train_dataset, val_dataset, min_speed, max_speed, sequence_length, num_features
)
# =========================================================================

# BiLSTM搜索空间
lb_bilstm = [32, 1e-6, 32, 16, 1, 0.1, 0.1, 1]
ub_bilstm = [256, 5e-2, 512, 256, 3, 0.55, 0.55, 3]
int_con_bilstm = [0, 2, 3, 4, 7]

# ==================== 修改：增加接收保存的Pareto数据 ====================
all_pareto_fronts_bilstm, best_params_history_bilstm, best_performance_history_bilstm, best_complexity_history_bilstm, saved_pareto_bilstm = run_nsga2_optimization(
    'BiLSTM', lb_bilstm, ub_bilstm, int_con_bilstm,
    population_size, max_generations, 0.85, 20, 0.1, 20,
    train_dataset, val_dataset, min_speed, max_speed, sequence_length, num_features
)
# =========================================================================

# GRU搜索空间
lb_gru = [32, 1e-6, 32, 16, 1, 0.1, 0.1, 1]
ub_gru = [256, 5e-2, 512, 256, 3, 0.55, 0.55, 3]
int_con_gru = [0, 2, 3, 4, 7]

# ==================== 修改：增加接收保存的Pareto数据 ====================
all_pareto_fronts_gru, best_params_history_gru, best_performance_history_gru, best_complexity_history_gru, saved_pareto_gru = run_nsga2_optimization(
    'GRU', lb_gru, ub_gru, int_con_gru,
    population_size, max_generations, 0.85, 20, 0.1, 20,
    train_dataset, val_dataset, min_speed, max_speed, sequence_length, num_features
)
# =========================================================================

# CNN搜索空间
lb_cnn = [32, 1e-6, 32, 1, 16, 1, 1, 0.1, 1, 1]
ub_cnn = [256, 5e-2, 512, 7, 256, 7, 3, 0.55, 3, 4]
int_con_cnn = [0, 2, 3, 4, 5, 6, 8, 9]

# ==================== 修改：增加接收保存的Pareto数据 ====================
all_pareto_fronts_cnn, best_params_history_cnn, best_performance_history_cnn, best_complexity_history_cnn, saved_pareto_cnn = run_nsga2_optimization(
    'CNN', lb_cnn, ub_cnn, int_con_cnn,
    population_size, max_generations, 0.85, 20, 0.1, 20,
    train_dataset, val_dataset, min_speed, max_speed, sequence_length, num_features
)
# =========================================================================

# Transformer搜索空间 (4个参数)
# 0: Batch Size, 1: Learn Rate, 2: Dropout Prob, 3: Optimizer Type
lb_transformer = [32, 1e-6, 0.1, 1]
ub_transformer = [256, 5e-2, 0.55, 3]
int_con_transformer = [0, 3]  # Index 0: Batch Size (Int), Index 3: Optimizer Type (Cat.)

# ==================== 修改：增加接收保存的Pareto数据 ====================
all_pareto_fronts_transformer, best_params_history_transformer, best_performance_history_transformer, best_complexity_history_transformer, saved_pareto_transformer = run_nsga2_optimization(
    'Transformer', lb_transformer, ub_transformer, int_con_transformer,
    population_size, max_generations, 0.85, 20, 0.1, 20,
    train_dataset, val_dataset, min_speed, max_speed, sequence_length, num_features
)
# =========================================================================


# ==================== 6. 结果保存与最优选择 ====================

with open('nsga2_optimization_results_extended.pkl', 'wb') as f:
    pickle.dump({
        'CNNBiLSTM': {
            'allParetoFronts': all_pareto_fronts_cnn_bilstm,
            'bestParamsHistory': best_params_history_cnn_bilstm,
            'bestPerformanceHistory': best_performance_history_cnn_bilstm,
            'bestComplexityHistory': best_complexity_history_cnn_bilstm,
            'savedParetoData': saved_pareto_cnn_bilstm,  # 新增保存的数据
        },
        'BiLSTM': {
            'allParetoFronts': all_pareto_fronts_bilstm,
            'bestParamsHistory': best_params_history_bilstm,
            'bestPerformanceHistory': best_performance_history_bilstm,
            'bestComplexityHistory': best_complexity_history_bilstm,
            'savedParetoData': saved_pareto_bilstm,  # 新增保存的数据
        },
        'GRU': {
            'allParetoFronts': all_pareto_fronts_gru,
            'bestParamsHistory': best_params_history_gru,
            'bestPerformanceHistory': best_performance_history_gru,
            'bestComplexityHistory': best_complexity_history_gru,
            'savedParetoData': saved_pareto_gru,  # 新增保存的数据
        },
        'CNN': {
            'allParetoFronts': all_pareto_fronts_cnn,
            'bestParamsHistory': best_params_history_cnn,
            'bestPerformanceHistory': best_performance_history_cnn,
            'bestComplexityHistory': best_complexity_history_cnn,
            'savedParetoData': saved_pareto_cnn,  # 新增保存的数据
        },
        'Transformer': {
            'allParetoFronts': all_pareto_fronts_transformer,
            'bestParamsHistory': best_params_history_transformer,
            'bestPerformanceHistory': best_performance_history_transformer,
            'bestComplexityHistory': best_complexity_history_transformer,
            'savedParetoData': saved_pareto_transformer,  # 新增保存的数据
        }
    }, f)

print('优化结果已保存至 nsga2_optimization_results_extended.pkl')


def select_best_params(all_pareto_fronts):
    final_pareto = all_pareto_fronts[-1] if all_pareto_fronts else {'num_solutions': 0}

    all_valid_pareto = []
    all_valid_perf = []
    all_valid_comp = []
    for front in all_pareto_fronts:
        if front['num_solutions'] > 0:
            all_valid_pareto.extend(front['params'])
            all_valid_perf.extend(front['performance'])
            all_valid_comp.extend(front['complexity'])

    if final_pareto['num_solutions'] == 0 and all_valid_pareto:
        final_pareto = {
            'params': np.array(all_valid_pareto),
            'performance': np.array(all_valid_perf),
            'complexity': np.array(all_valid_comp),
            'num_solutions': len(all_valid_pareto)
        }

    if final_pareto['num_solutions'] > 0:
        perf_vals = final_pareto['performance']
        complexity_vals = final_pareto['complexity']
        normalized_perf = (perf_vals - np.min(perf_vals)) / (np.max(perf_vals) - np.min(perf_vals) + 1e-10)
        normalized_comp = (complexity_vals - np.min(complexity_vals)) / (
                np.max(complexity_vals) - np.min(complexity_vals) + 1e-10)
        trade_off_scores = np.sqrt(normalized_perf ** 2 + normalized_comp ** 2)
        trade_off_idx = np.argmin(trade_off_scores)
        return final_pareto['params'][trade_off_idx]
    else:
        # Fallback to the single best historical performance if no Pareto points exist
        if all_valid_perf:
            best_idx = np.argmin(all_valid_perf)
            return all_valid_pareto[best_idx]
        raise ValueError('最终Pareto前沿无有效解')


best_params_cnn_bilstm = select_best_params(all_pareto_fronts_cnn_bilstm)
best_params_bilstm = select_best_params(all_pareto_fronts_bilstm)
best_params_gru = select_best_params(all_pareto_fronts_gru)
best_params_cnn = select_best_params(all_pareto_fronts_cnn)
best_params_transformer = select_best_params(all_pareto_fronts_transformer)


# ==================== 7. 最终模型训练与评估 ====================

def train_and_evaluate_model(model, train_loader, val_loader, test_loader, criterion,
                             optimizer, scheduler, patience, max_epochs, min_speed,
                             max_speed, model_name):
    best_val_loss = float('inf')
    counter = 0
    train_losses = []
    val_losses = []
    best_model_state = deepcopy(model.state_dict())

    for epoch in range(max_epochs):
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
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step()
        print(f'{model_name} Epoch {epoch + 1}/{max_epochs}, 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            counter += 1
            if counter >= patience:
                print(f'{model_name} 早停于第 {epoch + 1} 轮')
                break

    model.load_state_dict(best_model_state)
    model.eval()

    # 验证集评估
    all_val_targets = []
    all_val_outputs = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_val_targets.extend(targets.cpu().numpy())
            all_val_outputs.extend(outputs.cpu().numpy())

    all_val_targets = np.array(all_val_targets) * (max_speed - min_speed) + min_speed
    all_val_outputs = np.array(all_val_outputs) * (max_speed - min_speed) + min_speed

    mae_val = mean_absolute_error(all_val_targets, all_val_outputs)
    rmse_val = np.sqrt(mean_squared_error(all_val_targets, all_val_outputs))
    mape_val = np.mean(np.abs((all_val_targets - all_val_outputs) / all_val_targets)) * 100
    corr_matrix_val = np.corrcoef(all_val_targets.flatten(), all_val_outputs.flatten())
    r_val = corr_matrix_val[0, 1] if corr_matrix_val.shape[0] >= 2 else 0

    print(f'{model_name} 验证集性能: MAE = {mae_val:.4f} (m/s), RMSE = {rmse_val:.4f} (m/s), '
          f'MAPE = {mape_val:.2f}%, R = {r_val:.4f}')

    # 测试集评估
    all_test_targets = []
    all_test_outputs = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_test_targets.extend(targets.cpu().numpy())
            all_test_outputs.extend(outputs.cpu().numpy())

    all_test_targets = np.array(all_test_targets) * (max_speed - min_speed) + min_speed
    all_test_outputs = np.array(all_test_outputs) * (max_speed - min_speed) + min_speed

    mae_test = mean_absolute_error(all_test_targets, all_test_outputs)
    rmse_test = np.sqrt(mean_squared_error(all_test_targets, all_test_outputs))
    mape_test = np.mean(np.abs((all_test_targets - all_test_outputs) / all_test_targets)) * 100
    corr_matrix_test = np.corrcoef(all_test_targets.flatten(), all_test_outputs.flatten())
    r_test = corr_matrix_test[0, 1] if corr_matrix_test.shape[0] >= 2 else 0

    print(f'{model_name} 测试集性能: MAE = {mae_test:.4f} (m/s), RMSE = {rmse_test:.4f} (m/s), '
          f'MAPE = {mape_test:.2f}%, R = {r_test:.4f}')

    return model, all_val_targets, all_val_outputs, all_test_targets, all_test_outputs, mae_val, rmse_val, mape_val, r_val, mae_test, rmse_test, mape_test, r_test


# CNNBiLSTM模型训练 (使用优化后的参数)
batch_size = round(best_params_cnn_bilstm[0])
learn_rate = best_params_cnn_bilstm[1]
pool_type = round(best_params_cnn_bilstm[2])
num_filters1 = round(best_params_cnn_bilstm[3])
filter_size1 = round(best_params_cnn_bilstm[4])
lstm_units1 = round(best_params_cnn_bilstm[5])
lstm_units2 = round(best_params_cnn_bilstm[6])
reg_type = round(best_params_cnn_bilstm[7])
dropout_prob1 = best_params_cnn_bilstm[8]
dropout_prob2 = best_params_cnn_bilstm[9]
optimizer_type = round(best_params_cnn_bilstm[10])
num_conv_layers = round(best_params_cnn_bilstm[11])
conv_dropout = best_params_cnn_bilstm[12]
activation_function = round(best_params_cnn_bilstm[13])

num_filters2 = max(8, num_filters1 // 2) if num_conv_layers == 2 else 0
filter_size2 = filter_size1

model_cnn_bilstm = CNNBiLSTM(
    num_features, num_filters1, filter_size1, num_filters2, filter_size2,
    lstm_units1, lstm_units2, pool_type, num_conv_layers,
    dropout_prob1, dropout_prob2, conv_dropout, activation_function
).to(device)

criterion = nn.MSELoss()
weight_decay_opt = 1e-4 if reg_type in [1, 3] else 0

if optimizer_type == 1:
    optimizer = optim.Adam(model_cnn_bilstm.parameters(), lr=learn_rate,
                           weight_decay=weight_decay_opt)
elif optimizer_type == 2:
    optimizer = optim.SGD(model_cnn_bilstm.parameters(), lr=learn_rate, momentum=0.9,
                          weight_decay=weight_decay_opt)
else:
    optimizer = optim.RMSprop(model_cnn_bilstm.parameters(), lr=learn_rate,
                              weight_decay=weight_decay_opt)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.15)
train_loader_cnn_bilstm = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader_cnn_bilstm = DataLoader(val_dataset, batch_size=batch_size)
test_loader_cnn_bilstm = DataLoader(test_dataset, batch_size=batch_size)

model_cnn_bilstm, all_val_targets_opt, all_val_outputs_opt, all_test_targets, all_test_outputs_opt, mae_val_opt, rmse_val_opt, mape_val_opt, r_val_opt, mae_test_opt, rmse_test_opt, mape_test_opt, r_test_opt = train_and_evaluate_model(
    model_cnn_bilstm, train_loader_cnn_bilstm, val_loader_cnn_bilstm, test_loader_cnn_bilstm,
    criterion, optimizer, scheduler, 10, 200, min_speed, max_speed, 'CNNBiLSTM'  # 修改为200
)

# BiLSTM模型训练
batch_size_bilstm = round(best_params_bilstm[0])
learn_rate_bilstm = best_params_bilstm[1]
lstm_units1_bilstm = round(best_params_bilstm[2])
lstm_units2_bilstm = round(best_params_bilstm[3])
reg_type_bilstm = round(best_params_bilstm[4])
dropout_prob1_bilstm = best_params_bilstm[5]
dropout_prob2_bilstm = best_params_bilstm[6]
optimizer_type_bilstm = round(best_params_bilstm[7])

model_bilstm = BiLSTM(num_features, lstm_units1_bilstm, lstm_units2_bilstm, dropout_prob1_bilstm,
                      dropout_prob2_bilstm).to(device)
criterion_bilstm = nn.MSELoss()
weight_decay_bilstm = 1e-4 if reg_type_bilstm in [1, 3] else 0

if optimizer_type_bilstm == 1:
    optimizer_bilstm = optim.Adam(model_bilstm.parameters(), lr=learn_rate_bilstm,
                                  weight_decay=weight_decay_bilstm)
elif optimizer_type_bilstm == 2:
    optimizer_bilstm = optim.SGD(model_bilstm.parameters(), lr=learn_rate_bilstm, momentum=0.9,
                                 weight_decay=weight_decay_bilstm)
else:
    optimizer_bilstm = optim.RMSprop(model_bilstm.parameters(), lr=learn_rate_bilstm,
                                     weight_decay=weight_decay_bilstm)

scheduler_bilstm = optim.lr_scheduler.StepLR(optimizer_bilstm, step_size=20, gamma=0.15)
train_loader_bilstm = DataLoader(train_dataset, batch_size=batch_size_bilstm, shuffle=True)
val_loader_bilstm = DataLoader(val_dataset, batch_size=batch_size_bilstm)
test_loader_bilstm = DataLoader(test_dataset, batch_size=batch_size_bilstm)

_, _, _, _, all_test_outputs_bilstm, _, _, _, _, mae_test_bilstm, rmse_test_bilstm, mape_test_bilstm, r_test_bilstm = train_and_evaluate_model(
    model_bilstm, train_loader_bilstm, val_loader_bilstm, test_loader_bilstm,
    criterion_bilstm, optimizer_bilstm, scheduler_bilstm, 10, 200, min_speed, max_speed, 'BiLSTM'  # 修改为200
)

# GRU模型训练
batch_size_gru = round(best_params_gru[0])
learn_rate_gru = best_params_gru[1]
gru_units1_gru = round(best_params_gru[2])
gru_units2_gru = round(best_params_gru[3])
reg_type_gru = round(best_params_gru[4])
dropout_prob1_gru = best_params_gru[5]
dropout_prob2_gru = best_params_gru[6]
optimizer_type_gru = round(best_params_gru[7])

model_gru = GRU(num_features, gru_units1_gru, gru_units2_gru, dropout_prob1_gru, dropout_prob2_gru).to(device)
criterion_gru = nn.MSELoss()
weight_decay_gru = 1e-4 if reg_type_gru in [1, 3] else 0

if optimizer_type_gru == 1:
    optimizer_gru = optim.Adam(model_gru.parameters(), lr=learn_rate_gru,
                               weight_decay=weight_decay_gru)
elif optimizer_type_gru == 2:
    optimizer_gru = optim.SGD(model_gru.parameters(), lr=learn_rate_gru, momentum=0.9,
                              weight_decay=weight_decay_gru)
else:
    optimizer_gru = optim.RMSprop(model_gru.parameters(), lr=learn_rate_gru,
                                  weight_decay=weight_decay_gru)

scheduler_gru = optim.lr_scheduler.StepLR(optimizer_gru, step_size=20, gamma=0.15)
train_loader_gru = DataLoader(train_dataset, batch_size=batch_size_gru, shuffle=True)
val_loader_gru = DataLoader(val_dataset, batch_size=batch_size_gru)
test_loader_gru = DataLoader(test_dataset, batch_size=batch_size_gru)

_, _, _, _, all_test_outputs_gru, _, _, _, _, mae_test_gru, rmse_test_gru, mape_test_gru, r_test_gru = train_and_evaluate_model(
    model_gru, train_loader_gru, val_loader_gru, test_loader_gru,
    criterion_gru, optimizer_gru, scheduler_gru, 10, 200, min_speed, max_speed, 'GRU'  # 修改为200
)

# CNN模型训练
batch_size_cnn = round(best_params_cnn[0])
learn_rate_cnn = best_params_cnn[1]
num_filters1_cnn = round(best_params_cnn[2])
filter_size1_cnn = round(best_params_cnn[3])
num_filters2_cnn = round(best_params_cnn[4])
filter_size2_cnn = round(best_params_cnn[5])
reg_type_cnn = round(best_params_cnn[6])
conv_dropout_cnn = best_params_cnn[7]
optimizer_type_cnn = round(best_params_cnn[8])
activation_function_cnn = round(best_params_cnn[9])

model_cnn = CNN(num_features, num_filters1_cnn, filter_size1_cnn, num_filters2_cnn, filter_size2_cnn).to(device)
criterion_cnn = nn.MSELoss()
weight_decay_cnn = 1e-4 if reg_type_cnn in [1, 3] else 0

if optimizer_type_cnn == 1:
    optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=learn_rate_cnn,
                               weight_decay=weight_decay_cnn)
elif optimizer_type_cnn == 2:
    optimizer_cnn = optim.SGD(model_cnn.parameters(), lr=learn_rate_cnn, momentum=0.9,
                              weight_decay=weight_decay_cnn)
else:
    optimizer_cnn = optim.RMSprop(model_cnn.parameters(), lr=learn_rate_cnn,
                                  weight_decay=weight_decay_cnn)

scheduler_cnn = optim.lr_scheduler.StepLR(optimizer_cnn, step_size=20, gamma=0.15)
train_loader_cnn = DataLoader(train_dataset, batch_size=batch_size_cnn, shuffle=True)
val_loader_cnn = DataLoader(val_dataset, batch_size=batch_size_cnn)
test_loader_cnn = DataLoader(test_dataset, batch_size=batch_size_cnn)

_, _, _, _, all_test_outputs_cnn, _, _, _, _, mae_test_cnn, rmse_test_cnn, mape_test_cnn, r_test_cnn = train_and_evaluate_model(
    model_cnn, train_loader_cnn, val_loader_cnn, test_loader_cnn,
    criterion_cnn, optimizer_cnn, scheduler_cnn, 10, 200, min_speed, max_speed, 'CNN'  # 修改为200
)

# Transformer模型训练 (使用优化后的4个参数，固定其他参数)
batch_size_transformer = round(best_params_transformer[0])
learn_rate_transformer = best_params_transformer[1]
dropout_prob_transformer = best_params_transformer[2]
optimizer_type_transformer = round(best_params_transformer[3])

# 固定参数
embedding_dim_transformer = 64
num_heads_transformer = 4
ffn_dim_transformer = 128
reg_type_transformer = 1  # L2 正则化

model_transformer = TransformerModel(num_features, embedding_dim_transformer, num_heads_transformer,
                                     ffn_dim_transformer, sequence_length, dropout_prob_transformer).to(device)
criterion_transformer = nn.MSELoss()
weight_decay_transformer = 1e-4 if reg_type_transformer in [1, 3] else 0

if optimizer_type_transformer == 1:
    optimizer_transformer = optim.Adam(model_transformer.parameters(), lr=learn_rate_transformer,
                                       weight_decay=weight_decay_transformer)
elif optimizer_type_transformer == 2:
    optimizer_transformer = optim.SGD(model_transformer.parameters(), lr=learn_rate_transformer, momentum=0.9,
                                      weight_decay=weight_decay_transformer)
else:
    optimizer_transformer = optim.RMSprop(model_transformer.parameters(), lr=learn_rate_transformer,
                                          weight_decay=weight_decay_transformer)

scheduler_transformer = optim.lr_scheduler.StepLR(optimizer_transformer, step_size=20, gamma=0.15)
train_loader_transformer = DataLoader(train_dataset, batch_size=batch_size_transformer, shuffle=True)
val_loader_transformer = DataLoader(val_dataset, batch_size=batch_size_transformer)
test_loader_transformer = DataLoader(test_dataset, batch_size=batch_size_transformer)

_, _, _, _, all_test_outputs_transformer, _, _, _, _, mae_test_transformer, rmse_test_transformer, mape_test_transformer, r_test_transformer = train_and_evaluate_model(
    model_transformer, train_loader_transformer, val_loader_transformer, test_loader_transformer,
    criterion_transformer, optimizer_transformer, scheduler_transformer, 10, 200, min_speed, max_speed, 'Transformer'  # 修改为200
)

# ==================== 8. 可视化分析 (新增RMSE进化图) ====================

# 新增：RMSE 进化图
plt.figure(figsize=(10, 6))
generations = np.arange(1, max_generations + 1)


# 确保所有历史记录的长度一致
def trim_history(history, max_len):
    return history[:max_len]


max_len = max_generations
histories = {
    'CNNBiLSTM': trim_history(best_performance_history_cnn_bilstm, max_len),
    'BiLSTM': trim_history(best_performance_history_bilstm, max_len),
    'GRU': trim_history(best_performance_history_gru, max_len),
    'CNN': trim_history(best_performance_history_cnn, max_len),
    'Transformer': trim_history(best_performance_history_transformer, max_len)
}

models = list(histories.keys())
colors = ['red', 'green', 'blue', 'purple', 'orange']

print('\n绘制模型RMSE进化图...')

for model, color in zip(models, colors):
    if len(histories[model]) > 0:
        plt.plot(generations[:len(histories[model])], histories[model],
                 label=model, color=color, linewidth=2)

plt.title('各模型最优RMSE随代数变化', fontsize=14)
plt.xlabel('进化代数', fontsize=12)
plt.ylabel('最优RMSE (m/s)', fontsize=12)
plt.legend(loc='best')
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()

# 学习到的特征权重 (仅CNNBiLSTM)
feature_weights = model_cnn_bilstm.feature_weight.weights.detach().cpu().numpy()
print('\n学习到的特征权重（CNNBiLSTM）：')
for i, col in enumerate(feature_columns):
    print(f' {col}: {feature_weights[i, 0]:.4f}')

plt.figure(figsize=(16, 12))
plt.suptitle('最终模型 - 预测分析', fontsize=16)

# 验证集预测效果 (CNNBiLSTM)
plt.subplot(2, 3, 1)
plt.plot(all_val_targets_opt, 'b-', linewidth=1.5, label='验证集真实值')
plt.plot(all_val_outputs_opt, 'r--', linewidth=1.5, label='优化模型预测')
plt.xlabel('时间步')
plt.ylabel('风速 (m/s)')
plt.title(f'验证集预测效果 (RMSE={rmse_val_opt:.3f})')
plt.legend(loc='best')
plt.grid(True)

# 测试集预测效果 (CNNBiLSTM)
plt.subplot(2, 3, 2)
plt.plot(all_test_targets, 'b-', linewidth=1.5, label='测试集真实值')
plt.plot(all_test_outputs_opt, 'r--', linewidth=1.5, label='优化模型预测')
plt.xlabel('时间步')
plt.ylabel('风速 (m/s)')
plt.title(f'测试集预测效果 (RMSE={rmse_test_opt:.3f})')
plt.legend(loc='best')
plt.grid(True)

# 测试集误差分布 (CNNBiLSTM)
plt.subplot(2, 3, 3)
errors_test_opt = all_test_targets - all_test_outputs_opt
plt.hist(errors_test_opt, 30, density=True, alpha=0.7)
plt.xlabel('预测误差 (m/s)')
plt.ylabel('概率密度')
plt.title(f'测试集误差分布 (均值={np.mean(errors_test_opt):.3f})')
plt.grid(True)

# 预测 vs 真实 (CNNBiLSTM)
plt.subplot(2, 3, 4)
plt.scatter(all_test_targets, all_test_outputs_opt, 50, alpha=0.6)
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
lstm_metrics = [mae_test_bilstm, rmse_test_bilstm, mape_test_bilstm, r_test_bilstm]
gru_metrics = [mae_test_gru, rmse_test_gru, mape_test_gru, r_test_gru]
cnn_metrics = [mae_test_cnn, rmse_test_cnn, mape_test_cnn, r_test_cnn]
transformer_metrics = [mae_test_transformer, rmse_test_transformer, mape_test_transformer, r_test_transformer]
x = np.arange(len(metrics))
width = 0.15
plt.bar(x - 2 * width, opt_metrics, width, label='CNNBiLSTM')
plt.bar(x - width, lstm_metrics, width, label='BiLSTM')
plt.bar(x, gru_metrics, width, label='GRU')
plt.bar(x + width, cnn_metrics, width, label='CNN')
plt.bar(x + 2 * width, transformer_metrics, width, label='Transformer')
plt.xticks(x, metrics)
plt.ylabel('指标值')
plt.title('不同模型测试集性能对比')
plt.legend(loc='best')
plt.grid(True, axis='y')

# 优化模型测试集绝对误差时间序列 (CNNBiLSTM)
plt.subplot(2, 3, 6)
plt.plot(np.abs(errors_test_opt), 'g-', linewidth=1)
plt.xlabel('时间步')
plt.ylabel('绝对误差 (m/s)')
plt.title('CNNBiLSTM 测试集绝对误差时间序列')
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

# 特征权重可视化 (CNNBiLSTM)
plt.figure(figsize=(10, 6))
weights = feature_weights.flatten()
plt.bar(range(len(feature_columns)), weights, color=[0.2, 0.5, 0.8])
plt.xticks(range(len(feature_columns)), feature_columns, rotation=15)
plt.xlabel('输入特征')
plt.ylabel('学习到的权重（越大越重要）')
plt.title('CNNBiLSTM 中各输入特征的权重')
plt.grid(True, axis='y')
for i, w in enumerate(weights):
    plt.text(i, w + 0.02, f'{w:.4f}', ha='center', fontsize=9)
plt.tight_layout()
plt.show()

# 各模型预测对比
plt.figure(figsize=(16, 10))
plt.suptitle('各模型预测 vs 真实值', fontsize=16)

# CNNBiLSTM
plt.subplot(2, 3, 1)
plt.plot(all_test_targets, 'b-', linewidth=1.5, label='真实值')
plt.plot(all_test_outputs_opt, 'r--', linewidth=1.5, label='CNNBiLSTM')
plt.title('CNNBiLSTM预测 vs 真实')
plt.xlabel('时间步')
plt.ylabel('风速 (m/s)')
plt.legend(loc='best')
plt.grid(True)

# BiLSTM
plt.subplot(2, 3, 2)
plt.plot(all_test_targets, 'b-', linewidth=1.5, label='真实值')
plt.plot(all_test_outputs_bilstm, 'g--', linewidth=1.5, label='BiLSTM')
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
plt.plot(all_test_outputs_transformer, 'y--', linewidth=1.5, label='Transformer')
plt.title('Transformer预测 vs 真实')
plt.xlabel('时间步')
plt.ylabel('风速 (m/s)')
plt.legend(loc='best')
plt.grid(True)

# 所有模型对比
plt.subplot(2, 3, 6)
plt.plot(all_test_targets, 'k-', linewidth=2, label='真实值')
plt.plot(all_test_outputs_opt, 'r--', linewidth=1, label='CNNBiLSTM')
plt.plot(all_test_outputs_bilstm, 'g--', linewidth=1, label='BiLSTM')
plt.plot(all_test_outputs_gru, 'm--', linewidth=1, label='GRU')
plt.plot(all_test_outputs_cnn, 'c--', linewidth=1, label='CNN')
plt.plot(all_test_outputs_transformer, 'y--', linewidth=1, label='Transformer')
plt.title('所有模型预测对比')
plt.xlabel('时间步')
plt.ylabel('风速 (m/s)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# ==================== 新增：绘制各模型特定代数的Pareto前沿进化图 ====================

def plot_pareto_evolution_for_model(saved_pareto_data, model_name, max_generations):
    """绘制特定代数的Pareto前沿进化图"""
    plt.figure(figsize=(12, 8))

    # 定义代数和对应的颜色、标记
    generations_to_plot = [1, 5, 10, 20, 40, 60]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', '*', 'x']
    sizes = [50, 60, 70, 80, 90, 100]

    for idx, gen in enumerate(generations_to_plot):
        if gen > max_generations:
            continue  # 跳过超过最大代数的点

        if saved_pareto_data[gen] is not None and saved_pareto_data[gen]['num_solutions'] > 0:
            # 按照复杂度排序
            sorted_indices = np.argsort(saved_pareto_data[gen]['complexity'])
            sorted_complexity = saved_pareto_data[gen]['complexity'][sorted_indices]
            sorted_performance = saved_pareto_data[gen]['performance'][sorted_indices]

            # 绘制散点
            plt.scatter(
                sorted_complexity,
                sorted_performance,
                c=colors[idx],
                marker=markers[idx],
                s=sizes[idx],
                alpha=0.6,
                label=f'第{gen}代 (n={saved_pareto_data[gen]["num_solutions"]})',
                edgecolors='black',
                linewidth=0.5
            )

            # 连接点
            plt.plot(
                sorted_complexity,
                sorted_performance,
                color=colors[idx],
                linestyle='-',
                linewidth=1.5,
                alpha=0.5
            )

    plt.xlabel('模型复杂度（参数数量）', fontsize=12)
    plt.ylabel('预测误差 (RMSE, m/s)', fontsize=12)
    plt.title(f'{model_name} - Pareto前沿进化过程', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# 为每个模型绘制进化图
print("\n绘制各模型特定代数的Pareto前沿进化图...")

# CNNBiLSTM
plot_pareto_evolution_for_model(saved_pareto_cnn_bilstm, 'CNNBiLSTM', max_generations)

# BiLSTM
plot_pareto_evolution_for_model(saved_pareto_bilstm, 'BiLSTM', max_generations)

# GRU
plot_pareto_evolution_for_model(saved_pareto_gru, 'GRU', max_generations)

# CNN
plot_pareto_evolution_for_model(saved_pareto_cnn, 'CNN', max_generations)

# Transformer
plot_pareto_evolution_for_model(saved_pareto_transformer, 'Transformer', max_generations)

# =============================================================================


# ==================== 新增：不同模型参数数量对比柱状图 ====================

# 计算每个模型的参数数量 (使用最终训练后的模型)
complexity_cnn_bilstm = sum(p.numel() for p in model_cnn_bilstm.parameters())
complexity_bilstm = sum(p.numel() for p in model_bilstm.parameters())
complexity_gru = sum(p.numel() for p in model_gru.parameters())
complexity_cnn = sum(p.numel() for p in model_cnn.parameters())
complexity_transformer = sum(p.numel() for p in model_transformer.parameters())

# 绘制柱状图
plt.figure(figsize=(10, 6))
models_list = ['CNNBiLSTM', 'BiLSTM', 'GRU', 'CNN', 'Transformer']
complexities = [complexity_cnn_bilstm, complexity_bilstm, complexity_gru, complexity_cnn, complexity_transformer]

plt.bar(models_list, complexities, color=['red', 'green', 'blue', 'purple', 'orange'])
plt.xlabel('模型')
plt.ylabel('参数数量')
plt.title('不同模型参数数量对比')
plt.grid(True, axis='y')
for i, comp in enumerate(complexities):
    plt.text(i, comp + 0.02 * max(complexities), f'{comp}', ha='center', fontsize=9)
plt.tight_layout()
plt.show()

# =============================================================================


# ==================== 9. 打印优化后的超参数 ====================

param_names_cnn_bilstm = ['Batch Size', 'Learn Rate', 'Pool Type',
                          'Num Filters1', 'Filter Size1', 'LSTM Units1',
                          'LSTM Units2', 'Reg Type', 'Dropout Prob1', 'Dropout Prob2',
                          'OptimizerType', 'NumConvLayers', 'ConvDropout', 'ActivationFunction']
print('\n=== CNNBiLSTM 最终优化的超参数 ===')
for i, name in enumerate(param_names_cnn_bilstm):
    if i in int_con_cnn_bilstm:
        print(f'{name}: {round(best_params_cnn_bilstm[i])}')
    else:
        print(f'{name}: {best_params_cnn_bilstm[i]:.6f}')

param_names_transformer = ['Batch Size', 'Learn Rate', 'Dropout Prob', 'Optimizer Type']
print('\n=== Transformer 最终优化的超参数 (简化版) ===')
# 注意: best_params_transformer 只有4个元素
for i, name in enumerate(param_names_transformer):
    if i in int_con_transformer:
        print(f'{name}: {round(best_params_transformer[i])}')
    else:
        print(f'{name}: {best_params_transformer[i]:.6f}')
print('以下参数已固定: Embedding Dim=64, Num Heads=4, FFN Dim=128, Reg Type=1')

# ==================== 10. 生成优化摘要报告 ====================

print('\n=== CNNBiLSTM NSGA-II 优化摘要报告 ===')
print(f'总进化代数: {max_generations}')
print(f'种群大小: {population_size}')
final_pareto_struct_cnn_bilstm = all_pareto_fronts_cnn_bilstm[-1] if all_pareto_fronts_cnn_bilstm else {
    'num_solutions': 0}
print(f'最终Pareto解数量: {final_pareto_struct_cnn_bilstm["num_solutions"]}')
if len(best_performance_history_cnn_bilstm) > 0:
    print(f'初始最优RMSE: {best_performance_history_cnn_bilstm[0]:.4f} (m/s)')
    print(f'最终最优RMSE: {best_performance_history_cnn_bilstm[-1]:.4f} (m/s)')
    if best_performance_history_cnn_bilstm[0] > 0:
        improvement = (best_performance_history_cnn_bilstm[0] - best_performance_history_cnn_bilstm[-1]) / \
                      best_performance_history_cnn_bilstm[0] * 100
        print(f'性能改进: {improvement:.2f}%')
else:
    print('未记录性能历史。')

print('\n=== 最终模型测试集性能 ===')
print(f'CNNBiLSTM: MAE={mae_test_opt:.4f}, RMSE={rmse_test_opt:.4f}, MAPE={mape_test_opt:.2f}%, R={r_test_opt:.4f}')
print(
    f'BiLSTM: MAE={mae_test_bilstm:.4f}, RMSE={rmse_test_bilstm:.4f}, MAPE={mape_test_bilstm:.2f}%, R={r_test_bilstm:.4f}')
print(f'GRU: MAE={mae_test_gru:.4f}, RMSE={rmse_test_gru:.4f}, MAPE={mape_test_gru:.2f}%, R={r_test_gru:.4f}')
print(f'CNN: MAE={mae_test_cnn:.4f}, RMSE={rmse_test_cnn:.4f}, MAPE={mape_test_cnn:.2f}%, R={r_test_cnn:.4f}')
print(
    f'Transformer: MAE={mae_test_transformer:.4f}, RMSE={rmse_test_transformer:.4f}, MAPE={mape_test_transformer:.2f}%, R={r_test_transformer:.4f}')

print('\n可视化图表已生成。')
print('优化完成！所有结果已保存至 nsga2_optimization_results_extended.pkl')
