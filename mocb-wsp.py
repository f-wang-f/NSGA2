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
except Exception as e:
    print(f'无法读取 Excel 文件: {e}')
    raise

# 数据清洗
valid_rows = np.all(~np.isnan(feature_data) & ~np.isinf(feature_data), axis=1)
feature_data = feature_data[valid_rows, :]
if feature_data.shape[0] < 100:
    raise ValueError('数据不足(少于100个样本),请提供更多数据。')

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
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length, -1]
        return torch.FloatTensor(x), torch.FloatTensor([y])

sequence_length = 1
dataset = WindDataset(feature_data_norm, sequence_length)

# 按6:1:1比例划分（实际为7:1.5:1.5）
train_ratio = 0.7
val_ratio = 0.15
num_samples = len(dataset)
num_train = int(train_ratio * num_samples)
num_val = int(val_ratio * num_samples)
num_test = num_samples - num_train - num_val

train_dataset = Subset(dataset, range(num_train))
val_dataset = Subset(dataset, range(num_train, num_train + num_val))
test_dataset = Subset(dataset, range(num_train + num_val, num_samples))

min_speed = min_vals[-1]
max_speed = max_vals[-1]
print(f'数据预处理完成。训练样本数:{num_train},验证样本数:{num_val},测试样本数:{num_test}')

# ==================== 2. 模型定义(保持原结构) ====================
class HybridCNNBiLSTM(nn.Module):
    def __init__(self, topo, cnn_params, lstm_params, num_features, sequence_length):
        super(HybridCNNBiLSTM, self).__init__()
        self.num_features = num_features
        self.sequence_length = sequence_length
        n = 5
        self.n = n
        self.feature_weights = nn.Parameter(torch.ones(num_features))

        # 解析拓扑结构
        bits_length = n * (n - 1) // 2
        if len(topo) != bits_length:
            raise ValueError(f"拓扑长度不匹配: {len(topo)} != {bits_length}")
        self.adj = np.zeros((n, n), dtype=int)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                self.adj[i, j] = topo[k]
                k += 1

        # 计算每个模块的输入通道数
        in_channels_list = [0] * n
        in_channels_list[0] = num_features
        for i in range(1, n):
            num_incoming = np.sum(self.adj[:, i])
            if num_incoming == 0:
                num_incoming = 1
            incoming_channels = []
            for j in range(i):
                if self.adj[j, i] == 1:
                    if j < 3:
                        incoming_channels.append(16 * (cnn_params[j][0] + 1))
                    else:
                        incoming_channels.append(32 * (lstm_params[j - 3][0] + 1))
            if incoming_channels:
                in_channels_list[i] = sum(incoming_channels)
            else:
                in_channels_list[i] = 16

        # 创建模块
        self.convs = nn.ModuleList()
        self.lstms = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.adaptive_pools = nn.ModuleList()

        act_map = {
            0: nn.Softplus, 1: nn.Softsign, 2: nn.ELU, 3: nn.Softmax,
            4: nn.Sigmoid, 5: nn.Tanh, 6: nn.ReLU, 7: nn.Identity
        }

        for i in range(n):
            in_channels = max(in_channels_list[i], 1)
            if i < 3:  # CNN modules
                knum, ksize, kact, pt, ps = cnn_params[i]
                out_channels = 16 * (knum + 1)
                kernel_size = 2 * ksize + 3
                self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding='same'))
            else:  # BiLSTM modules
                knum, lnum, kact, pt, ps = lstm_params[i - 3]
                hidden = 16 * (knum + 1)
                dropout = ps * 0.1
                num_layers = lnum + 1
                self.lstms.append(
                    nn.LSTM(in_channels, hidden, num_layers=num_layers,
                            bidirectional=True, batch_first=False, dropout=dropout)
                )
                out_channels = 2 * hidden
                self.bns.append(nn.BatchNorm1d(out_channels))
                if kact == 3:
                    self.acts.append(act_map[kact](dim=1))
                else:
                    self.acts.append(act_map[kact]())
                effective_ps = max(ps, 1)
                pool_size = 2 * effective_ps + 3
                if pt == 0:
                    self.pools.append(nn.MaxPool1d(pool_size, stride=1, padding=(pool_size - 1) // 2))
                elif pt == 1:
                    self.pools.append(nn.AvgPool1d(pool_size, stride=1, padding=(pool_size - 1) // 2))
                else:
                    self.pools.append(nn.Identity())
                self.adaptive_pools.append(nn.AdaptiveAvgPool1d(sequence_length))

        total_out_channels = sum(
            [16 * (cnn_params[i][0] + 1) if i < 3 else 32 * (lstm_params[i - 3][0] + 1) for i in range(n)]
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(total_out_channels, 1)

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("输入包含NaN或Inf值")
        x = x.transpose(1, 2)
        x = x * self.feature_weights.unsqueeze(0).unsqueeze(2)

        outputs = []
        for i in range(self.n):
            inputs = []
            if i == 0:
                inputs.append(x)
            else:
                for prev in range(i):
                    if self.adj[prev, i] == 1:
                        inputs.append(outputs[prev])
            if not inputs:
                inputs.append(x)
            input_i = torch.cat(inputs, dim=1) if len(inputs) > 1 else inputs[0]

            if i < 3:
                out = self.convs[i](input_i)
            else:
                input_lstm = input_i.permute(2, 0, 1)
                out_lstm, _ = self.lstms[i - 3](input_lstm)
                out = out_lstm.permute(1, 2, 0)
                out = self.bns[i-3](out)
                out = self.acts[i-3](out)
                out = self.pools[i-3](out)
                out = self.adaptive_pools[i-3](out)

            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError(f"模块{i}输出包含NaN或Inf值")
            outputs.append(out)

        final = torch.cat(outputs, dim=1)
        final = self.global_pool(final).squeeze(-1)
        final = self.fc(final)
        return final

# ==================== 3. 编码和解码函数 ====================
def encode_individual(topo, cnn_params, lstm_params, setting):
    individual = topo + [item for sublist in cnn_params for item in sublist] + \
                 [item for sublist in lstm_params for item in sublist] + setting
    return np.array(individual, dtype=float)

def decode_individual(individual):
    idx = 0
    n = 5
    bits_length = n * (n - 1) // 2
    topo = [int(round(individual[idx + i])) for i in range(bits_length)]
    idx += bits_length

    cnn_params = []
    for i in range(3):
        knum = int(round(individual[idx]))
        ksize = int(round(individual[idx + 1]))
        kact = int(round(individual[idx + 2]))
        pt = int(round(individual[idx + 3]))
        ps = int(round(individual[idx + 4]))
        knum = max(0, min(7, knum))
        ksize = max(0, min(3, ksize))
        kact = max(0, min(7, kact))
        pt = max(0, min(2, pt))
        ps = max(0, min(3, ps))
        cnn_params.append([knum, ksize, kact, pt, ps])
        idx += 5

    lstm_params = []
    for i in range(2):
        knum = int(round(individual[idx]))
        lnum = int(round(individual[idx + 1]))
        kact = int(round(individual[idx + 2]))
        pt = int(round(individual[idx + 3]))
        ps = int(round(individual[idx + 4]))
        knum = max(0, min(7, knum))
        lnum = max(0, min(3, lnum))
        kact = max(0, min(7, kact))
        pt = max(0, min(2, pt))
        ps = max(0, min(3, ps))
        lstm_params.append([knum, lnum, kact, pt, ps])
        idx += 5

    bs = int(round(individual[idx]))
    opt = int(round(individual[idx + 1]))
    lr = individual[idx + 2]
    reg = int(round(individual[idx + 3]))
    bs = max(0, min(3, bs))
    opt = max(0, min(3, opt))
    lr = max(0.0001, min(0.01, lr))
    reg = max(0, min(3, reg))
    setting = [bs, opt, lr, reg]
    return topo, cnn_params, lstm_params, setting

def get_individual_length():
    n = 5
    return n * (n - 1) // 2 + 5 * 3 + 5 * 2 + 4

# ==================== 4. 初始化 ====================
def initialize_individual():
    n = 5
    bits_length = n * (n - 1) // 2
    topo = [np.random.randint(0, 2) for _ in range(bits_length)]
    cnn_params = []
    for i in range(3):
        knum = np.random.randint(0, 8)
        ksize = np.random.randint(0, 4)
        kact = np.random.randint(0, 8)
        pt = np.random.randint(0, 3)
        ps = np.random.randint(0, 4)
        cnn_params.append([knum, ksize, kact, pt, ps])
    lstm_params = []
    for i in range(2):
        knum = np.random.randint(0, 8)
        lnum = np.random.randint(0, 4)
        kact = np.random.randint(0, 8)
        pt = np.random.randint(0, 3)
        ps = np.random.randint(0, 4)
        lstm_params.append([knum, lnum, kact, pt, ps])
    bs = np.random.randint(0, 4)
    opt = np.random.randint(0, 4)
    lr = np.random.uniform(0.0001, 0.01)
    reg = np.random.randint(0, 4)
    setting = [bs, opt, lr, reg]
    return encode_individual(topo, cnn_params, lstm_params, setting)

def is_valid_individual(individual):
    try:
        topo, cnn_params, lstm_params, setting = decode_individual(individual)
        n = 5
        for i in range(1, n):
            num_incoming = 0
            k = 0
            for ii in range(n):
                for jj in range(ii + 1, n):
                    if jj == i and topo[k] == 1:
                        num_incoming += 1
                    k += 1
            if num_incoming == 0:
                return False
        return True
    except:
        return False

def initialize_population(pop_size):
    population = []
    attempts = 0
    max_attempts = pop_size * 100
    while len(population) < pop_size and attempts < max_attempts:
        ind = initialize_individual()
        if is_valid_individual(ind):
            try:
                topo, cnn_params, lstm_params, setting = decode_individual(ind)
                batch_size, _, _, _ = decode_hyperparams(setting)
                if batch_size <= len(train_dataset):
                    population.append(ind)
            except:
                pass
        attempts += 1
    if len(population) < pop_size:
        raise ValueError(f"初始化失败:仅生成{len(population)}/{pop_size}个有效个体")
    return population

# ==================== 5. 变异与交叉 ====================
def variable_length_mutation(individual):
    topo, cnn_params, lstm_params, setting = decode_individual(individual)
    n = 5
    max_attempts = 100
    attempts = 0
    cnn_param_bounds = [(0, 7), (0, 3), (0, 7), (0, 2), (0, 3)]
    lstm_param_bounds = [(0, 7), (0, 3), (0, 7), (0, 2), (0, 3)]
    setting_bounds = [(0, 3), (0, 3), (0.0001, 0.01), (0, 3)]

    while attempts < max_attempts:
        mutation_region = np.random.choice([0, 1, 2])
        new_topo = topo.copy()
        new_cnn_params = [p.copy() for p in cnn_params]
        new_lstm_params = [p.copy() for p in lstm_params]
        new_setting = setting.copy()
        mutated = False

        if mutation_region == 0:
            bits_length = len(topo)
            while True:
                candidate_topo = [np.random.randint(0, 2) for _ in range(bits_length)]
                if candidate_topo != topo:
                    temp_ind = encode_individual(candidate_topo, new_cnn_params, new_lstm_params, new_setting)
                    if is_valid_individual(temp_ind):
                        new_topo = candidate_topo
                        mutated = True
                        break
        elif mutation_region == 1:
            module_type = np.random.choice([0, 1])
            if module_type == 0:
                module_idx = np.random.randint(0, 3)
                param_idx = np.random.randint(0, 5)
                original_val = new_cnn_params[module_idx][param_idx]
                min_val, max_val = cnn_param_bounds[param_idx]
                candidate_vals = [v for v in range(min_val, max_val + 1) if v != original_val]
                if candidate_vals:
                    new_val = np.random.choice(candidate_vals)
                    new_cnn_params[module_idx][param_idx] = new_val
                    mutated = True
            else:
                module_idx = np.random.randint(0, 2)
                param_idx = np.random.randint(0, 5)
                original_val = new_lstm_params[module_idx][param_idx]
                min_val, max_val = lstm_param_bounds[param_idx]
                candidate_vals = [v for v in range(min_val, max_val + 1) if v != original_val]
                if candidate_vals:
                    new_val = np.random.choice(candidate_vals)
                    new_lstm_params[module_idx][param_idx] = new_val
                    mutated = True
        elif mutation_region == 2:
            param_idx = np.random.randint(0, 4)
            original_val = new_setting[param_idx]
            if param_idx == 2:
                min_val, max_val = setting_bounds[param_idx]
                while True:
                    new_val = np.random.uniform(min_val, max_val)
                    if abs(new_val - original_val) > 0.00001:
                        new_setting[param_idx] = new_val
                        mutated = True
                        break
            else:
                min_val, max_val = setting_bounds[param_idx]
                candidate_vals = [v for v in range(int(min_val), int(max_val) + 1) if v != original_val]
                if candidate_vals:
                    new_val = np.random.choice(candidate_vals)
                    new_setting[param_idx] = new_val
                    mutated = True

        if mutated:
            new_individual = encode_individual(new_topo, new_cnn_params, new_lstm_params, new_setting)
            if not np.array_equal(new_individual, individual) and is_valid_individual(new_individual):
                return new_individual
        attempts += 1
    print(f"警告:变异尝试{max_attempts}次后仍未生成有效子代,返回父代副本")
    return individual.copy()

def crossover_population(mating_pool):
    offspring = []
    crossover_prob = 0.8
    for i in range(0, len(mating_pool), 2):
        if i + 1 < len(mating_pool):
            p1, p2 = mating_pool[i], mating_pool[i + 1]
            if np.random.random() < crossover_prob:
                min_len = min(len(p1), len(p2))
                if min_len > 1:
                    k = np.random.randint(1, min_len)
                    c1 = np.concatenate([p1[:k], p2[k:]])
                    c2 = np.concatenate([p2[:k], p1[k:]])
                    if not is_valid_individual(c1):
                        c1 = p1.copy()
                    if not is_valid_individual(c2):
                        c2 = p2.copy()
                    offspring.append(c1)
                    offspring.append(c2)
                else:
                    offspring.append(p1.copy())
                    offspring.append(p2.copy())
            else:
                offspring.append(p1.copy())
                offspring.append(p2.copy())
        else:
            offspring.append(mating_pool[i].copy())
    return offspring

# ==================== 6. 超参数解码 ====================
def decode_hyperparams(setting):
    bs, opt, lr, reg = setting
    batch_size = 32 * (bs + 1) if bs < 3 else 128
    learn_rate = lr
    optimizer_map = {0: 'SGD', 1: 'Adam', 2: 'AdaDelta', 3: 'RMSprop'}
    reg_map = {0: None, 1: 'L1', 2: 'L2', 3: 'L1L2'}
    return batch_size, learn_rate, optimizer_map[opt], reg_map[reg]

# ==================== 7. 模型评估（核心修复点） ====================
def evaluate_individual(individual, train_dataset, val_dataset, min_speed, max_speed, num_features, sequence_length, device, epochs=20):
    try:
        topo, cnn_params, lstm_params, setting = decode_individual(individual)
        batch_size, learn_rate, opt_type, _ = decode_hyperparams(setting)

        # ====================== 关键修复1：强制 batch_size >= 2 ======================
        batch_size = max(2, batch_size)
        max_batch_size = min(batch_size, len(train_dataset) // 10)
        max_batch_size = max(2, max_batch_size)
        batch_size = max_batch_size

        model = HybridCNNBiLSTM(topo, cnn_params, lstm_params, num_features, sequence_length).to(device)
        model_size = sum(p.numel() for p in model.parameters())
        if model_size > 1e7:
            raise ValueError(f"模型过大: {model_size} 参数")

        # ====================== 关键修复2：train_loader 添加 drop_last=True ======================
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        criterion = nn.MSELoss()
        if opt_type == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        elif opt_type == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)
        elif opt_type == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learn_rate)
        else:
            optimizer = optim.Adadelta(model.parameters(), lr=learn_rate)

        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    raise ValueError("输入数据包含NaN或Inf")
                optimizer.zero_grad()
                outputs = model(inputs)
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    raise ValueError("模型输出包含NaN或Inf")
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            val_loss = 0
            val_samples = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, targets).item() * inputs.size(0)
                    val_loss += batch_loss
                    val_samples += inputs.size(0)
            if val_samples > 0:
                val_loss /= val_samples
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 10:
                break

        model.eval()
        all_targets = []
        all_outputs = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
        if not all_targets or not all_outputs:
            raise ValueError("没有有效的预测结果")
        all_targets = np.array(all_targets)
        all_outputs = np.array(all_outputs)
        all_targets = all_targets * (max_speed - min_speed) + min_speed
        all_outputs = all_outputs * (max_speed - min_speed) + min_speed
        rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))
        if np.isnan(rmse) or np.isinf(rmse):
            raise ValueError(f"无效的RMSE值: {rmse}")
        complexity = sum(p.numel() for p in model.parameters())
        return rmse, complexity
    except Exception as e:
        print(f"\n{'=' * 50}")
        print(f"评估失败 - 个体ID: {id(individual)}")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误消息: {e}")
        print("完整堆栈追踪:")
        traceback.print_exc()
        print(f"{'=' * 50}\n")
        return float('inf'), float('inf')

def evaluate_population(population, train_dataset, val_dataset, min_speed, max_speed, num_features, sequence_length, device):
    performance = []
    complexity = []
    for i, ind in enumerate(population):
        print(f" 评估个体 {i + 1}/{len(population)}...")
        rmse, comp = evaluate_individual(ind, train_dataset, val_dataset, min_speed, max_speed, num_features, sequence_length, device)
        performance.append(rmse)
        complexity.append(comp)
    return np.array(performance), np.array(complexity)

# ==================== 8. NSGA-II核心算法（保持原样） ====================
def fast_non_dominated_sort(performance, complexity):
    pop_size = len(performance)
    fronts = []
    rank = np.zeros(pop_size, dtype=int)
    domination_count = np.zeros(pop_size, dtype=int)
    dominated_solutions = [[] for _ in range(pop_size)]
    valid_mask = np.isfinite(performance) & np.isfinite(complexity)
    for i in range(pop_size):
        if not valid_mask[i]:
            continue
        for j in range(pop_size):
            if i == j or not valid_mask[j]:
                continue
            if (performance[i] <= performance[j] and complexity[i] <= complexity[j]) and \
               (performance[i] < performance[j] or complexity[i] < complexity[j]):
                dominated_solutions[i].append(j)
            elif (performance[j] <= performance[i] and complexity[j] <= complexity[i]) and \
                 (performance[j] < performance[i] or complexity[j] < complexity[i]):
                domination_count[i] += 1
    current_front = np.where((domination_count == 0) & valid_mask)[0]
    if len(current_front) == 0:
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            current_front = valid_indices[:1]
        else:
            current_front = np.array([0])
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
    rank[~valid_mask] = front_idx + 1
    return fronts, rank

def crowding_distance(performance, complexity, fronts):
    pop_size = len(performance)
    distance = np.zeros(pop_size)
    for front in fronts:
        if len(front) <= 2:
            distance[front] = np.inf if len(front) == 2 else 0
            continue
        valid_mask = np.isfinite(performance[front]) & np.isfinite(complexity[front])
        if not valid_mask.any():
            continue
        front_valid = front[valid_mask]
        sorted_perf_idx = np.argsort(performance[front_valid])
        distance[front_valid[sorted_perf_idx[0]]] = np.inf
        distance[front_valid[sorted_perf_idx[-1]]] = np.inf
        sorted_comp_idx = np.argsort(complexity[front_valid])
        distance[front_valid[sorted_comp_idx[0]]] = np.inf
        distance[front_valid[sorted_comp_idx[-1]]] = np.inf
        perf_range = performance[front_valid[sorted_perf_idx[-1]]] - performance[front_valid[sorted_perf_idx[0]]]
        comp_range = complexity[front_valid[sorted_comp_idx[-1]]] - complexity[front_valid[sorted_comp_idx[0]]]
        if perf_range > 0:
            for i in range(1, len(front_valid) - 1):
                idx = sorted_perf_idx[i]
                prev_idx = sorted_perf_idx[i - 1]
                next_idx = sorted_perf_idx[i + 1]
                distance[front_valid[idx]] += (performance[front_valid[next_idx]] - performance[front_valid[prev_idx]]) / perf_range
        if comp_range > 0:
            for i in range(1, len(front_valid) - 1):
                idx = sorted_comp_idx[i]
                prev_idx = sorted_comp_idx[i - 1]
                next_idx = sorted_comp_idx[i + 1]
                distance[front_valid[idx]] += (complexity[front_valid[next_idx]] - complexity[front_valid[prev_idx]]) / comp_range
    return distance

def tournament_selection(population, rank, distance, pop_size):
    mating_pool = []
    pop_size_actual = len(population)
    for _ in range(pop_size):
        idx1, idx2 = np.random.randint(0, pop_size_actual, 2)
        if not np.isfinite(rank[idx1]):
            rank[idx1] = 1e6
        if not np.isfinite(rank[idx2]):
            rank[idx2] = 1e6
        if not np.isfinite(distance[idx1]):
            distance[idx1] = 0
        if not np.isfinite(distance[idx2]):
            distance[idx2] = 0
        if rank[idx1] < rank[idx2] or (rank[idx1] == rank[idx2] and distance[idx1] > distance[idx2]):
            mating_pool.append(population[idx1].copy())
        else:
            mating_pool.append(population[idx2].copy())
    return mating_pool

def environmental_selection(combined_pop, combined_perf, combined_complex, combined_rank, combined_dist, pop_size):
    sorted_idx = np.argsort(combined_rank)
    combined_pop = [combined_pop[i] for i in sorted_idx]
    combined_perf = combined_perf[sorted_idx]
    combined_complex = combined_complex[sorted_idx]
    combined_rank = combined_rank[sorted_idx]
    combined_dist = combined_dist[sorted_idx]
    new_pop = []
    new_perf = []
    new_complex = []
    current_size = 0
    for rank in np.unique(combined_rank):
        if current_size >= pop_size:
            break
        mask = combined_rank == rank
        front_pop = [combined_pop[i] for i in range(len(combined_pop)) if mask[i]]
        front_perf = combined_perf[mask]
        front_complex = combined_complex[mask]
        front_dist = combined_dist[mask]
        if current_size + len(front_pop) <= pop_size:
            new_pop.extend(front_pop)
            new_perf.extend(front_perf)
            new_complex.extend(front_complex)
            current_size += len(front_pop)
        else:
            remaining = pop_size - current_size
            valid_dist = np.isfinite(front_dist)
            if not valid_dist.any():
                selected = np.random.choice(len(front_pop), remaining, replace=False)
            else:
                sorted_dist_idx = np.argsort(front_dist[valid_dist])[::-1]
                selected = np.where(valid_dist)[0][sorted_dist_idx[:remaining]]
            new_pop.extend([front_pop[i] for i in selected])
            new_perf.extend(front_perf[selected])
            new_complex.extend(front_complex[selected])
            current_size = pop_size
    return new_pop, np.array(new_perf), np.array(new_complex)

# ==================== 9. 综合报告生成函数（保持原样） ====================
def generate_comprehensive_report(model, test_loader, val_loader, device, min_speed, max_speed, all_pareto_fronts, feature_columns, topo, cnn_params, lstm_params, setting, test_performance, r_test):
    print('\n========== 生成综合预测报告 ==========')
    model.eval()
    all_val_targets = []
    all_val_outputs = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_val_targets.extend(targets.cpu().numpy())
            all_val_outputs.extend(outputs.cpu().numpy())
    all_val_targets = np.array(all_val_targets).flatten()
    all_val_outputs = np.array(all_val_outputs).flatten()
    all_val_targets = all_val_targets * (max_speed - min_speed) + min_speed
    all_val_outputs = all_val_outputs * (max_speed - min_speed) + min_speed

    all_test_targets = []
    all_test_outputs = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_test_targets.extend(targets.cpu().numpy())
            all_test_outputs.extend(outputs.cpu().numpy())
    all_test_targets = np.array(all_test_targets).flatten()
    all_test_outputs = np.array(all_test_outputs).flatten()
    all_test_targets = all_test_targets * (max_speed - min_speed) + min_speed
    all_test_outputs = all_test_outputs * (max_speed - min_speed) + min_speed

    val_residuals = all_val_targets - all_val_outputs
    test_residuals = all_test_targets - all_test_outputs

    mae_val = mean_absolute_error(all_val_targets, all_val_outputs)
    rmse_val = np.sqrt(mean_squared_error(all_val_targets, all_val_outputs))
    mape_val = np.mean(np.abs(val_residuals / all_val_targets)) * 100
    r2_val = r2_score(all_val_targets, all_val_outputs)
    r_val = pearsonr(all_val_targets, all_val_outputs)[0]

    mae_test = test_performance['mae']
    rmse_test = test_performance['rmse']
    mape_test = test_performance['mape']
    r2_test = r2_score(all_test_targets, all_test_outputs)

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('CNN-BiLSTM 模型预测效果综合报告', fontsize=20, fontweight='bold', y=0.98)

    ax1 = plt.subplot(2, 3, 1)
    sample_idx = np.arange(0, min(500, len(all_val_targets)))
    ax1.plot(sample_idx, all_val_targets[sample_idx], 'b-', linewidth=1.5, label='真实值', alpha=0.8)
    ax1.plot(sample_idx, all_val_outputs[sample_idx], 'r--', linewidth=1.5, label='预测值', alpha=0.8)
    ax1.set_xlabel('时间步', fontsize=12)
    ax1.set_ylabel('风速 (m/s)', fontsize=12)
    ax1.set_title(f'验证集预测效果\n(RMSE={rmse_val:.3f}, MAE={mae_val:.3f})', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    sample_idx = np.arange(0, min(500, len(all_test_targets)))
    ax2.plot(sample_idx, all_test_targets[sample_idx], 'b-', linewidth=1.5, label='真实值', alpha=0.8)
    ax2.plot(sample_idx, all_test_outputs[sample_idx], 'r--', linewidth=1.5, label='预测值', alpha=0.8)
    ax2.set_xlabel('时间步', fontsize=12)
    ax2.set_ylabel('风速 (m/s)', fontsize=12)
    ax2.set_title(f'测试集预测效果\n(RMSE={rmse_test:.3f}, MAE={mae_test:.3f})', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(test_residuals, bins=50, color='darkcyan', alpha=0.7, edgecolor='black')
    ax3.axvline(x=np.mean(test_residuals), color='red', linestyle='--', linewidth=2, label=f'均值: {np.mean(test_residuals):.3f}')
    ax3.set_xlabel('残差 (真实值 - 预测值)', fontsize=12)
    ax3.set_ylabel('频数', fontsize=12)
    ax3.set_title('测试集残差分布', fontsize=14)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(test_residuals, bins=50, density=True, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差线')
    ax4.set_xlabel('预测误差 (m/s)', fontsize=12)
    ax4.set_ylabel('概率密度', fontsize=12)
    ax4.set_title('测试集误差概率密度', fontsize=14)
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    final_pareto = all_pareto_fronts[-1]
    if final_pareto['num_solutions'] > 0:
        perf_vals = final_pareto['performance']
        comp_vals = final_pareto['complexity']
        valid_mask = np.isfinite(perf_vals) & np.isfinite(comp_vals)
        if np.any(valid_mask):
            ax5.scatter(comp_vals[valid_mask], perf_vals[valid_mask], c='darkgreen', s=100, edgecolors='black', linewidth=1, alpha=0.8)
    ax5.set_xlabel('模型复杂度 (参数数量)', fontsize=12)
    ax5.set_ylabel('验证集 RMSE (m/s)', fontsize=12)
    ax5.set_title('最后一代Pareto前沿', fontsize=14)
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)
    metrics = ['RMSE', 'MAE', 'MAPE', 'R²']
    test_metrics = [rmse_test, mae_test, mape_test, r2_test]
    x = np.arange(len(metrics))
    width = 0.35
    bars2 = ax6.bar(x + width / 2, test_metrics, width, label='测试集', color='darkcyan', alpha=0.8)
    ax6.set_xlabel('评估指标', fontsize=12)
    ax6.set_ylabel('指标值', fontsize=12)
    ax6.set_title('性能指标对比', fontsize=14)
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)

    def add_value_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    add_value_labels(ax6, bars2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('comprehensive_cnn_bilstm_report.png', dpi=300, bbox_inches='tight')
    plt.show()

    if isinstance(model, nn.DataParallel):
        feature_weights = model.module.feature_weights.detach().cpu().numpy()
    else:
        feature_weights = model.feature_weights.detach().cpu().numpy()
    plt.figure(figsize=(10, 6))
    weights = feature_weights.flatten()
    plt.bar(range(len(feature_columns)), weights, color=[0.2, 0.5, 0.8])
    plt.xticks(range(len(feature_columns)), feature_columns, rotation=15)
    plt.xlabel('输入特征')
    plt.ylabel('学习到的权重（越大越重要）')
    plt.title('HybridCNNBiLSTM 中各输入特征的权重')
    plt.grid(True, axis='y')
    for i, w in enumerate(weights):
        plt.text(i, w + 0.02, f'{w:.4f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('feature_weights.png', dpi=300)
    plt.show()

    print('\n========== 详细性能报告 ==========')
    print(f'验证集性能:')
    print(f' MAE: {mae_val:.4f} m/s')
    print(f' RMSE: {rmse_val:.4f} m/s')
    print(f' MAPE: {mape_val:.2f}%')
    print(f' R²: {r2_val:.4f}')
    print(f' 相关系数: {r_val:.4f}')
    print(f'\n测试集性能:')
    print(f' MAE: {mae_test:.4f} m/s')
    print(f' RMSE: {rmse_test:.4f} m/s')
    print(f' MAPE: {mape_test:.2f}%')
    print(f' R²: {r2_test:.4f}')
    print(f' 相关系数: {r_test:.4f}')
    print(f'\n特征重要性排名:')
    feature_importance = sorted(zip(feature_columns, weights), key=lambda x: x[1], reverse=True)
    for i, (feature, weight) in enumerate(feature_importance, 1):
        print(f' {i}. {feature}: {weight:.4f}')
    print('\n综合报告已生成并保存！')
    return {
        'val_metrics': {'mae': mae_val, 'rmse': rmse_val, 'mape': mape_val, 'r2': r2_val, 'r': r_val},
        'test_metrics': {'mae': mae_test, 'rmse': rmse_test, 'mape': mape_test, 'r2': r2_test, 'r': r_test}
    }

# ==================== 10. 主程序（已应用所有修复） ====================
if __name__ == '__main__':
    POP_SIZE = 2
    MAX_GEN = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print(f'检测到 {torch.cuda.device_count()} 块GPU,将使用多GPU并行')
    print(f'使用设备: {device}')
    print('\n========== 开始NSGA-II优化MODEO-CNN ==========')
    print('基于PyTorch实现,无TensorFlow依赖')
    print('【改进】学习率采用连续优化,范围:[0.0001, 0.01]')
    start_time = time.time()

    population = initialize_population(POP_SIZE)
    print(f'初始化完成,种群大小: {len(population)}')

    best_rmse_history = []
    best_complexity_history = []
    all_pareto_fronts = []
    mutation_prob = 0.5

    for gen in range(MAX_GEN):
        print(f'\n第 {gen + 1}/{MAX_GEN} 代...')
        performance, complexity = evaluate_population(
            population, train_dataset, val_dataset, min_speed, max_speed,
            len(feature_columns), sequence_length, device
        )
        fronts, rank = fast_non_dominated_sort(performance, complexity)
        distance = crowding_distance(performance, complexity, fronts)
        mating_pool = tournament_selection(population, rank, distance, POP_SIZE)
        offspring = crossover_population(mating_pool)

        mutated_offspring = []
        for child in offspring:
            if np.random.random() < mutation_prob:
                mutated_child = variable_length_mutation(child)
                mutated_offspring.append(mutated_child)
            else:
                mutated_offspring.append(child.copy())

        offspring_perf, offspring_comp = evaluate_population(
            mutated_offspring, train_dataset, val_dataset, min_speed, max_speed,
            len(feature_columns), sequence_length, device
        )

        combined_pop = population + mutated_offspring
        combined_perf = np.concatenate([performance, offspring_perf])
        combined_comp = np.concatenate([complexity, offspring_comp])
        combined_fronts, combined_rank = fast_non_dominated_sort(combined_perf, combined_comp)
        combined_dist = crowding_distance(combined_perf, combined_comp, combined_fronts)

        population, performance, complexity = environmental_selection(
            combined_pop, combined_perf, combined_comp, combined_rank, combined_dist, POP_SIZE
        )

        pareto_front = {
            'params': [combined_pop[i] for i in combined_fronts[0]],
            'performance': combined_perf[combined_fronts[0]],
            'complexity': combined_comp[combined_fronts[0]],
            'num_solutions': len(combined_fronts[0]),
            'generation': gen + 1
        }
        all_pareto_fronts.append(pareto_front)

        valid_combined_perf = combined_perf[np.isfinite(combined_perf)]
        if len(valid_combined_perf) > 0:
            best_idx_combined = np.argmin(valid_combined_perf)
            best_rmse_history.append(valid_combined_perf[best_idx_combined])
            best_complexity_history.append(combined_comp[best_idx_combined])
        else:
            best_rmse_history.append(float('inf'))
            best_complexity_history.append(float('inf'))

        if gen + 1 < MAX_GEN:
            print(f' 合并种群Pareto解数量: {len(combined_fronts[0])}, 最优RMSE: {best_rmse_history[-1]:.4f}')

    end_time = time.time()
    print(f'\n优化完成!总耗时: {end_time - start_time:.2f} 秒 ({(end_time - start_time) / 60:.2f} 分钟)')

    # 选择最终最优个体
    final_pareto = all_pareto_fronts[-1]
    if final_pareto['num_solutions'] > 0:
        perf_vals = final_pareto['performance']
        comp_vals = final_pareto['complexity']
        valid_mask = np.isfinite(perf_vals) & np.isfinite(comp_vals)
        if valid_mask.any():
            perf_vals = perf_vals[valid_mask]
            comp_vals = comp_vals[valid_mask]
            params = [final_pareto['params'][i] for i in range(len(valid_mask)) if valid_mask[i]]
            normalized_perf = (perf_vals - perf_vals.min()) / (perf_vals.max() - perf_vals.min() + 1e-10)
            normalized_comp = (comp_vals - comp_vals.min()) / (comp_vals.max() - comp_vals.min() + 1e-10)
            trade_off_scores = np.sqrt(normalized_perf ** 2 + normalized_comp ** 2)
            best_idx = np.argmin(trade_off_scores)
            best_individual = params[best_idx]
        else:
            raise ValueError("最后一代没有有效解")
    else:
        all_perf = np.concatenate([pf['performance'] for pf in all_pareto_fronts])
        all_params = [p for pf in all_pareto_fronts for p in pf['params']]
        valid_mask = np.isfinite(all_perf)
        if valid_mask.any():
            best_idx = np.argmin(all_perf[valid_mask])
            best_individual = np.array(all_params)[valid_mask][best_idx]
        else:
            raise ValueError("所有个体评估均失败!")

    # 解码最优个体
    topo, cnn_params, lstm_params, setting = decode_individual(best_individual)
    batch_size, learn_rate, opt_type, reg_type = decode_hyperparams(setting)
    # ====================== 关键修复：最终训练也强制 batch_size >=2 ======================
    batch_size = max(2, batch_size)

    print(f'\n最终优化的超参数数值:')
    print(f' batch_size: {batch_size}')
    print(f' learning_rate: {learn_rate:.6f}')
    print(f' optimizer: {opt_type}')
    print(f' regularizer: {reg_type}')
    print(f'\n模型拓扑结构:')
    print(f' topo: {topo}')
    print(f' cnn_params: {cnn_params}')
    print(f' lstm_params: {lstm_params}')

    print(f'\n最后一代Pareto前沿的每个解:')
    for i in range(final_pareto['num_solutions']):
        ind = final_pareto['params'][i]
        perf = final_pareto['performance'][i]
        comp = final_pareto['complexity'][i]
        topo_i, cnn_params_i, lstm_params_i, setting_i = decode_individual(ind)
        batch_size_i, learn_rate_i, opt_type_i, reg_type_i = decode_hyperparams(setting_i)
        print(f' 解 {i + 1}:')
        print(f' Pareto坐标: RMSE = {perf:.4f}, Complexity = {comp}')
        print(f' 拓扑结构: {topo_i}')
        print(f' CNN参数: {cnn_params_i}')
        print(f' LSTM参数: {lstm_params_i}')
        print(f' 设置参数: batch_size={batch_size_i}, learning_rate={learn_rate_i:.6f}, optimizer={opt_type_i}, regularizer={reg_type_i}')

    # 训练最终模型（已修复）
    print('\n训练最终模型...')
    model = HybridCNNBiLSTM(topo, cnn_params, lstm_params, len(feature_columns), sequence_length).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = nn.MSELoss()
    if opt_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    elif opt_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)
    elif opt_type == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learn_rate)
    else:
        optimizer = optim.Adadelta(model.parameters(), lr=learn_rate)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    for epoch in range(200):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0
        val_samples = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets).item() * inputs.size(0)
                val_loss += batch_loss
                val_samples += inputs.size(0)
        if val_samples > 0:
            val_loss /= val_samples
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss)
        if (epoch + 1) % 5 == 0:
            print(f' Epoch {epoch + 1}/200, 训练损失: {train_loss / len(train_loader):.6f}, 验证损失: {val_loss:.6f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f'早停于 epoch {epoch + 1}')
            break

    # 绘制训练损失曲线
    print('\n正在绘制训练损失曲线...')
    plt.figure(figsize=(12, 8))
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, 'b-', linewidth=2, label='训练损失', alpha=0.8)
    plt.plot(epochs_range, val_losses, 'r--', linewidth=2, label='验证损失', alpha=0.8)
    plt.xlabel('训练轮次 (Epoch)', fontsize=14)
    plt.ylabel('损失值 (MSE)', fontsize=14)
    plt.title('模型训练过程损失曲线', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 评估最终模型
    model.eval()
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
    all_test_targets = all_test_targets.flatten()
    all_test_outputs = all_test_outputs.flatten()

    mae_test = mean_absolute_error(all_test_targets, all_test_outputs)
    rmse_test = np.sqrt(mean_squared_error(all_test_targets, all_test_outputs))
    mape_test = np.mean(np.abs((all_test_targets - all_test_outputs) / all_test_targets)) * 100
    r2_test = r2_score(all_test_targets, all_test_outputs)
    r_test = pearsonr(all_test_targets, all_test_outputs)[0]

    print(f'\n测试集最终性能:')
    print(f' MAE: {mae_test:.4f} m/s')
    print(f' RMSE: {rmse_test:.4f} m/s')
    print(f' MAPE: {mape_test:.2f}%')
    print(f' R²: {r2_test:.4f}')
    print(f' 相关系数: {r_test:.4f}')

    test_performance = {'mae': mae_test, 'rmse': rmse_test, 'mape': mape_test}

    # Pareto前沿演化图
    print('\n正在绘制 Pareto 前沿演化图...')
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_pareto_fronts)))
    for idx, pf in enumerate(all_pareto_fronts):
        if pf['num_solutions'] > 0:
            perf = pf['performance']
            comp = pf['complexity']
            valid_mask = np.isfinite(perf) & np.isfinite(comp)
            if np.any(valid_mask):
                plt.scatter(comp[valid_mask], perf[valid_mask], c=[colors[idx]], s=60, alpha=0.6, label=f'第 {pf["generation"]} 代', edgecolors='k', linewidth=0.5)
    plt.xlabel('模型复杂度 (参数数量)', fontsize=14)
    plt.ylabel('验证集 RMSE (m/s)', fontsize=14)
    plt.title('NSGA-II Pareto 前沿演化过程', fontsize=16)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pareto_evolution_continuous_lr.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Pareto前沿对比图
    print('\n正在绘制关键代数Pareto前沿对比图（带连接线）...')
    target_generations = [1, 5, 10, 20, 25, 30]
    plt.figure(figsize=(14, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    plotted_any = False
    for idx, target_gen in enumerate(target_generations):
        pf = None
        for front in all_pareto_fronts:
            if front['generation'] == target_gen:
                pf = front
                break
        if pf is None or pf['num_solutions'] == 0:
            print(f' 警告: 第 {target_gen} 代数据不存在或无有效解')
            continue
        perf = pf['performance']
        comp = pf['complexity']
        valid_mask = np.isfinite(perf) & np.isfinite(comp)
        if not np.any(valid_mask):
            print(f' 警告: 第 {target_gen} 代无有效数据')
            continue
        perf = perf[valid_mask]
        comp = comp[valid_mask]
        sort_idx = np.argsort(comp)
        comp_sorted = comp[sort_idx]
        perf_sorted = perf[sort_idx]
        plt.plot(comp_sorted, perf_sorted, color=colors[idx], linewidth=2, alpha=0.7, label=f'第 {target_gen} 代 (n={len(perf)})')
        plt.scatter(comp, perf, c=colors[idx], marker=markers[idx], s=80, alpha=0.9, edgecolors='black', linewidth=0.5)
        plotted_any = True
    if not plotted_any:
        print("警告：没有找到任何指定代数的有效Pareto前沿数据！")
    else:
        plt.xlabel('模型复杂度 (参数数量)', fontsize=14)
        plt.ylabel('验证集 RMSE (m/s)', fontsize=14)
        plt.title('NSGA-II Pareto前沿进化过程对比（带连接线）', fontsize=16, fontweight='bold')
        plt.legend(fontsize=11, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.figtext(0.5, 0.02, '注：第1代为初始种群，第50代为最终进化结果\n每条实线连接该代的所有Pareto最优解，展示前沿面形状', ha='center', fontsize=10, style='italic')
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        evolution_comparison_path = 'pareto_evolution_comparison_connected.png'
        plt.savefig(evolution_comparison_path, dpi=300, bbox_inches='tight')
        print(f'带连接线的Pareto前沿对比图已保存：{evolution_comparison_path}')
    plt.show()

    # 最终代Pareto前沿详细图（已修正为第50代）
    print('\n正在绘制最终代Pareto前沿详细图（第50代）...')
    final_pf = None
    for pf in all_pareto_fronts:
        if pf['generation'] == 50:
            final_pf = pf
            break
    if final_pf and final_pf['num_solutions'] > 0:
        perf_final = final_pf['performance']
        comp_final = final_pf['complexity']
        valid_final = np.isfinite(perf_final) & np.isfinite(comp_final)
        if np.any(valid_final):
            perf_final = perf_final[valid_final]
            comp_final = comp_final[valid_final]
            sort_idx = np.argsort(comp_final)
            comp_sorted = comp_final[sort_idx]
            perf_sorted = perf_final[sort_idx]
            plt.figure(figsize=(12, 8))
            plt.plot(comp_sorted, perf_sorted, color='darkred', linewidth=3, alpha=0.8, marker='o', markersize=10,
                     markerfacecolor='red', markeredgecolor='black', markeredgewidth=1.5)
            plt.xlabel('模型复杂度 (参数数量)', fontsize=14)
            plt.ylabel('验证集 RMSE (m/s)', fontsize=14)
            plt.title('最终Pareto前沿（第50代）', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)
            for i, (comp_val, perf_val) in enumerate(zip(comp_sorted, perf_sorted)):
                plt.annotate(f'({comp_val:.0f}, {perf_val:.3f})', xy=(comp_val, perf_val),
                             xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)
            plt.tight_layout()
            plt.savefig('pareto_front_final_gen50.png', dpi=300, bbox_inches='tight')
            plt.show()

    # 保存结果
    with open('modeo_cnn_optimization_continuous_lr.pkl', 'wb') as f:
        pickle.dump({
            'best_individual': best_individual,
            'best_rmse_history': best_rmse_history,
            'pareto_fronts': all_pareto_fronts,
            'test_performance': test_performance,
            'training_losses': train_losses,
            'validation_losses': val_losses,
            'model_params': {
                'topo': topo,
                'cnn_params': cnn_params,
                'lstm_params': lstm_params,
                'setting': setting,
                'batch_size': batch_size,
                'learn_rate': learn_rate,
                'opt_type': opt_type
            }
        }, f)

    # 生成综合报告
    report_metrics = generate_comprehensive_report(
        model, test_loader, val_loader, device, min_speed, max_speed, all_pareto_fronts,
        feature_columns, topo, cnn_params, lstm_params, setting, test_performance, r_test
    )

    print('\n========== 优化完成!所有结果已保存 ==========')
