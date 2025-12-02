import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import warnings

# Configuration
warnings.filterwarnings('ignore')
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters
SEQ_LEN = 24  # Input sequence length (e.g., past 24 time steps)
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
HIDDEN_SIZE = 64  # LSTM hidden size
NUM_LAYERS = 2  # LSTM layers
DROPOUT = 0.3
CNN_FILTERS = 64
KERNEL_SIZE = 3


# ==================== 1. Utilities (Replaces sklearn) ====================
class CustomMinMaxScaler:
    """
    A lightweight implementation of MinMaxScaler to avoid sklearn dependencies
    and binary incompatibility issues.
    """

    def __init__(self):
        self.min_ = None
        self.scale_ = None

    def fit_transform(self, data):
        """Compute the min and max to be used for later scaling and scale the data."""
        self.min_ = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        self.scale_ = max_val - self.min_

        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0

        return (data - self.min_) / self.scale_

    def inverse_transform(self, data):
        """Undo the scaling of X according to feature_range."""
        return data * self.scale_ + self.min_


def calculate_metrics(y_true, y_pred):
    """
    Calculate MAE, RMSE, R2 score, and MAPE manually using NumPy.
    """
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))

    # Root Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    # R2 Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # Avoid division by zero for R2
    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = 1 - (ss_res / ss_tot)

    # Mean Absolute Percentage Error (handling division by zero)
    nonzero_mask = y_true != 0
    if np.any(nonzero_mask):
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = np.nan  # or 0, depending on preference

    return mae, rmse, r2, mape


# ==================== 2. Data Processing ====================
class WindDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(data, target, seq_len):
    """Creates sliding window sequences from the data."""
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)]
        y = target[i + seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def load_and_process_data(filename, seq_len):
    print(f'Reading data from {filename}...')
    try:
        # Check if file exists to prevent crash if running without data
        if not os.path.exists(filename):
            print(f"File {filename} not found. Generating dummy data for demonstration.")
            # Generate dummy data matching your columns
            dates = pd.date_range(start='2023-01-01', periods=1000, freq='15T')
            data = pd.DataFrame({
                'Wind Direction': np.random.uniform(0, 360, 1000),
                'Theoretical_Power_Curve (KWh)': np.random.uniform(0, 3000, 1000),
                'LV ActivePower (kW)': np.random.uniform(0, 3000, 1000),
                'Wind Speed (m/s)': np.random.uniform(0, 25, 1000)
            })
        else:
            data = pd.read_excel(filename)
        feature_columns = ['Wind Direction', 'Theoretical_Power_Curve (KWh)', 'LV ActivePower (kW)', 'Wind Speed (m/s)']
        target_column = 'Wind Speed (m/s)'

        # Verify columns
        for col in feature_columns:
            if col not in data.columns:
                raise ValueError(f'Feature column not found: {col}')

        # Normalization using Custom Scaler
        scaler_X = CustomMinMaxScaler()
        scaler_y = CustomMinMaxScaler()

        feature_data = data[feature_columns].values
        target_data = data[target_column].values.reshape(-1, 1)

        feature_data_scaled = scaler_X.fit_transform(feature_data)
        target_data_scaled = scaler_y.fit_transform(target_data)

        # Create Sequences
        X, y = create_sequences(feature_data_scaled, target_data_scaled, seq_len)

        # Train/Test Split (80% Train, 20% Test)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return X_train, y_train, X_test, y_test, scaler_y, len(feature_columns)

    except Exception as e:
        print(f'Error reading file: {e}')
        raise


# ==================== 3. Model Definition (CNN-BiLSTM-AM) ====================
class Attention(nn.Module):
    """
    Attention Mechanism as described in the paper.
    Weights the importance of different time steps in the LSTM output.
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        # Equation 18: s_t = tanh(W_h * h_t + b_h)
        self.dense = nn.Linear(hidden_size * 2, hidden_size * 2)  # *2 for bidirectional
        self.tanh = nn.Tanh()
        # Equation 19: Attention weights
        self.v = nn.Parameter(torch.rand(hidden_size * 2))

    def forward(self, hidden_states):
        # hidden_states shape: [batch, seq_len, hidden_size * 2]

        # Score calculation
        energy = self.tanh(self.dense(hidden_states))
        energy = energy.transpose(2, 1)  # [batch, hidden*2, seq_len]

        # Calculate weights using softmax (Equation 19)
        # We compute the product with v vector by doing matrix multiplication
        v = self.v.repeat(hidden_states.size(0), 1).unsqueeze(1)  # [batch, 1, hidden*2]
        weights = torch.bmm(v, energy).squeeze(1)  # [batch, seq_len]
        weights = torch.softmax(weights, dim=1)

        # Context vector (Equation 20): weighted sum of hidden states
        # [batch, 1, seq_len] * [batch, seq_len, hidden*2] -> [batch, 1, hidden*2]
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)

        return context


class CNN_BiLSTM_AM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim=1, dropout=0.3):
        super(CNN_BiLSTM_AM, self).__init__()

        # 1. CNN Layer (Extract spatial/local features)
        # Input to Conv1d: (Batch, Channels/Features, Seq_Len)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=CNN_FILTERS, kernel_size=KERNEL_SIZE, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # 2. BiLSTM Layer (Extract temporal features)
        # Input to LSTM: (Batch, Seq_Len, Features)
        self.lstm = nn.LSTM(
            input_size=CNN_FILTERS,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 3. Attention Mechanism
        self.attention = Attention(hidden_size)

        # 4. Dropout & Fully Connected
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_dim)  # *2 for bidirectional

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]

        # CNN expects [batch, channels, seq_len]
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Permute back for LSTM: [batch, seq_len_new, filters]
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention (Pools the time dimension)
        attn_out = self.attention(lstm_out)

        # Final layers
        out = self.dropout(attn_out)
        out = self.fc(out)

        return out


# ==================== 4. Training & Evaluation ====================
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_model(model, test_loader, scaler):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            predictions.append(output.cpu().numpy())
            actuals.append(y_batch.numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    # Inverse transform to original scale
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)

    return actuals, predictions


def visualize_results(train_losses, y_true, y_pred):
    """
    绘制训练损失、预测对比图、散点图和误差分布图。
    """
    # 1. 训练损失图 (Training Loss)
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Training Loss', color='navy')
    plt.title('Training Loss over Epochs (训练损失)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    # 2. 时间序列预测对比图 (Time Series Comparison)
    plt.figure(figsize=(15, 6))
    subset_len = min(300, len(y_true))
    plt.plot(y_true[:subset_len], label='真实值 (Actual)', color='blue')
    plt.plot(y_pred[:subset_len], label='预测值 (Prediction)', color='red', linestyle='--')
    plt.title('Wind Speed Prediction: Actual vs Predicted (局部对比)')
    plt.xlabel('Time Step')
    plt.ylabel('Wind Speed (m/s)')
    plt.legend()
    plt.grid(True)
    plt.show()
    # 3. 散点图 (Scatter Plot)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, color='green', s=10)
    # 绘制对角线 y=x
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal')
    plt.title('Scatter Plot: True vs Predicted (散点图)')
    plt.xlabel('True Values (m/s)')
    plt.ylabel('Predicted Values (m/s)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    # 4. 误差分布直方图 (Error Distribution)
    errors = y_true - y_pred
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=50, color='purple', alpha=0.7, edgecolor='black')
    plt.title('Prediction Error Distribution (误差分布)')
    plt.xlabel('Error (True - Predicted)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def main():
    # 1. Load Data
    filename = 'winddata.xlsx'
    X_train, y_train, X_test, y_test, scaler, num_features = load_and_process_data(filename, SEQ_LEN)

    # Create DataLoaders
    train_dataset = WindDataset(X_train, y_train)
    test_dataset = WindDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

    # 2. Initialize Model
    model = CNN_BiLSTM_AM(
        input_dim=num_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    print("Starting training...")
    train_losses = []

    for epoch in range(EPOCHS):
        loss = train_model(model, train_loader, criterion, optimizer)
        train_losses.append(loss)
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss:.6f}')

    # 4. Evaluation
    print("Evaluating model...")
    y_true, y_pred = evaluate_model(model, test_loader, scaler)

    # Calculate Metrics (Manually, no sklearn)
    mae, rmse, r2, mape = calculate_metrics(y_true, y_pred)

    print("=" * 30)
    print(f"Model Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"MAPE: {mape:.4f}%")
    print("=" * 30)

    # Visualization
    print("Generating visualizations...")
    visualize_results(train_losses, y_true, y_pred)

    # Save the model
    torch.save(model.state_dict(), 'cnn_bilstm_am_model.pth')
    print("Model saved to cnn_bilstm_am_model.pth")


if __name__ == '__main__':
    main()