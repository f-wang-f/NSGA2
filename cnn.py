import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt



 #==================== 1. 数据读取与预处理 ====================
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

# ==================== 2. Feature Engineering and Sequence Creation ====================
def create_sequences(data, feature_columns, target_column, seq_length=8):
    """
    Create time series samples
    Input: (seq_length, 4) -> Output: next time step wind speed
    """
    features = data[feature_columns].values
    target = data[target_column].values

    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(target[i + seq_length])

    return np.array(X), np.array(y)


# Create sequence data
X, y = create_sequences(data, feature_columns, target_column, seq_length=8)
print(f"Sequence data shapes - X: {X.shape}, y: {y.shape}")
# Data standardization (critical for neural networks)
# Standardize features
X_reshaped = X.reshape(-1, X.shape[-1])
x_mean = np.mean(X_reshaped, axis=0)
x_std = np.std(X_reshaped, axis=0)
X_scaled = (X_reshaped - x_mean) / (x_std + 1e-8)  # Avoid division by zero
X = X_scaled.reshape(X.shape[0], X.shape[1], X.shape[2])
# Standardize target variable
y_reshaped = y.reshape(-1, 1)
y_mean = np.mean(y_reshaped, axis=0)[0]
y_std = np.std(y_reshaped, axis=0)[0]
y_scaled = ((y_reshaped - y_mean) / (y_std + 1e-8)).flatten()
# ==================== 3. Dataset Splitting ====================
# Split into train-val and test sets (9:1), with 10% as test set
n_total = len(X)
n_train_val = int(n_total * 0.9)
X_train_val, X_test = X[:n_train_val], X[n_train_val:]
y_train_val, y_test = y_scaled[:n_train_val], y_scaled[n_train_val:]
# Further split train-val into train and val
n_train = int(len(X_train_val) * 0.9)
X_train, X_val = X_train_val[:n_train], X_train_val[n_train:]
y_train, y_val = y_train_val[:n_train], y_train_val[n_train:]
print(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")


# ==================== 4. PyTorch Dataset and DataLoader ====================
class WindSpeedDataset(Dataset):
    def __init__(self, X, y):
        # X: (samples, seq_len, features)
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return (seq_len, features) and target
        return self.X[idx], self.y[idx]


train_dataset = WindSpeedDataset(X_train, y_train)
val_dataset = WindSpeedDataset(X_val, y_val)
test_dataset = WindSpeedDataset(X_test, y_test)
# Create DataLoaders
batch_size = 32  # Batch size (not specified in paper, using common value)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ==================== 5. Build CNN Model (Strictly Follow Paper Architecture) ====================
class WindSpeedCNN(nn.Module):
    def __init__(self, input_channels=4, seq_length=8, dropout_rate=0.2):
        super(WindSpeedCNN, self).__init__()

        self.dropout_rate = dropout_rate

        # 1D-CNN: 32 kernels, size 3, input channels=4
        self.conv1d = nn.Conv1d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=3,
            padding=1  # Keep output length unchanged
        )
        self.relu = nn.ReLU()

        # 2D-CNN first layer: 32 3x3 kernels
        self.conv2d_1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.pool2d_1 = nn.MaxPool2d(kernel_size=2)  # Halve size

        # 2D-CNN second layer: 64 3x3 kernels
        self.conv2d_2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.pool2d_2 = nn.MaxPool2d(kernel_size=2)  # Halve size again

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Calculate flattened dimension: (batch, 64, seq_length//4, 32//4)
        # seq_length=8 -> 8//4=2, 32//4=8
        flatten_dim = 64 * (seq_length // 4) * 8

        # Fully connected layer: 100 hidden units
        self.fc1 = nn.Linear(flatten_dim, 100)

        # Output layer: 1 unit
        self.fc_out = nn.Linear(100, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Custom weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass
        x: (batch, seq_len, features) = (batch, 8, 4)
        """
        batch_size = x.size(0)

        # Transpose for PyTorch Conv1d: (batch, features, seq_len)
        x = x.transpose(1, 2)  # -> (batch, 4, 8)

        # 1D-CNN feature extraction
        x = self.conv1d(x)  # -> (batch, 32, 8)
        x = self.relu(x)

        # Reshape to 2D feature map: (batch, 32, 8) -> (batch, 1, 8, 32)
        x = x.permute(0, 2, 1).unsqueeze(-1)  # -> (batch, 8, 32, 1)
        x = x.permute(0, 3, 1, 2)  # -> (batch, 1, 8, 32)

        # 2D-CNN first layer
        x = self.conv2d_1(x)  # -> (batch, 32, 8, 32)
        x = self.relu(x)
        x = self.pool2d_1(x)  # -> (batch, 32, 4, 16)

        # 2D-CNN second layer
        x = self.conv2d_2(x)  # -> (batch, 64, 4, 16)
        x = self.relu(x)
        x = self.pool2d_2(x)  # -> (batch, 64, 2, 8)

        # Flatten
        x = x.view(batch_size, -1)  # -> (batch, 64*2*8=1024)

        # Dropout + Fully connected layers
        x = self.dropout(x)
        x = self.fc1(x)  # -> (batch, 100)
        x = self.relu(x)
        x = self.dropout(x)

        # Output layer
        x = self.fc_out(x)  # -> (batch, 1)

        return x.squeeze()


# Instantiate model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = WindSpeedCNN(input_channels=4, seq_length=8).to(device)
# ==================== 6. Training Parameters ====================
learning_rate = 0.03  # Paper value
l2_reg = 0.0005  # Paper L2 regularization coefficient λ1=λ2=0.0005
dropout_rate = 0.2  # Paper dropout threshold
# Adam optimizer (L2 regularization via weight_decay)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
# Loss function: Mean Squared Error (MSE)
criterion = nn.MSELoss()
# ==================== 7. Model Training Loop ====================
num_epochs = 200  # Paper iteration count
patience = 20  # Early stopping patience
best_val_loss = float('inf')
patience_counter = 0
# Record training history
train_losses = []
val_losses = []
train_maes = []
val_maes = []
print("\nStarting model training...")
print("=" * 60)
for epoch in range(num_epochs):
    # Training mode
    model.train()
    train_loss = 0.0
    train_mae = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate losses
        train_loss += loss.item() * X_batch.size(0)
        train_mae += torch.abs(outputs - y_batch).sum().item()

    # Validation mode
    model.eval()
    val_loss = 0.0
    val_mae = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            val_loss += loss.item() * X_batch.size(0)
            val_mae += torch.abs(outputs - y_batch).sum().item()

    # Calculate average metrics
    train_loss /= len(train_dataset)
    val_loss /= len(val_dataset)
    train_mae /= len(train_dataset)
    val_mae /= len(val_dataset)

    # Record history
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_maes.append(train_mae)
    val_maes.append(val_mae)

    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1:03d}/{num_epochs:03d}] | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"Train MAE: {train_mae:.4f} m/s | Val MAE: {val_mae:.4f} m/s")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_wind_speed_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
print("Model training completed!")
print("=" * 60)
# ==================== 8. Load Best Model for Evaluation ====================
model.load_state_dict(torch.load('best_wind_speed_model.pth'))
model.eval()
# Predict on test set
all_preds = []
all_targets = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)

        all_preds.append(outputs.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())
y_pred_scaled = np.concatenate(all_preds)
y_test_scaled = np.concatenate(all_targets)
# Inverse standardize to original scale
y_pred = (y_pred_scaled * y_std + y_mean)
y_test_original = (y_test_scaled * y_std + y_mean)
# ==================== 9. Calculate Evaluation Metrics ====================
mae = np.mean(np.abs(y_test_original - y_pred))
rmse = np.sqrt(np.mean((y_test_original - y_pred) ** 2))
mape = np.mean(np.abs((y_test_original - y_pred) / (y_test_original + 1e-8))) * 100
ss_res = np.sum((y_test_original - y_pred) ** 2)
ss_tot = np.sum((y_test_original - np.mean(y_test_original)) ** 2)
r2 = 1 - (ss_res / ss_tot)
print("\n" + "=" * 60)
print("Model Performance Evaluation Results:")
print(f"MAE (Mean Absolute Error): {mae:.4f} m/s")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f} m/s")
print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
print(f"R² (Coefficient of Determination): {r2:.4f}")
print("=" * 60)
# ==================== 10. Results Visualization ====================
fig, axes = plt.subplots(2, 1, figsize=(15, 10))
# Prediction curve comparison (first 200 samples)
axes[0].plot(y_test_original[:200], label='Actual Values', color='blue', linewidth=2)
axes[0].plot(y_pred[:200], label='Predicted Values', color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Time Steps', fontsize=12)
axes[0].set_ylabel('Wind Speed (m/s)', fontsize=12)
axes[0].set_title('CNN Model Wind Speed Prediction Results', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
# Scatter plot
axes[1].scatter(y_test_original, y_pred, alpha=0.5, s=20, color='steelblue')
axes[1].plot([y_test_original.min(), y_test_original.max()],
             [y_test_original.min(), y_test_original.max()],
             'r--', lw=2, label='Ideal Line')
axes[1].set_xlabel('Actual Wind Speed (m/s)', fontsize=12)
axes[1].set_ylabel('Predicted Wind Speed (m/s)', fontsize=12)
axes[1].set_title('Predicted vs Actual Values Scatter Plot', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('wind_speed_prediction_results.png', dpi=300, bbox_inches='tight')
plt.show()
# Training history curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(train_losses, label='Train Loss', linewidth=2, color='darkblue')
axes[0].plot(val_losses, label='Val Loss', linewidth=2, color='darkorange')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('MSE', fontsize=12)
axes[0].set_title('Train and Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[1].plot(train_maes, label='Train MAE', linewidth=2, color='darkblue')
axes[1].plot(val_maes, label='Val MAE', linewidth=2, color='darkorange')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('MAE (m/s)', fontsize=12)
axes[1].set_title('Train and Validation MAE', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()
# Error analysis
absolute_errors = np.abs(y_test_original - y_pred)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].hist(absolute_errors, bins=50, edgecolor='black', alpha=0.7, color='teal')
axes[0].set_xlabel('Absolute Error (m/s)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Absolute Error Distribution', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[1].plot(absolute_errors[:200], color='purple', linewidth=1.5)
axes[1].set_xlabel('Time Steps', fontsize=12)
axes[1].set_ylabel('Absolute Error (m/s)', fontsize=12)
axes[1].set_title('Absolute Error Over Time', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nError Statistics:")
print(f"Mean Absolute Error: {np.mean(absolute_errors):.4f} m/s")
print(f"Error Standard Deviation: {np.std(absolute_errors):.4f} m/s")
print(f"Maximum Error: {np.max(absolute_errors):.4f} m/s")
print(f"Minimum Error: {np.min(absolute_errors):.4f} m/s")
# ==================== 11. Comparison with Persistence Baseline Model ====================
print("\n" + "=" * 60)
print("Comparison with Persistence Baseline Model:")
# Persistence model: Use current value to predict next value
persistence_pred = y_test_original[:-1]
persistence_true = y_test_original[1:]
persistence_mae = np.mean(np.abs(persistence_true - persistence_pred))
persistence_rmse = np.sqrt(np.mean((persistence_true - persistence_pred) ** 2))
persistence_mape = np.mean(np.abs((persistence_true - persistence_pred) / (persistence_true + 1e-8))) * 100
persistence_ss_res = np.sum((persistence_true - persistence_pred) ** 2)
persistence_ss_tot = np.sum((persistence_true - np.mean(persistence_true)) ** 2)
persistence_r2 = 1 - (persistence_ss_res / persistence_ss_tot)
print(f"Persistence MAE: {persistence_mae:.4f} m/s")
print(f"Persistence RMSE: {persistence_rmse:.4f} m/s")
print(f"Persistence MAPE: {persistence_mape:.2f}%")
print(f"Persistence R²: {persistence_r2:.4f}")
improvement = (persistence_mae - mae) / persistence_mae * 100
print(f"CNN Model Relative Improvement: {improvement:.2f}%")
print("=" * 60)
# ==================== 12. Model Saving ====================
torch.save({
    'model_state_dict': model.state_dict(),
    'x_mean': x_mean,
    'x_std': x_std,
    'y_mean': y_mean,
    'y_std': y_std,
    'feature_columns': feature_columns,
    'target_column': target_column
}, 'wind_speed_cnn_model.pth')
print("\nComplete model saved to wind_speed_cnn_model.pth")