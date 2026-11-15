import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import json
import time
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class PermeabilityCNN(nn.Module):
    def __init__(self, input_size):
        super(PermeabilityCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        flattened_size = (input_size // 8) * (input_size // 8) * 128
        
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


def load_dataset(filename="nn_dataset.npz"):
    print("Loading dataset...")
    data = np.load(filename, allow_pickle=True)
    X = data['X']
    y = data['y']
    metadata = json.loads(str(data['metadata']))
    
    print(f"Dataset loaded:")
    print(f"  Samples: {len(X)}")
    print(f"  Input shape: {X.shape}")
    print(f"  Permeability range: [{y.min():.4f}, {y.max():.4f}]")
    
    return X, y, metadata


def prepare_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    grid_size = X.shape[1]
    n_samples = X.shape[0]
    
    X_bitstring = X.reshape(n_samples, -1).astype(np.float32)
    
    print(f"\nBitstring preparation:")
    print(f"  Flattened shape: {X_bitstring.shape}")
    print(f"  Grid size: {grid_size}×{grid_size}")
    print(f"  Bitstring length: {grid_size*grid_size}")
    print(f"  Values: 0s and 1s (solid=0, fluid=1)")
    
    
    y = y.astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_bitstring, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size / (1 - test_size), random_state=random_state
    )
    
    print(f"\nTrain/Val/Test split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, grid_size


def train_cnn(X_train, y_train, X_val, y_val, X_test, grid_size, epochs=100, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1)
    X_test_t = torch.FloatTensor(X_test)
    
    X_train_t = X_train_t.reshape(-1, 1, grid_size, grid_size)
    X_val_t = X_val_t.reshape(-1, 1, grid_size, grid_size)
    X_test_t = X_test_t.reshape(-1, 1, grid_size, grid_size)
    
    print(f"\nReshaped bitstrings to 2D for CNN:")
    print(f"  Training shape: {X_train_t.shape} (N, C, H, W)")
    print(f"  Testing shape: {X_test_t.shape}")
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = PermeabilityCNN(grid_size).to(device)
    
    print(f"\nInput: Bitstring of length {grid_size*grid_size}")
    print(f"  Reshaped to: {grid_size}×{grid_size} image")
    print(f"Architecture: CNN with 3 conv layers")
    print(f"  Conv layers: 1→32→64→128")
    print(f"  FC layers: {(grid_size//8)**2*128}→256→128→1")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    print(f"\nTraining for {epochs} epochs...")
    start = time.time()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        with torch.no_grad():
            X_val_d = X_val_t.to(device)
            y_val_d = y_val_t.to(device)
            val_outputs = model(X_val_d)
            val_loss = criterion(val_outputs, y_val_d).item()
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_model_state)
    
    print(f"\nTraining completed in {time.time()-start:.1f}s")
    
    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train_t.to(device)).cpu().numpy().flatten()
        y_test_pred = model(X_test_t.to(device)).cpu().numpy().flatten()
    
    return model, y_train_pred, y_test_pred


def evaluate(y_train, y_train_pred, y_test, y_test_pred):
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    print(f"\nTraining R²:  {train_r2:.4f}")
    print(f"Testing R²:   {test_r2:.4f}")
    print(f"Testing MAE:  {test_mae:.4f}")
    print(f"Testing MAPE: {test_mape:.2f}%")
    
    
    return {'train_r2': train_r2, 'test_r2': test_r2, 'mae': test_mae, 'mape': test_mape}


def plot_results(y_train, y_train_pred, y_test, y_test_pred):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=30)
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Permeability')
    axes[0].set_ylabel('Predicted Permeability')
    axes[0].set_title(f'Training Set (R²={r2_score(y_train, y_train_pred):.3f})')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(y_test, y_test_pred, alpha=0.5, s=30, color='green')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Permeability')
    axes[1].set_ylabel('Predicted Permeability')
    axes[1].set_title(f'Testing Set (R²={r2_score(y_test, y_test_pred):.3f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/cnn_2d_results.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    
    
    torch.manual_seed(42)
    
    X, y, metadata = load_dataset("nn_dataset.npz")
    
    X_train, X_val, X_test, y_train, y_val, y_test, grid_size = prepare_data(X, y)
    
    model, y_train_pred, y_test_pred = train_cnn(
        X_train, y_train, X_val, y_val, X_test, grid_size,
        epochs=100, 
        batch_size=32
    )
    
    metrics = evaluate(y_train, y_train_pred, y_test, y_test_pred)
    
    plot_results(y_train, y_train_pred, y_test, y_test_pred)
    
    torch.save(model.state_dict(), 'cnn_model.pth')
    print("\nModel saved to cnn_model.pth")
    