import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import json
import time
import pickle

def load_dataset(filename="nn_dataset.npz"):
    data = np.load(filename, allow_pickle=True)
    X = data['X']
    y = data['y']
    metadata = json.loads(str(data['metadata']))
    return X, y, metadata

def prepare_data(X, y, test_size=0.2, random_state=42):
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1).astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
        verbose=False,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=50,
        tol=1e-5,
        batch_size=32
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    return model, train_time

def evaluate(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'mae': test_mae,
        'mape': test_mape,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }

def plot_results(y_train, y_train_pred, y_test, y_test_pred, test_r2):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    axes[0].scatter(y_train, y_train_pred, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    min_val, max_val = y_train.min(), y_train.max()
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0].set_xlabel('Actual Permeability')
    axes[0].set_ylabel('Predicted Permeability')
    axes[0].set_title('Training Set')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(y_test, y_test_pred, alpha=0.6, s=30, color='green', edgecolors='k', linewidth=0.5)
    min_val, max_val = y_test.min(), y_test.max()
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[1].set_xlabel('Actual Permeability')
    axes[1].set_ylabel('Predicted Permeability')
    axes[1].set_title(f'Testing Set (R² = {test_r2:.4f})')
    axes[1].grid(True, alpha=0.3)
    
    residuals = y_test - y_test_pred
    axes[2].hist(residuals, bins=25, alpha=0.75, edgecolor='black')
    axes[2].axvline(0, color='r', linestyle='--', lw=2)
    axes[2].set_xlabel('Error')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Residual Distribution')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    X, y, metadata = load_dataset("nn_dataset.npz")
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)
    
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    model, train_time = train_model(X_train, y_train)
    results = evaluate(model, X_train, X_test, y_train, y_test)
    
    print(f"Training time: {train_time:.2f}s")
    print(f"Train R²: {results['train_r2']:.4f}")
    print(f"Test R²:  {results['test_r2']:.4f}")
    print(f"Test MAE: {results['mae']:.4f}")
    print(f"Test MAPE: {results['mape']:.2f}%")
    
    plot_results(y_train, results['y_train_pred'], y_test, results['y_test_pred'], results['test_r2'])
    
    with open('ann_flattened.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'results': results}, f)