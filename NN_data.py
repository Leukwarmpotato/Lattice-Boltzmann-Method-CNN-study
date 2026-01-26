import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.ndimage import zoom

def load_pm_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    circles = [(c["cx"], c["cy"], c["radius"]) for c in data["circles"]]
    return circles, data["domain_size"], data["porosity"], data["seed"]

def load_permeability_results(filename):
    with open(filename, 'r') as f:
        results = json.load(f)
    return results

def rasterize_pm_to_grid(circles, domain_size, grid_size_x, grid_size_y):
    im = np.ones((grid_size_x, grid_size_y), dtype=bool)
    
    y, x = np.ogrid[:grid_size_x, :grid_size_y]
    x_coords = (x / grid_size_x) * domain_size
    y_coords = (y / grid_size_y) * domain_size
    
    for cx, cy, radius in circles:
        dx_direct = np.abs(x_coords - cx)
        dx_wrapped = domain_size - dx_direct
        dx = np.minimum(dx_direct, dx_wrapped)
        
        dy_direct = np.abs(y_coords - cy)
        dy_wrapped = domain_size - dy_direct
        dy = np.minimum(dy_direct, dy_wrapped)
        
        dist = np.sqrt(dx**2 + dy**2)
        mask = dist <= radius
        im[mask] = False
    
    return im

def rescale_porous_media(porous_mask, target_size):
    current_size = porous_mask.shape[0]
    scale_factor = target_size / current_size
    
    float_mask = porous_mask.astype(float)
    
    rescaled_float = zoom(float_mask, scale_factor, order=3)
    
    original_porosity = np.mean(porous_mask)
    threshold = np.percentile(rescaled_float, (1 - original_porosity) * 100)
    rescaled_mask = rescaled_float > threshold
    
    return rescaled_mask

def prepare_nn_dataset(pm_dir="pm_data", target_resolution=64, output_file="nn_dataset.npz"):
    perm_results = load_permeability_results(f"{pm_dir}/permeability_results.json")
    
    n_samples = len(perm_results)
    
    X_images = np.zeros((n_samples, target_resolution, target_resolution), dtype=np.float32)
    y_permeability = np.zeros(n_samples, dtype=np.float32)
    metadata = []
    
    print(f"Preparing dataset with {n_samples} samples at {target_resolution}x{target_resolution} resolution")
    print("="*60)
    
    for i, result in enumerate(perm_results):
        pm_file = result["pm_file"]
        pm_path = os.path.join(pm_dir, pm_file)
        
        circles, domain_size, porosity, seed = load_pm_file(pm_path)
        
        porous_mask = rasterize_pm_to_grid(circles, domain_size, target_resolution, target_resolution)
        
        image = porous_mask.astype(np.float32)
        
        X_images[i] = image
        y_permeability[i] = result["permeability"]
        
        metadata.append({
            "sample_id": i,
            "pm_file": pm_file,
            "seed": seed,
            "porosity": porosity,
            "permeability": result["permeability"]
        })
        
        print(f"Sample {i+1}/{n_samples}: {pm_file}")
        print(f"  Porosity: {porosity:.3f}, Permeability: {result['permeability']:.6e}")
    
    np.savez(output_file,
             X=X_images,
             y=y_permeability,
             metadata=json.dumps(metadata))
    
    print(f"\n{'='*60}")
    print(f"Dataset saved to {output_file}")
    print(f"Shape: X={X_images.shape}, y={y_permeability.shape}")
    print(f"{'='*60}")
    
    return X_images, y_permeability, metadata

def visualize_dataset_samples(X, y, metadata, n_samples=5):
    n_samples = min(n_samples, len(X))
    
    fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 4))
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        axes[i].imshow(X[i].T, origin='lower', cmap='gray_r', aspect='equal')
        axes[i].set_title(f"Sample {i}\nφ={metadata[i]['porosity']:.3f}\nK={y[i]:.2e}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def compare_resolutions(circles, domain_size, resolutions=[32, 64, 128, 256]):
    n_res = len(resolutions)
    fig, axes = plt.subplots(1, n_res, figsize=(4*n_res, 4))
    
    for i, res in enumerate(resolutions):
        mask = rasterize_pm_to_grid(circles, domain_size, res, res)
        axes[i].imshow(mask.T, origin='lower', cmap='gray_r', aspect='equal')
        axes[i].set_title(f'{res}×{res}\nφ={np.mean(mask):.3f}')
        axes[i].axis('off')
    
    plt.suptitle('Porous Media at Different Resolutions')
    plt.tight_layout()
    plt.show()

def example_nn_training():
    print("\nExample: Loading dataset for neural network training")
    print("="*60)
    
    data = np.load("nn_dataset.npz", allow_pickle=True)
    X = data['X']
    y = data['y']
    metadata = json.loads(str(data['metadata']))
    
    print(f"Loaded dataset:")
    print(f"  Input images (X): {X.shape}")
    print(f"  Target permeability (y): {y.shape}")
    print(f"  Number of samples: {len(X)}")
    
    X_cnn = X[..., np.newaxis]
    
    print(f"\nReshaped for CNN: {X_cnn.shape}")
    
    split = int(0.8 * len(X))
    X_train, X_val = X_cnn[:split], X_cnn[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"\nTrain/Val split:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val: X={X_val.shape}, y={y_val.shape}")
    
    print("\n" + "="*60)
    print("Dataset ready for training!")
    print("Use X_train, y_train for training your neural network")
    print("="*60)
    
    return X_train, X_val, y_train, y_val

if __name__ == '__main__':
    PM_DIR = "pm_data"
    NN_RESOLUTION = 50
    
    perm_file = f"{PM_DIR}/permeability_results.json"
    if not os.path.exists(perm_file):
        print(f"ERROR: {perm_file} not found!")
        print("Please run the LBM permeability calculator first (script 2)")
        exit(1)
    
    X, y, metadata = prepare_nn_dataset(
        pm_dir=PM_DIR,
        target_resolution=NN_RESOLUTION,
        output_file="nn_dataset.npz"
    )
    
    visualize_dataset_samples(X, y, metadata, n_samples=5)
    
    example_nn_training()
