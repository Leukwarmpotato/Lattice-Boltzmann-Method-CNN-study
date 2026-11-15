import numpy as np
import os
import matplotlib.pyplot as plt
import json

def periodic_distance(x1, x2, L):
    direct = np.abs(x1 - x2)
    wrapped = L - direct
    return np.minimum(direct, wrapped)

def is_valid_placement(cx, cy, r, existing_circles, min_spacing, domain_size):
    for ex, ey, er in existing_circles:
        dx = periodic_distance(cx, ex, domain_size)
        dy = periodic_distance(cy, ey, domain_size)
        center_distance = np.sqrt(dx**2 + dy**2)
        min_center_distance = r + er + min_spacing
        if center_distance < min_center_distance:
            return False
    return True

def generate_porous_media_descriptor(domain_size=1.0, porosity=0.7, radius=0.05, 
                                     min_spacing=0.01, seed=None, max_attempts=1000):
    if seed is None:
        seed = np.random.randint(0, 1_000_000)
    rng = np.random.default_rng(seed)
    
    circles = []
    target_solid_fraction = 1 - porosity
    domain_area = domain_size * domain_size
    circle_area = np.pi * radius**2
    n_circles = int(target_solid_fraction * domain_area / circle_area * 1.5)
    
    for i in range(n_circles):
        placed = False
        for attempt in range(max_attempts):
            cx = rng.uniform(0, domain_size)
            cy = rng.uniform(0, domain_size)
            if is_valid_placement(cx, cy, radius, circles, min_spacing, domain_size):
                circles.append((cx, cy, radius))
                placed = True
                break
        if not placed:
            break
        total_circle_area = len(circles) * np.pi * radius**2
        current_porosity = 1 - (total_circle_area / domain_area)
        if current_porosity <= porosity:
            break
    
    total_circle_area = len(circles) * np.pi * radius**2
    actual_porosity = 1 - (total_circle_area / domain_area)
    return circles, actual_porosity, seed

def save_pm_file(circles, domain_size, actual_porosity, seed, filename="PM.json"):
    data = {
        "domain_size": domain_size,
        "porosity": actual_porosity,
        "seed": seed,
        "n_circles": len(circles),
        "circles": [{"cx": cx, "cy": cy, "radius": r} for cx, cy, r in circles]
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved PM descriptor to {filename}")

def visualize_pm_descriptor(circles, domain_size, resolution=200):
    im = np.ones((resolution, resolution), dtype=bool)
    y, x = np.ogrid[:resolution, :resolution]
    x_coords = (x / resolution) * domain_size
    y_coords = (y / resolution) * domain_size
    
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
    
    plt.figure(figsize=(8, 8))
    plt.imshow(im.T, origin='lower', cmap='gray_r', aspect='equal')
    plt.title(f'Porous Media Preview\n{len(circles)} circles, φ≈{np.mean(im):.3f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/porous_media_example.png", dpi=150, bbox_inches="tight")
    plt.close()

def generate_dataset(n_samples=1000, domain_size=1.0, porosity_range=(0.3, 0.7),
                    radius=0.05, min_spacing=0.01, output_dir="pm_data"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    rng = np.random.default_rng()
    dataset_info = []
    
    for i in range(n_samples):
        porosity = rng.uniform(*porosity_range)
        circles, actual_porosity, seed = generate_porous_media_descriptor(
            domain_size=domain_size,
            porosity=porosity,
            radius=radius,
            min_spacing=min_spacing,
            seed=None
        )
        
        filename = f"{output_dir}/PM_sample_{i:03d}.json"
        save_pm_file(circles, domain_size, actual_porosity, seed, filename)
        
        dataset_info.append({
            "sample_id": i,
            "filename": filename,
            "porosity": actual_porosity,
            "n_circles": len(circles),
            "seed": seed
        })
        
        if i == 0:
            visualize_pm_descriptor(circles, domain_size, resolution=200)
    
    summary_file = f"{output_dir}/dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    print(f"Dataset summary saved to {summary_file}")

if __name__ == '__main__':
    DOMAIN_SIZE = 1.0
    POROSITY_MIN = 0.6
    POROSITY_MAX = 0.85
    CIRCLE_RADIUS = 0.05
    MIN_SPACING = 0.01
    N_SAMPLES = 1000
    
    generate_dataset(
        n_samples=N_SAMPLES,
        domain_size=DOMAIN_SIZE,
        porosity_range=(POROSITY_MIN, POROSITY_MAX),
        radius=CIRCLE_RADIUS,
        min_spacing=MIN_SPACING,
        output_dir="pm_data"
    )
