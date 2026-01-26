import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def load_pm_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    circles = [(c["cx"], c["cy"], c["radius"]) for c in data["circles"]]
    
    print(f"Loaded PM from {filename}")
    print(f"  Domain size: {data['domain_size']}")
    print(f"  Porosity: {data['porosity']:.3f}")
    print(f"  Number of circles: {data['n_circles']}")
    
    return circles, data["domain_size"], data["porosity"], data["seed"]

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

def equilibrium_distribution(rho, ux, uy, c, w, cs2):
    u = jnp.stack((ux, uy), axis=-1)
    u_reshaped = u[:, :, jnp.newaxis, :]
    c_reshaped = c[jnp.newaxis, jnp.newaxis, :, :]
    c_dot_u = jnp.sum(c_reshaped * u_reshaped, axis=3)
    u_sq = jnp.sum(u**2, axis=2)
    rho_r = rho[:, :, jnp.newaxis]
    w_r = w[jnp.newaxis, jnp.newaxis, :]
    u_sq_r = u_sq[:, :, jnp.newaxis]
    term1 = w_r * rho_r
    term2 = 1 + (c_dot_u / cs2)
    term3 = (c_dot_u**2) / (2 * cs2**2)
    term4 = u_sq_r / (2 * cs2)
    return term1 * (term2 + term3 - term4)

def lbm_step(state, c, w, cs2, force, M, M_inv, S_base, nu, solid_mask):
    f_prev, _, _, _ = state

    f_streamed = f_prev
    for i in range(9):
        f_streamed = f_streamed.at[:, :, i].set(jnp.roll(jnp.roll(f_prev[:, :, i], c[i, 0], axis=0), c[i, 1], axis=1))
    
    bounce_back_indices = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
    f_streamed = jnp.where(
        solid_mask[:, :, jnp.newaxis],
        f_streamed[:, :, bounce_back_indices],
        f_streamed
    )

    rho = jnp.sum(f_streamed, axis=2)
    vel = jnp.einsum('xyq,qi->xyi', f_streamed, c)
    rho_r = rho[:, :, jnp.newaxis]
    vel += 0.5 * force * rho_r
    ux, uy = jnp.dsplit(jnp.where(rho_r > 0, vel / rho_r, 0), 2)
    ux, uy = ux.squeeze(axis=2), uy.squeeze(axis=2)

    tau = (nu / cs2) + 0.5
    
    m = jnp.einsum('ij,xyj->xyi', M, f_streamed)
    rho_ux, rho_uy = rho * ux, rho * uy
    m_eq = jnp.stack([
        rho,
        -2 * rho + 3 * (rho_ux**2 + rho_uy**2),
        rho - 3 * (rho_ux**2 + rho_uy**2),
        rho_ux,
        -rho_ux,
        rho_uy,
        -rho_uy,
        rho_ux**2 - rho_uy**2,
        rho_ux * rho_uy
    ], axis=-1)

    s_nu = 1.0 / tau
    S_local = jnp.broadcast_to(S_base, (rho.shape[0], rho.shape[1], 9))
    S_local = S_local.at[:, :, 7].set(s_nu)
    S_local = S_local.at[:, :, 8].set(s_nu)

    m_collided = m - S_local * (m - m_eq)
    f_collided = jnp.einsum('ij,xyj->xyi', M_inv, m_collided)

    u_vel = jnp.stack((ux, uy), axis=-1)[:, :, jnp.newaxis, :]
    c_dot_u = jnp.sum(c * u_vel, axis=3)
    force_r = force[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    c_r = c[jnp.newaxis, jnp.newaxis, :, :]
    w_r = w[jnp.newaxis, jnp.newaxis, :]
    term1 = (c_r - u_vel) / cs2
    term2 = (c_dot_u[..., jnp.newaxis] * c_r) / (cs2**2)
    force_term = (1 - 1 / (2 * tau)) * w_r * jnp.sum((term1 + term2) * force_r, axis=3)

    f_new = f_collided + force_term

    return (f_new, rho, ux, uy)

def calculate_permeability(porous_mask, nu, force_magnitude, num_steps=10000):
    ny, nx = porous_mask.shape
    print(f"\nCalculating permeability on {nx}x{ny} grid...")
    
    c = jnp.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], 
                   [1, 1], [-1, 1], [-1, -1], [1, -1]])
    w = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    cs2 = 1/3
    
    solid_mask = jnp.array(~porous_mask)
    force = jnp.array([force_magnitude, 0.0])
    
    M = jnp.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [-4, -1, -1, -1, -1, 2, 2, 2, 2],
        [4, -2, -2, -2, -2, 1, 1, 1, 1],
        [0, 1, 0, -1, 0, 1, -1, -1, 1],
        [0, -2, 0, 2, 0, 1, -1, -1, 1],
        [0, 0, 1, 0, -1, 1, 1, -1, -1],
        [0, 0, -2, 0, 2, 1, 1, -1, -1],
        [0, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 1, -1]
    ])
    M_inv = jnp.linalg.inv(M)
    S_base = jnp.array([1.0, 1.4, 1.4, 1.0, 1.2, 1.0, 1.2, 1.0, 1.0])
    
    rho = jnp.ones((nx, ny))
    ux = jnp.zeros((nx, ny))
    uy = jnp.zeros((nx, ny))
    f = equilibrium_distribution(rho, ux, uy, c, w, cs2)
    
    state = (f, rho, ux, uy)
    constants = (c, w, cs2, force, M, M_inv, S_base, nu, solid_mask)
    
    jitted_step = jax.jit(lambda s: lbm_step(s, *constants))
    
    print("Compiling and running simulation...")
    state = jitted_step(state)
    state[0].block_until_ready()
    
    for step in range(num_steps):
        state = jitted_step(state)
        if step % 1000 == 0:
            print(f"  Step {step}/{num_steps}")
    
    state[0].block_until_ready()
    
    _, rho_final, ux_final, uy_final = state
    
    fluid_mask = np.array(porous_mask)
    ux_np = np.array(ux_final)
    
    avg_velocity = np.mean(ux_np[fluid_mask])
    
    mu = nu
    permeability = (mu * avg_velocity) / force_magnitude
    
    print(f"\nResults:")
    print(f"  Average velocity: {avg_velocity:.6e}")
    print(f"  Permeability: {permeability:.6e}")
    
    return permeability, avg_velocity, state

if __name__ == '__main__':
    LBM_GRID_SIZE = 200
    
    NU = 0.01
    FORCE_MAGNITUDE = 1e-6
    NUM_STEPS = 10000
    
    pm_dir = "pm_data"
    output_file = f"{pm_dir}/permeability_results.json"
    
    results = []
    
    pm_files = sorted([f for f in os.listdir(pm_dir) if f.startswith("PM_sample_") and f.endswith(".json")])
    
    for pm_file in pm_files:
        pm_path = os.path.join(pm_dir, pm_file)
        
        print(f"\n{'='*60}")
        print(f"Processing: {pm_file}")
        print(f"{'='*60}")
        
        circles, domain_size, porosity, seed = load_pm_file(pm_path)
        
        porous_mask = rasterize_pm_to_grid(circles, domain_size, LBM_GRID_SIZE, LBM_GRID_SIZE)
        
        print(f"Rasterized to {LBM_GRID_SIZE}x{LBM_GRID_SIZE} grid")
        print(f"Grid porosity: {np.mean(porous_mask):.3f}")
        
        permeability, avg_velocity, final_state = calculate_permeability(
            porous_mask, NU, FORCE_MAGNITUDE, NUM_STEPS
        )
        
        result = {
            "pm_file": pm_file,
            "seed": seed,
            "porosity": float(porosity),
            "grid_size": LBM_GRID_SIZE,
            "permeability": float(permeability),
            "avg_velocity": float(avg_velocity),
            "nu": NU,
            "force": FORCE_MAGNITUDE
        }
        results.append(result)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Processed {len(results)} samples")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")
    
    print("\nPermeability Summary:")
    for r in results:
        print(f"  {r['pm_file']}: K = {r['permeability']:.6e}, φ = {r['porosity']:.3f}")
