import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import time
from functools import partial

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

    bounce_back_indices = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    f_streamed = f_prev
    for i in range(9):
        streamed = jnp.roll(jnp.roll(f_prev[:, :, i], c[i, 0], axis=0), c[i, 1], axis=1)
        from_solid = jnp.roll(jnp.roll(solid_mask, c[i, 0], axis=0), c[i, 1], axis=1)
        f_streamed = f_streamed.at[:, :, i].set(jnp.where(from_solid, f_prev[:, :, bounce_back_indices[i]], streamed))

    rho = jnp.sum(f_streamed, axis=2)
    vel = jnp.einsum('xyq,qi->xyi', f_streamed, c)
    rho_r = rho[:, :, jnp.newaxis]
    vel += 0.5 * force * rho_r
    ux, uy = jnp.dsplit(jnp.where(rho_r > 0, vel / rho_r, 0), 2)
    ux, uy = ux.squeeze(axis=2), uy.squeeze(axis=2)
    ux = jnp.where(solid_mask, 0.0, ux)
    uy = jnp.where(solid_mask, 0.0, uy)

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
    s_q = 8.0 * (2.0 - s_nu) / (8.0 - s_nu)
    S_local = S_local.at[:, :, 4].set(s_q)
    S_local = S_local.at[:, :, 6].set(s_q)
    S_local = S_local.at[:, :, 7].set(s_nu)
    S_local = S_local.at[:, :, 8].set(s_nu)

    u_vel = jnp.stack((ux, uy), axis=-1)[:, :, jnp.newaxis, :]
    c_dot_u = jnp.sum(c * u_vel, axis=3)
    force_r = force[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    c_r = c[jnp.newaxis, jnp.newaxis, :, :]
    w_r = w[jnp.newaxis, jnp.newaxis, :]
    term1 = (c_r - u_vel) / cs2
    term2 = (c_dot_u[..., jnp.newaxis] * c_r) / (cs2**2)
    force_term = w_r * jnp.sum((term1 + term2) * force_r, axis=3)
    m_force = jnp.einsum('ij,xyj->xyi', M, force_term)

    m_collided = m - S_local * (m - m_eq) + (1 - S_local / 2) * m_force
    f_new = jnp.einsum('ij,xyj->xyi', M_inv, m_collided)

    return (f_new, rho, ux, uy)

@partial(jax.jit, static_argnums=(2,))
def advance(state, solid_mask, n_steps, c, w, cs2, force, M, M_inv, S_base, nu):
    return jax.lax.fori_loop(0, n_steps, lambda i, s: lbm_step(s, c, w, cs2, force, M, M_inv, S_base, nu, solid_mask), state)

def calculate_permeability(porous_mask, nu, force_magnitude, max_steps=20000, check_every=500, min_steps=2000, tol=1e-4):
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
    ], dtype=float)
    M_inv = jnp.linalg.inv(M)
    S_base = jnp.array([1.0, 1.4, 1.4, 1.0, 1.2, 1.0, 1.2, 1.0, 1.0])
    
    rho = jnp.ones((nx, ny))
    ux = jnp.zeros((nx, ny))
    uy = jnp.zeros((nx, ny))
    f = equilibrium_distribution(rho, ux, uy, c, w, cs2)
    
    state = (f, rho, ux, uy)
    constants = (c, w, cs2, force, M, M_inv, S_base, nu)
    
    fluid_mask = np.array(porous_mask)
    n_nodes = fluid_mask.size
    
    steps = 0
    prev_velocity = None
    converged = False
    while steps < max_steps:
        state = advance(state, solid_mask, check_every, *constants)
        steps += check_every
        ux_np = np.array(state[2])
        superficial_velocity = ux_np[fluid_mask].sum() / n_nodes
        if prev_velocity is not None and steps >= min_steps and abs(superficial_velocity - prev_velocity) <= tol * abs(superficial_velocity):
            converged = True
            break
        prev_velocity = superficial_velocity
    
    _, rho_final, ux_final, uy_final = state
    
    mu = nu
    permeability = (mu * superficial_velocity) / force_magnitude
    
    print(f"\nResults:")
    print(f"  Steps: {steps} ({'converged' if converged else 'reached step limit'})")
    print(f"  Superficial velocity: {superficial_velocity:.6e}")
    print(f"  Permeability: {permeability:.6e}")
    
    return permeability, superficial_velocity, steps, converged, state

def save_results(results, output_file):
    tmp_file = output_file + ".tmp"
    with open(tmp_file, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_file, output_file)

if __name__ == '__main__':
    LBM_GRID_SIZE = 200
    
    NU = 1/6
    FORCE_MAGNITUDE = 1e-6
    MAX_STEPS = 20000
    
    pm_dir = "pm_data"
    output_file = f"{pm_dir}/permeability_results.json"
    
    results = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
    done = {r["pm_file"] for r in results}
    
    pm_files = sorted([f for f in os.listdir(pm_dir) if f.startswith("PM_sample_") and f.endswith(".json")])
    if len(sys.argv) > 1:
        pm_files = pm_files[:int(sys.argv[1])]
    todo = [f for f in pm_files if f not in done]
    print(f"{len(done)} samples already done, {len(todo)} to run")
    
    run_start = time.time()
    for k, pm_file in enumerate(todo):
        pm_path = os.path.join(pm_dir, pm_file)
        
        print(f"\n{'='*60}")
        print(f"Processing: {pm_file} ({k+1}/{len(todo)})")
        print(f"{'='*60}")
        
        circles, domain_size, porosity, seed = load_pm_file(pm_path)
        
        porous_mask = rasterize_pm_to_grid(circles, domain_size, LBM_GRID_SIZE, LBM_GRID_SIZE)
        grid_porosity = float(np.mean(porous_mask))
        
        print(f"Rasterized to {LBM_GRID_SIZE}x{LBM_GRID_SIZE} grid")
        print(f"Grid porosity: {grid_porosity:.3f}")
        
        sample_start = time.time()
        permeability, superficial_velocity, steps, converged, final_state = calculate_permeability(
            porous_mask, NU, FORCE_MAGNITUDE, MAX_STEPS
        )
        
        results.append({
            "pm_file": pm_file,
            "seed": seed,
            "porosity": float(porosity),
            "grid_porosity": grid_porosity,
            "grid_size": LBM_GRID_SIZE,
            "permeability": float(permeability),
            "superficial_velocity": float(superficial_velocity),
            "steps": steps,
            "converged": converged,
            "nu": NU,
            "force": FORCE_MAGNITUDE
        })
        save_results(results, output_file)
        
        elapsed = time.time() - run_start
        remaining = elapsed / (k + 1) * (len(todo) - k - 1)
        print(f"Sample time: {time.time() - sample_start:.1f}s, estimated time remaining: {remaining/3600:.1f} h")
    
    print(f"\n{'='*60}")
    print(f"Processed {len(results)} samples")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")
