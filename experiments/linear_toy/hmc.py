import jax
import jax.numpy as jnp
import numpyro
numpyro.set_host_device_count(4)

from jax_bnre_hmc.checkpointing import load_best_params
from jax_bnre_hmc.hmc import BoxPrior, make_log_ratio_fn, make_potential_fn, run_nuts, z_to_theta
from jax_bnre_hmc.model import RatioEstimatorMLP

# 1) rebuild the model *architecture* (must match training config)
model = RatioEstimatorMLP(hidden_dims=(50, 50, 50), activation="tanh", norm="layernorm")

# 2) load params (PyTree) and get apply_fn
best_dir = '/Users/diegogonzalez/Documents/Research/ENIGMA/jax_bnre_hmc/jax_bnre_hmc/outputs/linear_toy/2026-01-26_17-09-04/checkpoints/best/'
params = load_best_params(best_dir=best_dir)   # point this at your Hydra run checkpoints/best

# 3) observation
def simulate_linear_dataset(
    key: jax.Array,
    n: int,
    n_points: int,
    sigma: float,
    m_low: float,
    m_high: float,
    b_low: float,
    b_high: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulator:
      y = m * x + b + Normal(0, sigma)
    where x_grid = [0, 1, ..., n_points-1]
    Observation x_o is the full y vector (length n_points)
    and theta = (m, b)

    Returns:
      theta: (n, 2)
      x:     (n, n_points)
    """
    key_m, key_b, key_noise = jax.random.split(key, 3)

    m = jax.random.uniform(key_m, (n,), minval=m_low, maxval=m_high)
    b = jax.random.uniform(key_b, (n,), minval=b_low, maxval=b_high)
    theta = jnp.stack([m, b], axis=-1)  # (n, 2)

    x_grid = jnp.arange(n_points, dtype=jnp.float32)  # (n_points,)
    y_clean = m[:, None] * x_grid[None, :] + b[:, None]  # (n, n_points)

    noise = sigma * jax.random.normal(key_noise, (n, n_points))
    y_noisy = y_clean + noise

    return theta.astype(jnp.float32), y_noisy.astype(jnp.float32)

key = jax.random.PRNGKey(42)
# Simulate the full dataset
theta_true, x_obs = simulate_linear_dataset(
        key=key,
        n=1,
        n_points=10,
        sigma=0.1,
        m_low=0.0,
        m_high=1.0,
        b_low=0.0,
        b_high=1.0,
    )

# squeeze to get single observation
x_obs = x_obs[0]            # now shape (10,)
theta_true = theta_true[0]  # shape (2,)

# 4) log-ratio wrapper
log_ratio = make_log_ratio_fn(model.apply, params, x_obs)

# 5) prior bounds (D,)
prior = BoxPrior(
    low=jnp.array([0.0, 0.0], dtype=jnp.float32),
    high=jnp.array([ 1.0,  1.0], dtype=jnp.float32),
)

# 6) potential in unconstrained z-space
potential = make_potential_fn(log_ratio, prior)

# 7) init positions for chains
num_chains = 4
D = prior.low.shape[0]
init_z = jnp.zeros((num_chains, D), dtype=jnp.float32)

# 8) run NUTS
mcmc = run_nuts(potential, jax.random.PRNGKey(0), init_z, num_warmup=1000, num_samples=1000, num_chains=num_chains)

# 9) samples: numpyro will return z samples; map to theta
z_samples = mcmc.get_samples(group_by_chain=False)  # (num_chains*num_samples, D)
theta_samples, _ = jax.vmap(lambda z: z_to_theta(z, prior))(z_samples)

print(theta_samples.shape)
print(mcmc.print_summary())
