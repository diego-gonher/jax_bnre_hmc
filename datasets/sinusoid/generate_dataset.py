import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import h5py


# -----------------------------
# Core model
# -----------------------------

def make_time_grid(n_time):
    return jnp.linspace(0.0, 1.0, n_time)


def sample_theta(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)

    A = jax.random.uniform(k1, (), minval=0.5, maxval=2.0)
    f = jax.random.uniform(k2, (), minval=0.25, maxval=10.0)
    phi = jax.random.uniform(k3, (), minval=0.0, maxval=2.0 * jnp.pi)
    b = jax.random.uniform(k4, (), minval=-0.5, maxval=0.5)

    return jnp.array([A, f, phi, b], dtype=jnp.float32)


def clean_signal(theta, t):
    A, f, phi, b = theta
    return A * jnp.sin(2.0 * jnp.pi * f * t + phi) + b


def sample_noise(key, n_time):
    z = jax.random.normal(key, (n_time,))
    sigma = jnp.exp(-2.0 + 0.3 * z)  # LogNormal
    return sigma


def simulate_one(key, n_time):
    k1, k2, k3 = jax.random.split(key, 3)

    theta = sample_theta(k1)
    t = make_time_grid(n_time)

    y_clean = clean_signal(theta, t)
    sigma = sample_noise(k2, n_time)

    noise = jax.random.normal(k3, (n_time,)) * sigma
    y_obs = y_clean + noise

    return theta, y_obs


# -----------------------------
# Masking
# -----------------------------

def random_mask(key, n_time):
    k1, k2, k3 = jax.random.split(key, 3)

    n_blocks = jax.random.randint(k1, (), 1, 4)
    frac = jax.random.uniform(k2, (), minval=0.1, maxval=0.25)
    total_len = int(frac * n_time)

    m = jnp.ones(n_time)

    for i in range(int(n_blocks)):
        block_len = max(4, total_len // int(n_blocks))
        start = jax.random.randint(k3, (), 0, n_time - block_len + 1)
        idx = jnp.arange(n_time)
        in_block = (idx >= start) & (idx < start + block_len)
        m = jnp.where(in_block, 0.0, m)

    return m


# -----------------------------
# Dataset generation
# -----------------------------

def simulate_batch(key, n_sim, n_time, use_mask):
    keys = jax.random.split(key, n_sim)

    thetas = []
    xs = []
    masks = []

    for k in keys:
        k1, k2 = jax.random.split(k)

        theta, y_obs = simulate_one(k1, n_time)

        if use_mask:
            m = random_mask(k2, n_time)
        else:
            m = jnp.ones(n_time)

        x = jnp.stack([y_obs, m], axis=-1)

        thetas.append(theta)
        xs.append(x)
        masks.append(m)

    return (
        np.array(thetas),
        np.array(xs),
        np.array(masks),
    )


def make_splits(theta, x, mask, n_train, n_val):
    theta_train = theta[:n_train]
    x_train = x[:n_train]
    mask_train = mask[:n_train]

    theta_val = theta[n_train:n_train + n_val]
    x_val = x[n_train:n_train + n_val]
    mask_val = mask[n_train:n_train + n_val]

    theta_test = theta[n_train + n_val:]
    x_test = x[n_train + n_val:]
    mask_test = mask[n_train + n_val:]

    return {
        "theta_train": theta_train,
        "x_train": x_train,
        "mask_train": mask_train,
        "theta_val": theta_val,
        "x_val": x_val,
        "mask_val": mask_val,
        "theta_test": theta_test,
        "x_test": x_test,
        "mask_test": mask_test,
    }


def save_hdf5(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with h5py.File(path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v, compression="gzip")

    print(f"Saved dataset to {path}")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n_time", type=int, default=50)
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--n_val", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=117)
    parser.add_argument("--use_mask", action="store_true")

    args = parser.parse_args()

    n_total = args.n_train + args.n_val + args.n_test

    key = jax.random.PRNGKey(args.seed)

    theta, x, mask = simulate_batch(
        key,
        n_total,
        args.n_time,
        args.use_mask,
    )

    data = make_splits(theta, x, mask, args.n_train, args.n_val)

    save_hdf5(args.out, data)


if __name__ == "__main__":
    main()