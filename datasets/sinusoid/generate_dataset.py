import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import h5py


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
    sigma = jnp.exp(-2.0 + 0.3 * z)
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


def random_mask(key, n_time):
    k1, k2 = jax.random.split(key, 2)

    n_blocks = int(jax.random.randint(k1, (), 1, 4))
    frac = float(jax.random.uniform(k2, (), minval=0.1, maxval=0.25))
    total_len = max(1, int(frac * n_time))

    m = np.ones(n_time, dtype=np.float32)

    rng = np.random.default_rng(int(jax.random.randint(k1, (), 0, 1_000_000)))
    for _ in range(n_blocks):
        block_len = max(4, total_len // n_blocks)
        block_len = min(block_len, n_time)
        start = rng.integers(0, n_time - block_len + 1)
        m[start:start + block_len] = 0.0

    return jnp.array(m, dtype=jnp.float32)


def simulate_batch(key, n_sim, n_time, use_mask):
    keys = jax.random.split(key, n_sim)

    thetas = []
    xs = []
    masks = []

    for k in keys:
        k1, k2 = jax.random.split(k)

        theta, y_obs = simulate_one(k1, n_time)

        if use_mask:
            mask = random_mask(k2, n_time)
            x = jnp.stack([y_obs, mask], axis=-1)   # shape (T, 2)
            masks.append(np.array(mask, dtype=np.float32))
        else:
            x = y_obs  # shape (T,)

        thetas.append(np.array(theta, dtype=np.float32))
        xs.append(np.array(x, dtype=np.float32))

    theta = np.stack(thetas, axis=0)
    x = np.stack(xs, axis=0)

    if use_mask:
        mask = np.stack(masks, axis=0)
        return theta, x, mask
    else:
        return theta, x, None


def make_splits(theta, x, mask, n_train, n_val):
    data = {
        "theta_train": theta[:n_train],
        "x_train": x[:n_train],
        "theta_val": theta[n_train:n_train + n_val],
        "x_val": x[n_train:n_train + n_val],
        "theta_test": theta[n_train + n_val:],
        "x_test": x[n_train + n_val:],
    }

    if mask is not None:
        data["mask_train"] = mask[:n_train]
        data["mask_val"] = mask[n_train:n_train + n_val]
        data["mask_test"] = mask[n_train + n_val:]

    return data


def save_hdf5(path, data, use_mask, n_time, seed):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with h5py.File(path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v, compression="gzip")

        f.attrs["theta_names"] = "A, f, phi, b"
        f.attrs["n_time"] = int(n_time)
        f.attrs["seed"] = int(seed)
        f.attrs["use_mask"] = bool(use_mask)

    print(f"Saved dataset to {path}")
    for k, v in data.items():
        print(f"{k}: {v.shape}")


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
        key=key,
        n_sim=n_total,
        n_time=args.n_time,
        use_mask=args.use_mask,
    )

    data = make_splits(
        theta=theta,
        x=x,
        mask=mask,
        n_train=args.n_train,
        n_val=args.n_val,
    )

    save_hdf5(
        path=args.out,
        data=data,
        use_mask=args.use_mask,
        n_time=args.n_time,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()