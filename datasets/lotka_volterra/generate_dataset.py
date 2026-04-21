import os
import argparse
import numpy as np
import h5py
from scipy.integrate import solve_ivp


def rhs(t, u, alpha, beta, gamma, delta):
    x, y = u
    dx = alpha * x - beta * x * y
    dy = -gamma * y + delta * x * y
    return [dx, dy]


def sample_prior(rng, n):
    low = np.array([0.5, 0.01, 0.5, 0.01], dtype=np.float64)
    high = np.array([1.5, 0.10, 1.5, 0.10], dtype=np.float64)
    return rng.uniform(low=low, high=high, size=(n, 4))


def simulate_raw(theta, days, saveat, u0):
    t_eval = np.arange(0.0, days + 1e-12, saveat, dtype=np.float64)
    alpha, beta, gamma, delta = theta

    sol = solve_ivp(
        fun=lambda t, u: rhs(t, u, alpha, beta, gamma, delta),
        t_span=(t_eval[0], t_eval[-1]),
        y0=np.asarray(u0, dtype=np.float64),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    expected_shape = (2, len(t_eval))
    if (not sol.success) or sol.y.shape != expected_shape:
        return np.full(expected_shape, np.nan, dtype=np.float64)

    y = sol.y.astype(np.float64)
    if not np.all(np.isfinite(y)):
        return np.full(expected_shape, np.nan, dtype=np.float64)

    return y


def simulate_summary(theta, rng, days, saveat, u0, subsample_stride, obs_noise_log_scale):
    u = simulate_raw(theta, days=days, saveat=saveat, u0=u0)

    if np.isnan(u).any():
        n_time = len(np.arange(0.0, days + 1e-12, saveat, dtype=np.float64)[::subsample_stride])
        return np.full((2 * n_time,), np.nan, dtype=np.float64)

    u_sub = u[:, ::subsample_stride].reshape(-1)

    # Keep log well-defined and avoid absurd values
    u_clamped = np.clip(u_sub, 1e-10, 1e4)

    noisy = rng.lognormal(
        mean=np.log(u_clamped),
        sigma=obs_noise_log_scale,
    )

    return noisy.astype(np.float64)


def make_split(rng, n, days, saveat, u0, subsample_stride, obs_noise_log_scale):
    theta = sample_prior(rng, n)
    x = np.array(
        [
            simulate_summary(
                theta_i,
                rng=rng,
                days=days,
                saveat=saveat,
                u0=u0,
                subsample_stride=subsample_stride,
                obs_noise_log_scale=obs_noise_log_scale,
            )
            for theta_i in theta
        ],
        dtype=np.float64,
    )
    return theta, x


def save_dataset(
    out_path,
    theta_train,
    x_train,
    theta_val,
    x_val,
    theta_test,
    x_test,
    days,
    saveat,
    subsample_stride,
    u0,
    obs_noise_log_scale,
    seed,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("theta_train", data=theta_train, compression="gzip")
        f.create_dataset("x_train", data=x_train, compression="gzip")

        f.create_dataset("theta_val", data=theta_val, compression="gzip")
        f.create_dataset("x_val", data=x_val, compression="gzip")

        f.create_dataset("theta_test", data=theta_test, compression="gzip")
        f.create_dataset("x_test", data=x_test, compression="gzip")

        theta_names = np.array([r"$\alpha$", r"$\beta$", r"$\gamma$", r"$delta$"], dtype="S")
        f.attrs["theta_names"] = theta_names
        f.attrs["days"] = float(days)
        f.attrs["saveat"] = float(saveat)
        f.attrs["subsample_stride"] = int(subsample_stride)
        f.attrs["u0"] = np.asarray(u0, dtype=np.float64)
        f.attrs["obs_noise_log_scale"] = float(obs_noise_log_scale)
        f.attrs["seed"] = int(seed)

        f.attrs["prior_low"] = np.array([0.5, 0.01, 0.5, 0.01], dtype=np.float64)
        f.attrs["prior_high"] = np.array([1.5, 0.10, 1.5, 0.10], dtype=np.float64)

    print(f"Saved dataset to {out_path}")
    print(f"theta_train shape: {theta_train.shape}")
    print(f"x_train shape: {x_train.shape}")
    print(f"theta_val shape: {theta_val.shape}")
    print(f"x_val shape: {x_val.shape}")
    print(f"theta_test shape: {theta_test.shape}")
    print(f"x_test shape: {x_test.shape}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--n_val", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=57038)

    parser.add_argument("--days", type=float, default=20.0)
    parser.add_argument("--saveat", type=float, default=0.1)
    parser.add_argument("--subsample_stride", type=int, default=3)
    parser.add_argument("--obs_noise_log_scale", type=float, default=0.05)

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    u0 = (30.0, 1.0)

    theta_train, x_train = make_split(
        rng=rng,
        n=args.n_train,
        days=args.days,
        saveat=args.saveat,
        u0=u0,
        subsample_stride=args.subsample_stride,
        obs_noise_log_scale=args.obs_noise_log_scale,
    )

    theta_val, x_val = make_split(
        rng=rng,
        n=args.n_val,
        days=args.days,
        saveat=args.saveat,
        u0=u0,
        subsample_stride=args.subsample_stride,
        obs_noise_log_scale=args.obs_noise_log_scale,
    )

    theta_test, x_test = make_split(
        rng=rng,
        n=args.n_test,
        days=args.days,
        saveat=args.saveat,
        u0=u0,
        subsample_stride=args.subsample_stride,
        obs_noise_log_scale=args.obs_noise_log_scale,
    )

    save_dataset(
        out_path=args.out,
        theta_train=theta_train,
        x_train=x_train,
        theta_val=theta_val,
        x_val=x_val,
        theta_test=theta_test,
        x_test=x_test,
        days=args.days,
        saveat=args.saveat,
        subsample_stride=args.subsample_stride,
        u0=u0,
        obs_noise_log_scale=args.obs_noise_log_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()