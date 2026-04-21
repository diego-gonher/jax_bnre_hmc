import os
import argparse
import numpy as np
import h5py
from scipy.integrate import solve_ivp


def rhs(t, u, beta, gamma, N):
    S, I, R = u
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return [dS, dI, dR]


def sample_prior(rng, n):
    low = np.array([0.2, 0.08], dtype=np.float64)
    high = np.array([0.6, 0.18], dtype=np.float64)
    return rng.uniform(low=low, high=high, size=(n, 2))


def simulate_raw(theta, N, I0, R0, days, saveat):
    t_eval = np.arange(0.0, days + 1e-12, saveat, dtype=np.float64)

    beta, gamma = theta
    S0 = N - I0 - R0
    u0 = np.array([S0, I0, R0], dtype=np.float64)

    sol = solve_ivp(
        fun=lambda t, u: rhs(t, u, beta, gamma, N),
        t_span=(t_eval[0], t_eval[-1]),
        y0=u0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    expected_shape = (3, len(t_eval))
    if (not sol.success) or sol.y.shape != expected_shape:
        return np.full(expected_shape, np.nan, dtype=np.float64)

    y = sol.y.astype(np.float64)
    if not np.all(np.isfinite(y)):
        return np.full(expected_shape, np.nan, dtype=np.float64)

    if np.any(y < -1e-8):
        return np.full(expected_shape, np.nan, dtype=np.float64)

    return y


def simulate_summary(theta, rng, N, I0, R0, days, saveat, subsample_stride, obs_noise_log_scale):
    u = simulate_raw(
        theta=theta,
        N=N,
        I0=I0,
        R0=R0,
        days=days,
        saveat=saveat,
    )

    if np.isnan(u).any():
        n_time = len(np.arange(0.0, days + 1e-12, saveat, dtype=np.float64)[::subsample_stride])
        return np.full((3 * n_time,), np.nan, dtype=np.float64)

    u_sub = u[:, ::subsample_stride].reshape(-1)

    # Keep log well-defined and avoid absurd values
    u_clamped = np.clip(u_sub, 1e-10, 1e12)

    noisy = rng.lognormal(
        mean=np.log(u_clamped),
        sigma=obs_noise_log_scale,
    )

    return noisy.astype(np.float64)


def make_split(rng, n, N, I0, R0, days, saveat, subsample_stride, obs_noise_log_scale):
    theta = sample_prior(rng, n)
    x = np.array(
        [
            simulate_summary(
                theta_i,
                rng=rng,
                N=N,
                I0=I0,
                R0=R0,
                days=days,
                saveat=saveat,
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
    N,
    I0,
    R0,
    days,
    saveat,
    subsample_stride,
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

        f.attrs["theta_names"] = "beta, gamma"
        f.attrs["N"] = float(N)
        f.attrs["I0"] = float(I0)
        f.attrs["R0"] = float(R0)
        f.attrs["days"] = float(days)
        f.attrs["saveat"] = float(saveat)
        f.attrs["subsample_stride"] = int(subsample_stride)
        f.attrs["obs_noise_log_scale"] = float(obs_noise_log_scale)
        f.attrs["seed"] = int(seed)

        f.attrs["prior_low"] = np.array([0.2, 0.08], dtype=np.float64)
        f.attrs["prior_high"] = np.array([0.6, 0.18], dtype=np.float64)

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

    parser.add_argument("--N", type=float, default=1_000_000.0)
    parser.add_argument("--I0", type=float, default=1.0)
    parser.add_argument("--R0", type=float, default=0.0)
    parser.add_argument("--days", type=float, default=160.0)
    parser.add_argument("--saveat", type=float, default=1.0)
    parser.add_argument("--subsample_stride", type=int, default=2)
    parser.add_argument("--obs_noise_log_scale", type=float, default=0.05)

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    theta_train, x_train = make_split(
        rng=rng,
        n=args.n_train,
        N=args.N,
        I0=args.I0,
        R0=args.R0,
        days=args.days,
        saveat=args.saveat,
        subsample_stride=args.subsample_stride,
        obs_noise_log_scale=args.obs_noise_log_scale,
    )

    theta_val, x_val = make_split(
        rng=rng,
        n=args.n_val,
        N=args.N,
        I0=args.I0,
        R0=args.R0,
        days=args.days,
        saveat=args.saveat,
        subsample_stride=args.subsample_stride,
        obs_noise_log_scale=args.obs_noise_log_scale,
    )

    theta_test, x_test = make_split(
        rng=rng,
        n=args.n_test,
        N=args.N,
        I0=args.I0,
        R0=args.R0,
        days=args.days,
        saveat=args.saveat,
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
        N=args.N,
        I0=args.I0,
        R0=args.R0,
        days=args.days,
        saveat=args.saveat,
        subsample_stride=args.subsample_stride,
        obs_noise_log_scale=args.obs_noise_log_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()