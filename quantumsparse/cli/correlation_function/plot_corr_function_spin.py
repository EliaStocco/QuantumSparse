import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def load_pair_file(path):
    return pd.read_csv(path)


def parse_pair(fname):
    base = fname.replace("C_", "").replace(".csv", "")
    return base[0], base[1]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ----------------------------
    # collect files
    # ----------------------------
    files = [f for f in os.listdir(args.input)
             if f.startswith("C_") and f.endswith(".csv")]
    files.sort()

    print(f"\nFound {len(files)} correlation files\n")

    # ============================================================
    # FIGURE 1: nearest neighbor (r=1)
    # ============================================================
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    print("Plotting nearest-neighbor correlations (r=1) ...")

    for fname, ax in zip(files, axes):

        df = load_pair_file(os.path.join(args.input, fname))
        A, B = parse_pair(fname)

        if "r=1" not in df.columns:
            print(f"Warning: r=1 missing in {fname}")
            continue

        ax.plot(df["T"], df["r=1"])

        ax.set_title(f"{A}{B} (r=1)")
        ax.set_xlabel("T")
        ax.set_ylabel("C(T)")
        ax.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(args.output, "nearest_neighbor_3x3.png"), dpi=300)

    # ============================================================
    # FIGURE 2: C(T) for all distances
    # ============================================================
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    print("Plotting full distance dependence ...")

    for fname, ax in zip(files, axes):

        df = load_pair_file(os.path.join(args.input, fname))
        A, B = parse_pair(fname)

        r_cols = [c for c in df.columns if c.startswith("r=")]
        r_cols.sort(key=lambda x: int(x.split("=")[1]))

        for col in r_cols:
            ax.plot(df["T"], df[col], label=col)

        ax.set_title(f"{A}{B}")
        ax.set_xlabel("T")
        ax.set_ylabel("C(T)")
        ax.legend(fontsize=6)
        ax.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(args.output, "distance_vs_temperature_3x3.png"), dpi=300)

    # ============================================================
    # FIGURE 3: C(r,T) with LOG temperature colorbar
    # ============================================================
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    print("Plotting C(r,T) with log temperature colorbar ...")

    cmap = plt.cm.viridis

    # global normalization check (assumes consistent T grid)
    all_T = []

    for fname in files:
        df = load_pair_file(os.path.join(args.input, fname))
        all_T.append(df["T"].values)

    all_T = np.concatenate(all_T)

    if np.any(all_T <= 0):
        raise ValueError("Log color scale requires all T > 0")

    norm = LogNorm(vmin=all_T.min(), vmax=all_T.max())

    for fname, ax in zip(files, axes):

        df = load_pair_file(os.path.join(args.input, fname))
        A, B = parse_pair(fname)

        r_cols = [c for c in df.columns if c.startswith("r=")]
        r_vals = np.array([int(c.split("=")[1]) for c in r_cols])

        order = np.argsort(r_vals)
        r_vals = r_vals[order]
        r_cols = [r_cols[i] for i in order]

        T = df["T"].values

        for i, t in enumerate(T):
            C_r = [df[col].iloc[i] for col in r_cols]

            ax.plot(
                r_vals,
                C_r,
                marker="o",
                color=cmap(norm(t))
            )

        ax.set_title(f"{A}{B}")
        ax.set_xlabel("Distance r")
        ax.set_ylabel("C(r,T)")
        ax.grid(True)

    # shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="vertical",
        fraction=0.02,
        pad=0.02
    )
    cbar.set_label("Temperature (log scale)")

    # plt.tight_layout()
    fig.savefig(os.path.join(args.output, "CrT_3x3_logT.png"), dpi=300)
    
    # ============================================================
    # FIGURE 4: HEATMAP C(r,T)
    # ============================================================
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    print("Plotting heatmaps C(r,T) ...")

    cmap = plt.cm.viridis

    for fname, ax in zip(files, axes):

        df = load_pair_file(os.path.join(args.input, fname))
        A, B = parse_pair(fname)

        r_cols = [c for c in df.columns if c.startswith("r=")]
        r_vals = np.array([int(c.split("=")[1]) for c in r_cols])

        order = np.argsort(r_vals)
        r_vals = r_vals[order]
        r_cols = [r_cols[i] for i in order]

        T = df["T"].values

        # matrix: rows = T, cols = r
        Z = np.zeros((len(T), len(r_cols)))

        for i, col in enumerate(r_cols):
            Z[:, i] = df[col].values

        # heatmap
        im = ax.imshow(
            Z,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[
                r_vals.min(),
                r_vals.max(),
                T.min(),
                T.max()
            ]
        )

        ax.set_title(f"{A}{B}")
        ax.set_xlabel("Distance r")
        ax.set_ylabel("Temperature T")

        ax.set_yscale("log")  # matches your log-color physics intuition

    # shared colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("C(r,T)")

    plt.tight_layout()
    fig.savefig(os.path.join(args.output, "CrT_heatmap_3x3.png"), dpi=300)

    print("Heatmaps saved.")

    print("\nDone.")


if __name__ == "__main__":
    main()