import argparse
import numpy as np
import matplotlib.pyplot as plt
from quantumsparse.tools.plot import use_default_style
use_default_style()

def generate_temperatures(Tmin, Tmax, n, scale):
    if scale == "linspace":
        return np.linspace(Tmin, Tmax, n)

    elif scale == "logspace":
        if Tmin <= 0 or Tmax <= 0:
            raise ValueError("logspace requires Tmin and Tmax > 0")
        return np.logspace(np.log10(Tmin), np.log10(Tmax), n)

    elif scale == "geomspace":
        if Tmin <= 0 or Tmax <= 0:
            raise ValueError("geomspace requires Tmin and Tmax > 0")
        return np.geomspace(Tmin, Tmax, n)

    else:
        raise ValueError(f"Unknown scale type: {scale}")

def plot_temperatures(temps, file):
    temps = np.asarray(temps)

    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharey=True)

    # ------------------
    # Linear scale plot
    # ------------------
    axes[0].vlines(temps, 0, 1)
    axes[0].set_xlim(temps.min(), temps.max())
    axes[0].set_title("Temperature grid (linear scale)")
    axes[0].set_yticks([])

    # ------------------
    # Log scale plot
    # ------------------
    temps_pos = temps[temps > 0]

    if len(temps_pos) == 0:
        raise ValueError("No positive temperatures available for log-scale plot")

    axes[1].vlines(temps_pos, 0, 1)
    axes[1].set_xscale("log")
    axes[1].set_xlim(temps_pos.min(), temps_pos.max())
    axes[1].set_title("Temperature grid (log scale)")
    axes[1].set_yticks([])
    axes[1].set_xlabel("Temperature")

    plt.tight_layout()
    plt.savefig(file, dpi=300)
    plt.close(fig)
    
def main():
    description = "Generate temperature grids with different scalings."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-T1","--Tmin"  , type=float, required=True,help="Initial temperature")
    parser.add_argument("-T2","--Tmax"  , type=float, required=True,help="Final temperature")
    parser.add_argument("-n", "--num"   , type=int, required=True,help="Number of temperature points")
    parser.add_argument("-o", "--output", type=str, required=True,help="txt output file")
    parser.add_argument("-s", "--scale" ,choices=["linspace", "logspace"],default="linspace",help="scaling type (default: %(default)s)")
    parser.add_argument("--fmt", type=str, default="%.8e",help="Output format (default: %(default)s)")
    parser.add_argument(
        "-p", "--plot",
        type=str,
        help="Plot the temperature grid (default: %(default)s).",
        default=None,
    )
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")

    temps = generate_temperatures(
        args.Tmin, args.Tmax, args.num, args.scale
    )

    np.savetxt(
        args.output,
        temps,
        fmt=args.fmt,
        header=(
            f"{args.scale} temperatures from {args.Tmin} "
            f"to {args.Tmax} ({args.num} points)"
        )
    )
    
    if args.plot is not None:
        plot_temperatures(temps,args.plot)


if __name__ == "__main__":
    main()
