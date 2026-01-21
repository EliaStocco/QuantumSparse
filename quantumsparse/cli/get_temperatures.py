import argparse
import numpy as np


def generate_temperatures(tmin, tmax, n, scale):
    if scale == "linspace":
        return np.linspace(tmin, tmax, n)

    elif scale == "logspace":
        if tmin <= 0 or tmax <= 0:
            raise ValueError("logspace requires tmin and tmax > 0")
        return np.logspace(np.log10(tmin), np.log10(tmax), n)

    elif scale == "geomspace":
        if tmin <= 0 or tmax <= 0:
            raise ValueError("geomspace requires tmin and tmax > 0")
        return np.geomspace(tmin, tmax, n)

    else:
        raise ValueError(f"Unknown scale type: {scale}")


def main():
    description = "Generate temperature grids with different scalings."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--tmin", type=float, required=True,help="Initial temperature")
    parser.add_argument("--tmax", type=float, required=True,help="Final temperature")
    parser.add_argument("-n", "--num", type=int, required=True,help="Number of temperature points")
    parser.add_argument("-o", "--output", type=str, required=True,help="Output text file")
    parser.add_argument("-s", "--scale",choices=["linspace", "logspace", "geomspace"],default="linspace",help="Scaling type (default: linspace)")
    parser.add_argument("--fmt", type=str, default="%.6f",help="Output format (default: %%.6f)")
    args = parser.parse_args()
    
    print(f"\n=== {description} ===\n")

    temps = generate_temperatures(
        args.tmin, args.tmax, args.num, args.scale
    )

    np.savetxt(
        args.output,
        temps,
        fmt=args.fmt,
        header=(
            f"{args.scale} temperatures from {args.tmin} "
            f"to {args.tmax} ({args.num} points)"
        )
    )


if __name__ == "__main__":
    main()
