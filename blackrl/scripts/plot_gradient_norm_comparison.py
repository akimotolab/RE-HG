"""Script to plot gradient norm comparison between SoftQL and SoftVI."""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from plot_learning_curves import plot_gradient_norm_comparison


def main():
    parser = argparse.ArgumentParser(description="Plot gradient norm comparison between SoftQL and SoftVI")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing stats_softql.pkl and stats_softvi.pkl")
    parser.add_argument("--output", type=str, default=None, help="Output path for the plot (default: data_dir/gradient_norm_comparison.pdf)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    softql_path = data_dir / "stats_softql.pkl"
    if not softql_path.exists():
        raise FileNotFoundError(f"SoftQL stats not found at {softql_path}")

    with open(softql_path, "rb") as f:
        softql_stats = pickle.load(f)

    print(f"Loaded SoftQL stats from {softql_path}")
    if "leader_gradient_norm" in softql_stats:
        print(f"  SoftQL gradient norm: {len(softql_stats['leader_gradient_norm'])} iterations")

    softvi_path = data_dir / "stats_softvi.pkl"
    if not softvi_path.exists():
        raise FileNotFoundError(f"SoftVI stats not found at {softvi_path}")

    with open(softvi_path, "rb") as f:
        softvi_stats = pickle.load(f)

    print(f"Loaded SoftVI stats from {softvi_path}")
    if "leader_gradient_norm" in softvi_stats:
        print(f"  SoftVI gradient norm: {len(softvi_stats['leader_gradient_norm'])} iterations")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / "gradient_norm_comparison.pdf"

    # Create plot
    plot_gradient_norm_comparison(softql_stats, softvi_stats, save_path=str(output_path))

    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    main()
