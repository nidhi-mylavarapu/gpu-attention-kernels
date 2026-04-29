"""
Plot GPU memory vs sequence length from the benchmark CSV (``results.csv``).

Uses the same series colors and ordering as ``plot_runtime_vs_seqlen.py``.
By default plots **peak** GPU memory (column ``peak_MB``); optionally ``workspace_MB``.

Usage (from repo root)::

  uv run python python/plot_memory_vs_seqlen.py
  uv run python python/plot_memory_vs_seqlen.py --metric workspace
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_SERIES_DENSE = {
    "naive_cublas": ("Naive (cuBLAS)", "#bdd7e7", "-", "o"),
    "tiled_online": ("Tiled online", "#4292c6", "-", "s"),
    "tiled_online_wmma": ("Tiled online (WMMA)", "#2171b5", "-", "v"),
    "flash_attn_official": ("FlashAttention-2", "#084594", "-", "^"),
}
_SERIES_SPARSE = {
    "sparse_window": ("Sparse window", "#ea580c", "-", "D"),
}
SERIES_ORDER = [
    ("naive_cublas", _SERIES_DENSE["naive_cublas"]),
    ("tiled_online", _SERIES_DENSE["tiled_online"]),
    ("tiled_online_wmma", _SERIES_DENSE["tiled_online_wmma"]),
    ("flash_attn_official", _SERIES_DENSE["flash_attn_official"]),
    ("sparse_window", _SERIES_SPARSE["sparse_window"]),
]


def _default_csv_path() -> Path:
    cand = ROOT / "results.csv"
    if cand.is_file():
        return cand
    alt = ROOT / "results" / "results.csv"
    if alt.is_file():
        return alt
    return cand


def load_by_impl(
    csv_path: Path, metric: str
) -> Dict[str, List[Tuple[int, float]]]:
    """Return impl -> [(seq_len, memory_MB), ...] sorted by seq_len."""
    col = "peak_MB" if metric == "peak" else "workspace_MB"
    by_impl = defaultdict(list)  # type: Dict[str, List[Tuple[int, float]]]
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            impl = row["impl"].strip()
            seq_len = int(row["seq_len"])
            mem = float(row[col])
            by_impl[impl].append((seq_len, mem))
    for k in by_impl:
        by_impl[k].sort(key=lambda t: t[0])
    return dict(by_impl)


def plot_memory_vs_seqlen(
    csv_path: Path,
    metric: str,
    out_path: Optional[Path] = None,
) -> Path:
    data = load_by_impl(csv_path, metric)
    if out_path is None:
        plot_dir = ROOT / "results" / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        suffix = "peak_MB" if metric == "peak" else "workspace_MB"
        out_path = plot_dir / f"memory_vs_seq_len_{suffix}.png"

    ylabel = (
        "Peak GPU memory (MB)"
        if metric == "peak"
        else "Workspace allocation (MB)"
    )
    title = (
        "Peak GPU memory vs sequence length"
        if metric == "peak"
        else "Attention workspace vs sequence length"
    )

    plt.figure(figsize=(8.5, 5.5))

    for impl_key, (label, color, ls, marker) in SERIES_ORDER:
        rows = data.get(impl_key)
        if not rows:
            continue
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        plt.plot(
            xs,
            ys,
            label=label,
            color=color,
            linestyle=ls,
            marker=marker,
            markersize=6,
            linewidth=2.0,
        )

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Sequence length")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Plot GPU memory vs sequence length from CSV."
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV path (default: results.csv in repo root, or results/results.csv)",
    )
    ap.add_argument(
        "--metric",
        choices=("peak", "workspace"),
        default="peak",
        help="peak = peak_GPU memory (MB); workspace = workspace_bytes-based MB",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path (default: "
            "results/plots/memory_vs_seq_len_peak_MB.png or "
            "memory_vs_seq_len_workspace_MB.png)"
        ),
    )
    args = ap.parse_args()
    csv_path = args.csv if args.csv is not None else _default_csv_path()
    if not csv_path.is_file():
        raise SystemExit("CSV not found: {}".format(csv_path))

    out = plot_memory_vs_seqlen(csv_path, args.metric, args.output)
    print("wrote {}".format(out))


if __name__ == "__main__":
    main()
