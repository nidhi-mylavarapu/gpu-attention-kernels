"""
Grouped bar chart: speedup vs sequence length.

Speedup = baseline_mean_ms / method_mean_ms (higher is better). Default baseline
is naive_cublas. Only sequence lengths where both baseline and a method have data
are plotted for that method.

Requires matplotlib.

Usage (from repo root)::

  python3 python/plot_speedup_bars.py
  python3 python/plot_speedup_bars.py --csv results.csv -o figures/speedup.png
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# (impl key, bar label, color) — excludes baseline (plotted implicitly as 1×)
SPEEDUP_SERIES = [
    ("tiled_online", "Tiled online", "#4292c6"),
    ("tiled_online_wmma", "Tiled online (WMMA)", "#2171b5"),
    ("flash_attn_official", "FlashAttention-2", "#084594"),
    ("sparse_window", "Sparse window", "#ea580c"),
]


def _default_csv_path() -> Path:
    cand = ROOT / "results.csv"
    if cand.is_file():
        return cand
    alt = ROOT / "results" / "results.csv"
    if alt.is_file():
        return alt
    return cand


def load_by_impl(csv_path: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Return impl -> [(seq_len, mean_ms), ...] sorted by seq_len."""
    by_impl = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            impl = row["impl"].strip()
            seq_len = int(row["seq_len"])
            mean_ms = float(row["mean_ms"])
            by_impl[impl].append((seq_len, mean_ms))
    for k in by_impl:
        by_impl[k].sort(key=lambda t: t[0])
    return dict(by_impl)


def build_speedups(
    data,
    baseline_key="naive_cublas",
):  # type: (...) -> Tuple[List[int], List[Tuple[str, str, List[Optional[float]]]]]
    """
    Returns seq_lens_sorted, list of (impl_key, label, speedups_aligned).
    Speedup entries are float or None if missing data for that (seq_len, impl).
    """
    if baseline_key not in data:
        raise SystemExit(
            "Baseline '{}' not present in CSV (need naive timings for speedup)".format(
                baseline_key
            )
        )

    naive_map = dict(data[baseline_key])  # type: Dict[int, float]

    seq_set = set(naive_map.keys())
    for impl_key, _label, _c in SPEEDUP_SERIES:
        rows = data.get(impl_key)
        if rows:
            seq_set |= set(s for s, _ in rows)

    seq_lens = sorted(seq_set)

    rows_out = []  # type: List[Tuple[str, str, str, List[Optional[float]]]]
    for impl_key, label, color in SPEEDUP_SERIES:
        meth_map = dict(data.get(impl_key) or ())
        spd = []
        for s in seq_lens:
            if s not in naive_map or naive_map[s] <= 0:
                spd.append(None)
                continue
            if s not in meth_map or meth_map[s] <= 0:
                spd.append(None)
                continue
            spd.append(naive_map[s] / meth_map[s])
        rows_out.append((impl_key, label, color, spd))

    return seq_lens, rows_out


def plot_speedup_bars(
    csv_path: Path,
    out_path: Optional[Path] = None,
    baseline_key: str = "naive_cublas",
):  # type: (...) -> Path
    data = load_by_impl(csv_path)
    seq_lens, series_rows = build_speedups(data, baseline_key)

    # Drop trailing seq_lens where nothing has a speedup?
    # Better: filter to seq_lens where at least one bar is non-None
    keep_indices = []
    for i, _s in enumerate(seq_lens):
        if any(sr[3][i] is not None for sr in series_rows):
            keep_indices.append(i)

    if not keep_indices:
        raise SystemExit("No overlapping baseline/method timings to plot.")

    seq_plot = [seq_lens[i] for i in keep_indices]
    series_plot = []
    for impl_key, label, color, spd in series_rows:
        series_plot.append(
            (
                impl_key,
                label,
                color,
                [
                    spd[i] if spd[i] is not None else 0.0
                    for i in keep_indices
                ],
                [spd[i] is not None for i in keep_indices],
            )
        )

    if out_path is None:
        plot_dir = ROOT / "results" / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        out_path = plot_dir / "speedup_vs_seq_len_bars.png"

    n = len(seq_plot)
    n_series = len(series_plot)
    width = 0.8 / max(n_series, 1)
    x = list(range(n))

    fig, ax = plt.subplots(figsize=(max(10, 1.4 * n + 4), 5.8))
    for j, (_, label, color, vals, mask) in enumerate(series_plot):
        offset = width * j - width * (n_series - 1) / 2.0
        bars_x = [_x + offset for _x in x]
        bars_h = [vals[i] if mask[i] else 0 for i in range(n)]

        rects = ax.bar(
            bars_x,
            bars_h,
            width=width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.92,
        )
        # Only show label on bars where we have real data — zero-height placeholder
        for i, rect in enumerate(rects):
            if not mask[i]:
                rect.set_visible(False)
                continue
            h = rect.get_height()
            ax.annotate(
                "{:.2f}x".format(h),
                xy=(rect.get_x() + rect.get_width() / 2.0, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                clip_on=True,
            )

    labels = []
    for s in seq_plot:
        p = 0
        q = s
        while q % 2 == 0 and q > 1:
            p += 1
            q //= 2
        if q == 1 and p >= 4:
            labels.append(r"$2^{%d}$" % p)
        else:
            labels.append("{:d}".format(s))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1, label=None)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Speedup vs {}".format(baseline_key))
    baseline_pretty = "naive cuBLAS" if baseline_key == "naive_cublas" else baseline_key
    ax.set_title("Speedup relative to {}".format(baseline_pretty))
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("Speedup plotted for sequence lengths:", seq_plot)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Bar chart of speedup vs naive at each sequence length."
    )
    ap.add_argument("--csv", type=Path, default=None, help="Benchmark CSV path")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: results/plots/speedup_vs_seq_len_bars.png)",
    )
    ap.add_argument(
        "--baseline",
        default="naive_cublas",
        help="impl column used as baseline denominator (default: naive_cublas)",
    )
    args = ap.parse_args()
    csv_path = args.csv if args.csv is not None else _default_csv_path()
    if not csv_path.is_file():
        raise SystemExit("CSV not found: {}".format(csv_path))

    out = plot_speedup_bars(csv_path, args.output, baseline_key=args.baseline)
    print("wrote {}".format(out))


if __name__ == "__main__":
    main()
