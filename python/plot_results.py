"""
Plot benchmark results from results/*.csv

Reads all relevant CSVs, generates comparison plots in results/plots/.
Run from repo root or from python/ directory; auto-detects.
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Locate results directory (works whether you run from repo/ or python/)
# -------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
PLOT_DIR = RESULTS_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def load_csv(path: Path):
    """Return list of (seq_len, mean_ms) tuples, sorted by seq_len."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((int(r["seq_len"]), float(r["mean_ms"])))
    rows.sort()
    return rows


def find_csvs():
    """Return dict of label -> Path for every relevant CSV in results/."""
    csvs = {}
    for p in sorted(RESULTS_DIR.glob("*.csv")):
        csvs[p.stem] = p   # e.g. 'naive', 'sparse_window_w64'
    return csvs


# -------------------------------------------------------------------
# Plot 1: all kernels on one axes, mean_ms vs seq_len, log-log
# -------------------------------------------------------------------
def plot_all(csvs):
    plt.figure(figsize=(8, 5.5))

    order = []
    if "naive" in csvs: order.append(("naive", "naive (full attention)", "k", "-", "o"))
    if "tiled" in csvs: order.append(("tiled", "tiled", "tab:gray", "-", "s"))
    if "tiled_online" in csvs: order.append(("tiled_online", "tiled_online", "tab:purple", "-", "^"))

    sw_colors = {"64": "tab:blue", "128": "tab:cyan",
                 "256": "tab:green", "1024": "tab:red"}
    for w in [64, 128, 256, 1024]:
        key = f"sparse_window_w{w}"
        if key in csvs:
            order.append((key, f"sparse window, w={w}",
                          sw_colors[str(w)], "-", "D"))

    for key, label, color, ls, marker in order:
        rows = load_csv(csvs[key])
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        plt.plot(xs, ys, label=label, color=color,
                 linestyle=ls, marker=marker, markersize=5, linewidth=1.6)

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Sequence length")
    plt.ylabel("Mean runtime (ms)")
    plt.title("Attention kernel runtime vs sequence length")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    out = PLOT_DIR / "all_kernels.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


# -------------------------------------------------------------------
# Plot 2: speedup of sparse_window vs naive
# -------------------------------------------------------------------
def plot_speedup(csvs):
    if "naive" not in csvs:
        print("skipping speedup plot: no naive.csv")
        return

    naive = dict(load_csv(csvs["naive"]))   # seq_len -> ms

    plt.figure(figsize=(8, 5))
    colors = {"64": "tab:blue", "128": "tab:cyan",
              "256": "tab:green", "1024": "tab:red"}

    for w in [64, 128, 256, 1024]:
        key = f"sparse_window_w{w}"
        if key not in csvs:
            continue
        rows = load_csv(csvs[key])
        xs = [r[0] for r in rows if r[0] in naive]
        ys = [naive[r[0]] / r[1] for r in rows if r[0] in naive]
        plt.plot(xs, ys, label=f"sparse window, w={w}",
                 color=colors[str(w)], marker="D", linewidth=1.6)

    plt.axhline(1.0, color="k", linewidth=0.8, linestyle="--",
                label="naive baseline")
    plt.xscale("log", base=2)
    plt.xlabel("Sequence length")
    plt.ylabel("Speedup vs naive (×)")
    plt.title("Sparse window attention speedup vs naive")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(fontsize=9)
    plt.tight_layout()
    out = PLOT_DIR / "sparse_speedup.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


# -------------------------------------------------------------------
# Plot 3: naive vs sparse_window across window sizes
# -------------------------------------------------------------------
def plot_naive_vs_sparse_window(csvs):
    plt.figure(figsize=(8, 5))

    if "naive" in csvs:
        rows = load_csv(csvs["naive"])
        plt.plot([r[0] for r in rows], [r[1] for r in rows],
                 "k-o", label="naive (full attention)", linewidth=1.8)

    sw_colors = {"64": "tab:blue", "128": "tab:cyan",
                 "256": "tab:green", "1024": "tab:red"}
    for w in [64, 128, 256, 1024]:
        key = f"sparse_window_w{w}"
        if key not in csvs:
            continue
        rows = load_csv(csvs[key])
        plt.plot([r[0] for r in rows], [r[1] for r in rows],
                 color=sw_colors[str(w)], linestyle="-", marker="D",
                 label=f"sparse window, w={w}", linewidth=1.6)

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Sequence length")
    plt.ylabel("Mean runtime (ms)")
    plt.title("Sparse window attention runtime vs naive")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(fontsize=9)
    plt.tight_layout()
    out = PLOT_DIR / "naive_vs_sparse.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    csvs = find_csvs()
    if not csvs:
        raise SystemExit(f"No CSVs found in {RESULTS_DIR}")
    print(f"Found CSVs: {sorted(csvs.keys())}")
    plot_all(csvs)
    plot_speedup(csvs)
    plot_naive_vs_sparse_window(csvs)
    print(f"\nAll plots written to {PLOT_DIR}")
