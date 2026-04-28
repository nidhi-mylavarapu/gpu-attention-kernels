"""
Plot banded window attention vs naive (and flash) across multiple window sizes.
Reads CSVs from results/, writes PNGs to results/plots/.
"""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
PLOT_DIR = RESULTS_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "seq_len": int(r["seq_len"]),
                "mean_ms": float(r["mean_ms"]),
                "peak_MB": float(r["peak_MB"]),
            })
    rows.sort(key=lambda x: x["seq_len"])
    return rows


def find_csvs():
    csvs = {}
    for p in sorted(RESULTS_DIR.glob("*.csv")):
        csvs[p.stem] = p
    return csvs


WINDOW_SIZES = [64, 128, 256, 512, 1024]
COLORS = {
    64: "tab:blue",
    128: "tab:cyan",
    256: "tab:green",
    512: "tab:orange",
    1024: "tab:red",
}


def plot_speedup(csvs):
    if "naive" not in csvs:
        print("skipping speedup: no naive.csv")
        return
    naive = {r["seq_len"]: r["mean_ms"] for r in load_csv(csvs["naive"])}

    plt.figure(figsize=(10, 6))
    plt.axhline(1.0, linestyle="--", color="black", label="naive baseline")

    if "flash" in csvs:
        rows = load_csv(csvs["flash"])
        xs = [r["seq_len"] for r in rows if r["seq_len"] in naive]
        ys = [naive[r["seq_len"]] / r["mean_ms"] for r in rows if r["seq_len"] in naive]
        plt.plot(xs, ys, marker="o", linewidth=2,
                 color="tab:purple", label="flash")

    for w in WINDOW_SIZES:
        key = f"banded_window_w{w}"
        if key not in csvs:
            continue
        rows = load_csv(csvs[key])
        xs = [r["seq_len"] for r in rows if r["seq_len"] in naive]
        ys = [naive[r["seq_len"]] / r["mean_ms"] for r in rows if r["seq_len"] in naive]
        plt.plot(xs, ys, marker="D", linewidth=2,
                 color=COLORS[w], label=f"banded window, w={w}")

    plt.xscale("log", base=2)
    plt.xlabel("Sequence length")
    plt.ylabel("Speedup vs naive (×)")
    plt.title("Attention speedup vs naive")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out = PLOT_DIR / "speedup_vs_naive.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"wrote {out}")


def plot_memory(csvs):
    if "naive" not in csvs:
        print("skipping memory: no naive.csv")
        return
    naive = {r["seq_len"]: r["peak_MB"] for r in load_csv(csvs["naive"])}

    plt.figure(figsize=(10, 6))
    plt.axhline(1.0, linestyle="--", color="black", label="naive baseline")

    if "flash" in csvs:
        rows = load_csv(csvs["flash"])
        xs = [r["seq_len"] for r in rows if r["seq_len"] in naive]
        ys = [naive[r["seq_len"]] / r["peak_MB"] for r in rows if r["seq_len"] in naive]
        plt.plot(xs, ys, marker="o", linewidth=2,
                 color="tab:purple", label="flash")

    for w in WINDOW_SIZES:
        key = f"banded_window_w{w}"
        if key not in csvs:
            continue
        rows = load_csv(csvs[key])
        xs = [r["seq_len"] for r in rows if r["seq_len"] in naive]
        ys = [naive[r["seq_len"]] / r["peak_MB"] for r in rows if r["seq_len"] in naive]
        plt.plot(xs, ys, marker="D", linewidth=2,
                 color=COLORS[w], label=f"banded window, w={w}")

    plt.xscale("log", base=2)
    plt.xlabel("Sequence length")
    plt.ylabel("Memory reduction vs naive (×)")
    plt.title("Attention memory reduction vs naive")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out = PLOT_DIR / "memory_reduction_vs_naive.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    csvs = find_csvs()
    if not csvs:
        raise SystemExit(f"No CSVs found in {RESULTS_DIR}")
    print(f"Found CSVs: {sorted(csvs.keys())}")
    plot_speedup(csvs)
    plot_memory(csvs)
    print(f"\nPlots in {PLOT_DIR}")