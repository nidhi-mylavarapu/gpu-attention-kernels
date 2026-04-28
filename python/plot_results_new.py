import matplotlib.pyplot as plt
import numpy as np

# Data
seq = np.array([128, 256, 512, 1024, 2048, 4096, 8192])

naive = np.array([0.162, 0.216, 0.346, 0.648, 1.923, 5.663, 17.366])
flash = np.array([0.322, 0.566, 1.059, 2.520, 6.608, 25.916, 99.015])
banded = np.array([0.175, 0.318, 0.607, 1.197, 2.345, 4.685, 9.321])

# Speedups vs naive
flash_speedup = naive / flash
banded_speedup = naive / banded

# Plot
plt.figure(figsize=(10, 6))

plt.plot(seq, flash_speedup, marker='D', linewidth=2, label='flash attention')
plt.plot(seq, banded_speedup, marker='D', linewidth=2, label='banded window')

# Baseline line
plt.axhline(1.0, linestyle='--', color='black', label='naive baseline')

# log2 x-axis labels
plt.xscale('log', base=2)
plt.xticks(seq, [rf'$2^{{{int(np.log2(s))}}}$' for s in seq])

plt.xlabel('Sequence length')
plt.ylabel('Speedup vs naive (×)')
plt.title('Attention speedup vs naive')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("speedup_vs_naive.png", dpi=300)
plt.show()