import matplotlib.pyplot as plt
import numpy as np

# Data (representative bench; refresh from CSV after runs)
seq = np.array([128, 256, 512, 1024, 2048, 4096, 8192])

naive = np.array([0.160, 0.215, 0.345, 0.647, 1.924, 5.828, 18.757])
tiled_online = np.array([0.107, 0.194, 0.453, 1.505, 5.281, 20.385, 80.273])
sparse_window = np.array([0.173, 0.315, 0.605, 1.194, 2.344, 4.683, 9.321])

tiled_online_speedup = naive / tiled_online
sparse_window_speedup = naive / sparse_window

plt.figure(figsize=(10, 6))

plt.plot(seq, tiled_online_speedup, marker='D', linewidth=2, label='tiled_online')
plt.plot(seq, sparse_window_speedup, marker='D', linewidth=2, label='sparse window')

plt.axhline(1.0, linestyle='--', color='black', label='naive baseline')

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
