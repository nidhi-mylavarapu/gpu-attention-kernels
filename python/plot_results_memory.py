import matplotlib.pyplot as plt
import numpy as np

seq = np.array([128, 256, 512, 1024, 2048, 4096, 8192])

naive_peak = np.array([20016.1, 20256.1, 20268.1, 20654.1, 20768.1, 21188.1, 23032.1])
tiled_online_peak = np.array([20842.1, 20844.1, 20850.1, 20858.1, 20876.1, 21266.1, 21810.1])
tiled_online_wmma_peak = np.array([21668.1, 21670.1, 21794.1, 22038.1, 22056.1, 22092.1, 22164.1])
sparse_window_peak = np.array([22024.1, 22026.1, 22034.1, 22046.1, 22072.1, 22124.1, 22464.1])

tiled_online_mem_saving = naive_peak / tiled_online_peak
tiled_online_wmma_mem_saving = naive_peak / tiled_online_wmma_peak
sparse_window_mem_saving = naive_peak / sparse_window_peak

plt.figure(figsize=(10, 6))

plt.plot(seq, tiled_online_mem_saving, marker='D', linewidth=2, label='tiled_online')
plt.plot(seq, tiled_online_wmma_mem_saving, marker='D', linewidth=2, label='tiled_online_wmma')
plt.plot(seq, sparse_window_mem_saving, marker='D', linewidth=2, label='sparse_window')

plt.axhline(1.0, linestyle='--', color='black', label='naive baseline')

plt.xscale('log', base=2)
plt.xticks(seq, [rf'$2^{{{int(np.log2(s))}}}$' for s in seq])

plt.xlabel('Sequence length')
plt.ylabel('Memory reduction vs naive (×)')
plt.title('Attention memory reduction vs naive')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("memory_reduction_vs_naive.png", dpi=300)
plt.show()