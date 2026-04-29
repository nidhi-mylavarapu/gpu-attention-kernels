import matplotlib.pyplot as plt
import numpy as np

seq = np.array([128, 256, 512, 1024, 2048, 4096, 8192])

naive_peak = np.array([442.1, 446.1, 458.1, 490.1, 604.1, 1024.1, 2632.1])
tiled_online_peak = np.array([442.1, 444.1, 450.1, 458.1, 476.1, 512.1, 584.1])
banded_peak = np.array([444.1, 446.1, 454.1, 466.1, 492.1, 544.1, 648.1])

tiled_online_mem_saving = naive_peak / tiled_online_peak
banded_mem_saving = naive_peak / banded_peak

plt.figure(figsize=(10, 6))

plt.plot(seq, tiled_online_mem_saving, marker='D', linewidth=2, label='tiled_online')
plt.plot(seq, banded_mem_saving, marker='D', linewidth=2, label='banded window')

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
