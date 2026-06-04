import math
import numpy as np
from collections import Counter

def shannon_entropy(data: bytes) -> float:
    counts = Counter(data)
    total  = len(data)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def chi_square(data: bytes) -> float:
    counts   = np.array([Counter(data).get(i, 0) for i in range(256)])
    expected = len(data) / 256.0
    return float(np.sum((counts - expected) ** 2 / expected))

def byte_frequency_stats(data: bytes):
    freqs = np.array([Counter(data).get(i, 0) for i in range(256)]) / len(data)
    return freqs.mean(), freqs.std(), freqs.max(), freqs.min()

def run_length_stats(data: bytes):
    if len(data) == 0:
        return 0.0, 0.0, 0
    runs, run = [], 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            run += 1
        else:
            runs.append(run)
            run = 1
    runs.append(run)
    runs = np.array(runs)
    return float(runs.mean()), float(runs.std()), int(runs.max())

def serial_correlation(data: bytes) -> float:
    if len(data) < 2:
        return 0.0
    arr = np.frombuffer(data, dtype=np.uint8).astype(float)
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])

def extract_features(byte_list) -> dict:
    data = bytes([int(b) for b in byte_list])

    mean_f, std_f, max_f, min_f = byte_frequency_stats(data)
    rl_mean, rl_std, rl_max     = run_length_stats(data)

    return {
        "entropy":          shannon_entropy(data),
        "chi_square":       chi_square(data),
        "byte_freq_mean":   mean_f,
        "byte_freq_std":    std_f,
        "byte_freq_max":    max_f,
        "byte_freq_min":    min_f,
        "run_mean":         rl_mean,
        "run_std":          rl_std,
        "run_max":          rl_max,
        "unique_byte_ratio": len(set(data)) / 256.0,
        "zero_byte_ratio":  data.count(0) / len(data),
        "serial_corr":      serial_correlation(data),
    }
