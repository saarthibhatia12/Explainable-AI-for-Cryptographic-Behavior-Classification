import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────

def _run_length_stats(arr: np.ndarray):
    """Mean / std / max of run-length encoded consecutive equal bytes."""
    if len(arr) == 0:
        return 0.0, 0.0, 0
    runs, run = [], 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1]:
            run += 1
        else:
            runs.append(run)
            run = 1
    runs.append(run)
    r = np.array(runs)
    return float(r.mean()), float(r.std()), int(r.max())


def _autocorr_at_lag(arr: np.ndarray, lag: int) -> float:
    """Pearson correlation between arr and arr shifted by `lag` bytes."""
    if len(arr) <= lag:
        return 0.0
    a, b = arr[:-lag].astype(float), arr[lag:].astype(float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _bigram_entropy(arr: np.ndarray) -> float:
    """Shannon entropy of 2-byte (bigram) pair frequencies."""
    if len(arr) < 2:
        return 0.0
    # encode each bigram as a single int: hi*256 + lo
    bigrams = arr[:-1].astype(np.int32) * 256 + arr[1:].astype(np.int32)
    counts  = np.bincount(bigrams, minlength=65536)
    total   = counts.sum()
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def _block_variance(arr: np.ndarray, block_size: int) -> float:
    """
    Chop the array into non-overlapping blocks of `block_size` bytes,
    compute the mean byte value of each block, then return the variance
    of those means. Low variance → ciphertext (uniform per-block means);
    high variance → plaintext (uneven content per block).
    """
    n_blocks = len(arr) // block_size
    if n_blocks < 2:
        return 0.0
    blocks = arr[: n_blocks * block_size].reshape(n_blocks, block_size)
    means  = blocks.mean(axis=1)
    return float(means.var())


def _byte_diff_stats(arr: np.ndarray):
    """Stats on the absolute byte-to-byte differences."""
    if len(arr) < 2:
        return 0.0, 0.0
    diffs = np.abs(np.diff(arr.astype(np.int16)))
    # entropy of diff distribution (diffs range 0-255)
    counts = np.bincount(diffs, minlength=256)
    total  = counts.sum()
    p = counts[counts > 0] / total
    diff_entropy = float(-np.sum(p * np.log2(p)))
    return diff_entropy, float(diffs.std())


# ── main feature function ─────────────────────────────────────────────────────

def extract_features(byte_list, seq_len: int = None) -> dict:
    data  = bytes([int(b) for b in byte_list])
    arr   = np.frombuffer(data, dtype=np.uint8)
    total = len(arr)

    # seq_len = original ciphertext length before any truncation.
    # This is a first-class feature: AES outputs 288B, DES 272B, RSA 256B, plaintext 256B.
    # Including it explicitly is correct because packet/block length IS observable in
    # network traffic and it directly encodes IV-size and block-size structure.
    _seq_len = seq_len if seq_len is not None else total

    # --- byte frequency histogram (built once) ---
    counts = np.bincount(arr, minlength=256)
    freqs  = counts / total

    # --- global statistics ---
    entropy  = float(-np.sum(freqs[freqs > 0] * np.log2(freqs[freqs > 0])))
    # Normalise chi-square by N so it measures non-uniformity *per byte*,
    # removing the scaling artefact (chi_square ∝ N) from variable-length sequences.
    chi_sq_norm = float(np.sum((counts - total / 256.0) ** 2 / (total / 256.0))) / total

    # --- run-length ---
    rl_mean, rl_std, rl_max = _run_length_stats(arr)

    # --- serial correlation (lag 1) ---
    a, b = arr[:-1].astype(float), arr[1:].astype(float)
    serial = float(np.corrcoef(a, b)[0, 1]) if (a.std() > 0 and b.std() > 0) else 0.0

    # --- block-structure features (secondary AES vs DES signal) ---
    autocorr_lag8  = _autocorr_at_lag(arr, 8)
    autocorr_lag16 = _autocorr_at_lag(arr, 16)
    block_var_8    = _block_variance(arr, 8)
    block_var_16   = _block_variance(arr, 16)

    # --- higher-order features ---
    bigram_ent             = _bigram_entropy(arr)
    diff_entropy, diff_std = _byte_diff_stats(arr)

    return {
        # ── primary discriminator (AES=288, DES=272, RSA=256, plaintext=256) ──
        "seq_len":           _seq_len,
        # global byte distribution
        "entropy":           entropy,
        "chi_square_norm":   chi_sq_norm,   # per-byte chi-square, length-invariant
        "byte_freq_std":     float(freqs.std()),
        "byte_freq_max":     float(freqs.max()),
        "byte_freq_min":     float(freqs.min()),
        "unique_byte_ratio": float(np.count_nonzero(counts)) / 256.0,
        "zero_byte_ratio":   float(counts[0]) / total,
        # run-length
        "run_mean":          rl_mean,
        "run_std":           rl_std,
        "run_max":           rl_max,
        # serial / lag-1
        "serial_corr":       serial,
        # block-structure
        "autocorr_lag8":     autocorr_lag8,
        "autocorr_lag16":    autocorr_lag16,
        "block_var_8":       block_var_8,
        "block_var_16":      block_var_16,
        # higher-order
        "bigram_entropy":    bigram_ent,
        "diff_entropy":      diff_entropy,
        "diff_std":          diff_std,
    }
