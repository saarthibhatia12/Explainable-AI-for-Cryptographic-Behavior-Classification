# Code Review — Cryptographic Behaviour Classifier

Issues are ranked by severity: 🔴 Critical → 🟡 Moderate → 🟢 Minor.

---

## 🔴 Critical Issues

### 1. Variable-Length Ciphertext Is Pure Data Leakage
**File:** `data_generation.py` | **Lines:** 36–50

Each algorithm produces a different byte-sequence length:

| Class     | Size (bytes) | Reason |
|-----------|-------------|--------|
| AES       | 288          | 16-byte IV + 256 bytes padded to 272 |
| DES       | 272          | 8-byte IV + 256 bytes padded to 264  |
| RSA       | 256          | Fixed OAEP output                    |
| Plaintext | 256          | Raw input                            |

The model doesn't see raw length directly, but `chi_square` scales with sample size ($\chi^2 \propto N$), and `unique_byte_ratio` is biased upward for longer sequences. **The classifier is partly learning lengths, not cryptographic properties.**

**Fix:** Truncate or slice every ciphertext to a fixed length (e.g., 256 bytes) before saving:

```python
# in data_generation.py — the records.append line, line 63
records.append({"bytes": list(ciphertext[:256]), "label": label})
```

---

### 2. `Counter` Rebuilt 3× Per Sample — Major Performance Bug
**File:** `feature_extraction.py` | **Lines:** 6, 11, 16

The byte-frequency histogram (`Counter(data)`) is computed three separate times per sample inside `shannon_entropy`, `chi_square`, and `byte_frequency_stats`. For 2,000 samples, this constructs 6,000 unnecessary `Counter` objects, each scanning 256–288 bytes.

**Fix:** Build the histogram once and pass it in:

```python
def extract_features(byte_list) -> dict:
    data = bytes([int(b) for b in byte_list])
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    total  = len(data)
    freqs  = counts / total

    entropy = -np.sum(freqs[freqs > 0] * np.log2(freqs[freqs > 0]))
    expected = total / 256.0
    chi_sq  = float(np.sum((counts - expected) ** 2 / expected))
    
    rl_mean, rl_std, rl_max = run_length_stats(data)
    arr = np.frombuffer(data, dtype=np.uint8).astype(float)
    if arr[:-1].std() == 0 or arr[1:].std() == 0:
        serial = 0.0
    else:
        serial = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])

    return {
        "entropy":           float(entropy),
        "chi_square":        chi_sq,
        "byte_freq_std":     float(freqs.std()),
        "byte_freq_max":     float(freqs.max()),
        "byte_freq_min":     float(freqs.min()),
        "run_mean":          rl_mean,
        "run_std":           rl_std,
        "run_max":           rl_max,
        "unique_byte_ratio": float(np.count_nonzero(counts)) / 256.0,
        "zero_byte_ratio":   float(counts[0]) / total,
        "serial_corr":       serial,
    }
```

Note: `byte_freq_mean` is also intentionally **dropped** here (see Issue #5 below).

---

### 3. `serial_correlation` Can Return `NaN`
**File:** `feature_extraction.py` | **Line:** 37

If a sample has constant or near-constant bytes (e.g., all zeros, repetitive plaintext patterns), `np.corrcoef` returns `NaN` due to division by zero variance. `NaN` silently propagates into the DataFrame and corrupts model training — scikit-learn's Random Forest will silently ignore NaN features in some configurations but produce unreliable splits.

**Fix:**
```python
def serial_correlation(data: bytes) -> float:
    if len(data) < 2:
        return 0.0
    arr = np.frombuffer(data, dtype=np.uint8).astype(float)
    if arr[:-1].std() == 0.0 or arr[1:].std() == 0.0:
        return 0.0   # perfectly uniform or constant — no correlation
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
```

---

## 🟡 Moderate Issues

### 4. Single RSA Key Used for All 500 Samples
**File:** `data_generation.py` | **Line:** 46

```python
_rsa_key = RSA.generate(2048)  # ← generated once at import time
```

All 500 RSA ciphertexts are encrypted with the same public key. RSA-OAEP includes random padding (MGF1), so the outputs look random, but the fixed modulus $N$ introduces subtle structural biases (e.g., the ciphertext is always $< N$, the most-significant-byte distribution is constrained). The model may overfit to one key's modulus artifacts.

**Fix:** Generate a small pool of keys and rotate:
```python
_RSA_KEYS = [RSA.generate(2048) for _ in range(10)]

def encrypt_rsa(plaintext: bytes) -> bytes:
    key = random.choice(_RSA_KEYS)
    cipher = PKCS1_OAEP.new(key)
    return cipher.encrypt(plaintext[:190])
```

---

### 5. `byte_freq_mean` Is a Mathematically Constant Feature
**File:** `feature_extraction.py` | **Lines:** 16, 48

The mean of the 256-bin byte frequency histogram is always exactly $\frac{1}{256} = 0.00390625$ regardless of the input. It has zero variance across all samples and provides **zero information** to the classifier. It wastes a split evaluation at every tree node.

**Fix:** Delete the `"byte_freq_mean"` entry from the `extract_features` return dict.

---

### 6. Feature Extraction Loop Uses Slow `iterrows`
**File:** `main.ipynb` | **Cell: Phase 2**

```python
for _, row in raw_df.iterrows():
    byte_list = ast.literal_eval(row['bytes'])
    feats = extract_features(byte_list)
```

`iterrows()` is one of the slowest ways to iterate a DataFrame in pandas — it creates a new `Series` object per row. For 2,000 rows this is tolerable, but it stacks on top of the `Counter` bug (Issue #2).

**Fix:** Use `raw_df.apply(...)` or iterate `raw_df.itertuples()`:
```python
import ast
from feature_extraction import extract_features

records = [
    {**extract_features(ast.literal_eval(row.bytes)), "label": row.label}
    for row in raw_df.itertuples(index=False)
]
```

---

### 7. No `report/` or `models/` Directory Guard in Training Script
**File:** `model_training.py` | **Lines:** 39, 43

```python
plt.savefig("report/confusion_matrix.png", dpi=150)
joblib.dump(model, "models/rf_model.pkl")
```

If these directories don't exist (e.g., first run on a fresh clone), these lines raise `FileNotFoundError` with no helpful message.

**Fix:** Add `os.makedirs` calls:
```python
import os
os.makedirs("report", exist_ok=True)
os.makedirs("models", exist_ok=True)
```

Similarly add this to `xai_explanation.py`.

---

### 8. Cross-Validation Is Done on the Full Dataset Including Test Set
**File:** `model_training.py` | **Line:** 20

```python
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
```

Cross-validation is run on `X` and `y` — the full dataset — **after** the model has already been fitted on `X_train`. This doesn't cause direct leakage (cross_val_score re-trains internally), but it is methodologically wrong: the test split `X_test` should be held out entirely and never included in any validation procedure. Run CV only on `X_train / y_train`:

```python
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
```

---

## 🟢 Minor Issues

### 9. `'data'` Word Appears Twice in Plaintext Word List
**File:** `data_generation.py` | **Line:** 15

```python
words = [b'hello', b'world', b'test', b'data', b'info', b'message',
         b'request', b'response', b'server', b'client', b'user', b'data']  # ← duplicate
```

`b'data'` appears twice, doubling its sampling probability. Not a correctness error, but reflects a copy-paste oversight.

---

### 10. The `xai_explanation.py` Dependence Plot Color Bar Is Mislabeled
**File:** `xai_explanation.py` | **Line:** 71

```python
plt.colorbar(sc, label="Feature importance")
```

The scatter color is mapped to the feature's own value (`X_test.iloc[:, top_feature_idx].values`), not to feature importance. The colorbar label should say **"Feature value"**.

---

### 11. Waterfall Plot Title Is Added After `shap.waterfall_plot` — Silently Overwritten
**File:** `xai_explanation.py` | **Lines:** 48–49

```python
shap.waterfall_plot(explanation, show=False)
plt.title(f"Local explanation - predicted class: {class_name}")
```

SHAP's `waterfall_plot` renders the title internally using `ax.set_title`. Calling `plt.title(...)` afterward adds a *second* figure-level title that may collide visually or be ignored depending on the matplotlib backend. The title visible in the saved images (`"Local explanation - predicted class: AES"`) is correctly applied, but the behavior is fragile.

**Fix:** Pass the title to the underlying axes directly:
```python
shap.waterfall_plot(explanation, show=False)
plt.gcf().axes[0].set_title(f"Local explanation - predicted class: {class_name}")
```

---

## Summary Table

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | 🔴 Critical | `data_generation.py` | Variable ciphertext lengths leak class identity via chi_square/unique_byte_ratio |
| 2 | 🔴 Critical | `feature_extraction.py` | `Counter` rebuilt 3× per sample — major performance bug |
| 3 | 🔴 Critical | `feature_extraction.py` | `serial_correlation` produces `NaN` on constant inputs |
| 4 | 🟡 Moderate | `data_generation.py` | Single RSA key for all 500 samples — overfitting risk |
| 5 | 🟡 Moderate | `feature_extraction.py` | `byte_freq_mean` is constant — zero-information feature |
| 6 | 🟡 Moderate | `main.ipynb` | `iterrows()` is slow; prefer `itertuples()` |
| 7 | 🟡 Moderate | `model_training.py` | No directory guard — crashes on fresh clone |
| 8 | 🟡 Moderate | `model_training.py` | CV uses full dataset including the held-out test split |
| 9 | 🟢 Minor | `data_generation.py` | Duplicate word `b'data'` in plaintext generator |
| 10 | 🟢 Minor | `xai_explanation.py` | Colorbar label says "Feature importance" should say "Feature value" |
| 11 | 🟢 Minor | `xai_explanation.py` | `plt.title()` after `waterfall_plot` is fragile |
