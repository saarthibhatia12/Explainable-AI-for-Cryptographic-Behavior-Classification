# Explainable AI for Cryptographic Behaviour Classification
## Phase-by-Phase Implementation Plan

**Subject:** Cryptography & Network Security  
**Project Type:** Academic  
**Stack:** Python, scikit-learn, SHAP, pycryptodome, matplotlib, Jupyter  

---

## Overview

The project is split into 5 phases. Each phase has a clear goal, tasks, and expected output. Complete them in order — each phase feeds into the next.

```
Phase 1 → Data Generation
Phase 2 → Feature Extraction
Phase 3 → Model Training & Evaluation
Phase 4 → XAI with SHAP
Phase 5 → Report & Presentation
```

**Estimated total time:** 6–8 hours (spread across a few days)

---

## Phase 1 — Environment Setup & Data Generation

**Goal:** Set up your project, install dependencies, and generate labelled ciphertext samples from four classes: AES, DES, RSA, and plaintext.

### 1.1 Project folder structure

Create the following layout before writing any code:

```
xai_crypto_project/
├── data_generation.py
├── feature_extraction.py
├── model_training.py
├── xai_explanation.py
├── main.ipynb              ← your main Jupyter notebook
├── data/
│   └── dataset.csv         ← generated in Phase 1
├── models/
│   └── rf_model.pkl        ← saved in Phase 3
└── report/
    ├── confusion_matrix.png
    ├── shap_summary.png
    ├── shap_bar.png
    └── shap_waterfall_AES.png  (and others)
```

### 1.2 Install dependencies

```bash
pip install pycryptodome scikit-learn shap matplotlib seaborn pandas numpy joblib jupyter
```

### 1.3 Data generation code

**File:** `data_generation.py`

```python
import os
import pandas as pd
from Crypto.Cipher import AES, DES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Util.Padding import pad

SAMPLES_PER_CLASS = 500   # increase to 1000 if you want a bigger dataset
PLAINTEXT_SIZE    = 256   # bytes per sample

def encrypt_aes(plaintext: bytes) -> bytes:
    key    = os.urandom(16)                        # 128-bit key
    cipher = AES.new(key, AES.MODE_CBC)
    return cipher.iv + cipher.encrypt(pad(plaintext, AES.block_size))

def encrypt_des(plaintext: bytes) -> bytes:
    key    = os.urandom(8)                         # 64-bit key
    cipher = DES.new(key, DES.MODE_CBC)
    return cipher.iv + cipher.encrypt(pad(plaintext, DES.block_size))

def encrypt_rsa(plaintext: bytes) -> bytes:
    key    = RSA.generate(2048)
    cipher = PKCS1_OAEP.new(key)
    return cipher.encrypt(plaintext[:190])         # RSA max plaintext limit

def generate_dataset(output_path: str = "data/dataset.csv"):
    records = []
    for i in range(SAMPLES_PER_CLASS):
        plaintext = os.urandom(PLAINTEXT_SIZE)

        for label, ciphertext in [
            ("AES",       encrypt_aes(plaintext)),
            ("DES",       encrypt_des(plaintext)),
            ("RSA",       encrypt_rsa(plaintext)),
            ("plaintext", plaintext),
        ]:
            records.append({"bytes": list(ciphertext), "label": label})

        if i % 100 == 0:
            print(f"  Generated {i}/{SAMPLES_PER_CLASS} samples per class...")

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved → {output_path}  ({len(df)} total rows)")
    return df

if __name__ == "__main__":
    generate_dataset()
```

### 1.4 Expected output of Phase 1

- `data/dataset.csv` with 2000 rows (500 × 4 classes)
- Each row has a `bytes` column (list of integers 0–255) and a `label` column
- Class distribution is perfectly balanced — 500 samples each

### 1.5 What to note in your report

> The dataset is synthetically generated using the `pycryptodome` library. Each plaintext sample is a 256-byte random sequence independently encrypted with AES-CBC (128-bit key), DES-CBC (64-bit key), and RSA-OAEP (2048-bit key). A plaintext class is also included as a baseline, giving four balanced classes of 500 samples each.

---

## Phase 2 — Feature Extraction

**Goal:** Convert raw ciphertext bytes into a fixed set of numerical features that capture the statistical "fingerprint" of each encryption algorithm.

### 2.1 Why these features?

| Feature | What it captures | Why it matters |
|---|---|---|
| Shannon entropy | Randomness (0–8 bits) | Ciphertext is close to 8; plaintext is lower |
| Chi-square score | Deviation from uniform distribution | AES/DES score very close to 256; RSA differs slightly |
| Byte frequency stats | Mean, std, max, min of byte histogram | AES and DES are near-uniform; plaintext has clear peaks |
| Run-length stats | Consecutive repeated bytes (mean, std, max) | Plaintext has long runs; ciphertext does not |
| Unique byte ratio | Distinct bytes / 256 | High for ciphertext, variable for plaintext |
| Zero-byte ratio | Proportion of 0x00 bytes | Useful separator between classes |
| Serial correlation | How much each byte predicts the next | Low for good ciphertext; higher for plaintext |

### 2.2 Feature extraction code

**File:** `feature_extraction.py`

```python
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
    # byte_list may be a Python list or a numpy array of ints
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
```

### 2.3 Build the feature DataFrame

Add this to `main.ipynb` (Cell 2):

```python
import ast, pandas as pd
from feature_extraction import extract_features

raw_df = pd.read_csv("data/dataset.csv")

records = []
for _, row in raw_df.iterrows():
    byte_list = ast.literal_eval(row["bytes"])   # convert string back to list
    feats     = extract_features(byte_list)
    feats["label"] = row["label"]
    records.append(feats)

feature_df = pd.DataFrame(records)
feature_df.to_csv("data/features.csv", index=False)
print(feature_df.head())
print(feature_df["label"].value_counts())
```

### 2.4 Expected output of Phase 2

- `data/features.csv` with 2000 rows and 13 columns (12 features + label)
- No missing values — all features are computable from any byte sequence
- You should see clear separation when you `describe()` by class

### 2.5 Quick sanity check (run in notebook)

```python
feature_df.groupby("label")[["entropy", "chi_square", "unique_byte_ratio"]].mean()
```

Expected pattern:

| label | entropy | chi_square | unique_byte_ratio |
|---|---|---|---|
| AES | ~7.99 | ~245 | ~0.99 |
| DES | ~7.98 | ~248 | ~0.99 |
| RSA | ~7.90 | ~290 | ~0.97 |
| plaintext | ~5.50 | ~900 | ~0.65 |

---

## Phase 3 — Model Training & Evaluation

**Goal:** Train a Random Forest classifier on the extracted features and evaluate it with standard metrics. Random Forest is chosen because it works well out of the box, handles non-linear relationships, and is directly compatible with SHAP in Phase 4.

### 3.1 Model training code

**File:** `model_training.py`

```python
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

def train_and_evaluate(feature_df: pd.DataFrame):
    X = feature_df.drop("label", axis=1)
    y = feature_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Train ---
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # --- Cross-validation (5-fold) ---
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # --- Test set evaluation ---
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        cmap="Blues"
    )
    plt.title("Confusion Matrix — Cryptographic Algorithm Classifier")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("report/confusion_matrix.png", dpi=150)
    plt.show()
    print("Saved → report/confusion_matrix.png")

    # --- Save model ---
    joblib.dump(model, "models/rf_model.pkl")
    print("Model saved → models/rf_model.pkl")

    return model, X_train, X_test, y_train, y_test
```

### 3.2 Call it in the notebook (Cell 3)

```python
from model_training import train_and_evaluate
import pandas as pd

feature_df = pd.read_csv("data/features.csv")
model, X_train, X_test, y_train, y_test = train_and_evaluate(feature_df)
```

### 3.3 What good results look like

With 12 well-chosen features and 2000 balanced samples, expect:

- Overall accuracy: **90–98%**
- Plaintext class: near-perfect recall (it's very different from ciphertext)
- AES vs DES: the hardest pair — they are statistically very similar; some misclassification is expected and is itself a discussion point

### 3.4 What to discuss in your report

- Why does AES get confused with DES? (Both produce near-uniform byte distributions)
- Why is RSA slightly easier to distinguish? (Block structure and key-size artifacts differ)
- Is the high accuracy "real"? (Yes, for synthetic data. Real-world performance would be lower — mention this as a limitation)

---

## Phase 4 — XAI with SHAP

**Goal:** Use SHAP (SHapley Additive exPlanations) to explain the model's predictions — both globally (which features matter most across all samples) and locally (why the model predicted a specific label for a specific sample).

### 4.1 What SHAP does

SHAP assigns each feature a value for each prediction. A positive SHAP value means that feature pushed the prediction toward this class. A negative value means it pushed away. The sum of all SHAP values equals the difference between the prediction and the average prediction.

### 4.2 XAI explanation code

**File:** `xai_explanation.py`

```python
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt

def run_shap_analysis(model, X_train, X_test):
    print("Computing SHAP values (this may take 1–2 minutes)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)   # shape: [n_classes, n_samples, n_features]

    # ── Plot 1: Global summary — beeswarm ──────────────────────────────────
    shap.summary_plot(
        shap_values, X_test,
        class_names=model.classes_,
        show=False
    )
    plt.title("SHAP Summary — Feature Impact per Class")
    plt.tight_layout()
    plt.savefig("report/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → report/shap_summary.png")

    # ── Plot 2: Global bar — mean |SHAP| per feature ───────────────────────
    shap.summary_plot(
        shap_values, X_test,
        plot_type="bar",
        class_names=model.classes_,
        show=False
    )
    plt.title("SHAP Feature Importance (Mean |SHAP Value|)")
    plt.tight_layout()
    plt.savefig("report/shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → report/shap_bar.png")

    # ── Plot 3: Local waterfall — one sample per class ─────────────────────
    for class_idx, class_name in enumerate(model.classes_):
        # Pick a correctly classified sample of this class
        true_label_mask = (y_test == class_name).values   # requires y_test in scope
        sample_idx = np.where(true_label_mask)[0][0]

        explanation = shap.Explanation(
            values      = shap_values[class_idx][sample_idx],
            base_values = explainer.expected_value[class_idx],
            data        = X_test.iloc[sample_idx].values,
            feature_names = X_test.columns.tolist()
        )
        shap.waterfall_plot(explanation, show=False)
        plt.title(f"Local explanation — predicted class: {class_name}")
        plt.tight_layout()
        plt.savefig(f"report/shap_waterfall_{class_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → report/shap_waterfall_{class_name}.png")

    # ── Plot 4: SHAP dependence plot for top feature ───────────────────────
    top_feature = X_test.columns[
        np.argmax([np.abs(shap_values[i]).mean(0).max() for i in range(len(model.classes_))])
    ]
    shap.dependence_plot(
        top_feature, shap_values[0], X_test,
        show=False
    )
    plt.title(f"SHAP Dependence — {top_feature} (class: {model.classes_[0]})")
    plt.tight_layout()
    plt.savefig("report/shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → report/shap_dependence.png")

    print("\nAll SHAP plots generated successfully.")
```

### 4.3 Call it in the notebook (Cell 4)

```python
from xai_explanation import run_shap_analysis

run_shap_analysis(model, X_train, X_test)
```

### 4.4 How to read each plot

**Summary plot (beeswarm):**  
Each dot is one sample. Position on x-axis = SHAP value (positive → pushed toward this class). Color = feature value (red = high, blue = low). Wide spread = high impact feature.

**Bar plot:**  
Average absolute SHAP value per feature across all classes. Your "feature importance ranking." Entropy and chi_square will almost certainly top this.

**Waterfall plot:**  
For one sample, shows how each feature moved the model's output from the baseline (average prediction) to the final prediction. Read top-to-bottom. Red bars push toward the predicted class; blue bars push away.

**Dependence plot:**  
Shows how a single feature's value affects its SHAP contribution, with color encoding a second interacting feature.

### 4.5 Key findings to highlight

When you run the plots, look for and write about these:

- Entropy is the single most important feature for separating plaintext from ciphertext
- Chi-square and unique_byte_ratio dominate the AES vs RSA boundary
- For the AES vs DES boundary, SHAP values will be small — confirming these two are statistically similar
- The waterfall plots show the "reasoning path" of the model for each class

---

## Phase 5 — Report & Presentation

**Goal:** Document your project clearly with methodology, results, and analysis. This phase is about turning your code output into academic deliverables.

### 5.1 Report sections

Structure your written report as follows:

```
1. Introduction
   - Problem statement
   - Motivation for XAI in cryptography
   - Scope of the project

2. Background
   - AES, DES, RSA — brief algorithm overview
   - What is XAI? What is SHAP?
   - Why Random Forest for this task?

3. Methodology
   - Dataset generation (Phase 1)
   - Feature engineering (Phase 2)
   - Model training and evaluation (Phase 3)

4. Results
   - Accuracy, precision, recall, F1
   - Confusion matrix analysis
   - Cross-validation results

5. XAI Analysis (core section)
   - Global feature importance (SHAP bar + summary plots)
   - Local explanations (waterfall plots per class)
   - Insight: what features distinguish each algorithm?

6. Limitations
   - Synthetic data does not capture real-world TLS behavior
   - AES and DES are statistically very similar — harder to distinguish
   - Feature set can be extended (autocorrelation, spectral analysis, etc.)

7. Conclusion
   - Summary of findings
   - Future scope: apply to network traffic datasets (CICIDS-2017)

8. References
```

### 5.2 Figures checklist

Include at minimum these figures in your report:

- [ ] `confusion_matrix.png` — model accuracy across classes
- [ ] `shap_summary.png` — global beeswarm feature importance
- [ ] `shap_bar.png` — ranked feature importance bar chart
- [ ] `shap_waterfall_AES.png` — local explanation for an AES sample
- [ ] `shap_waterfall_plaintext.png` — local explanation for a plaintext sample
- [ ] `shap_dependence.png` — entropy dependence plot

### 5.3 Key academic claims you can confidently make

Based on your results, you should be able to support all of these:

- "Shannon entropy alone achieves near-perfect separation of plaintext from ciphertext"
- "AES and DES are the hardest to distinguish because both produce near-uniform byte distributions — consistent with the theoretical properties of strong block ciphers"
- "SHAP waterfall plots provide instance-level transparency, allowing us to explain individual model predictions in human-interpretable terms"
- "The model achieves X% accuracy with 5-fold cross-validation, indicating it generalises well beyond the training set"

### 5.4 Final notebook structure (`main.ipynb`)

Your notebook should have clean, labeled cells in this order:

| Cell | Title | What it does |
|---|---|---|
| 1 | Data Generation | Calls `generate_dataset()`, shows class counts |
| 2 | Feature Extraction | Builds `features.csv`, shows `describe()` by class |
| 3 | Sanity Check | `groupby` entropy/chi-square table to verify separation |
| 4 | Model Training | Calls `train_and_evaluate()`, shows confusion matrix |
| 5 | SHAP Global | Summary + bar plots |
| 6 | SHAP Local | Waterfall plots for AES and plaintext |
| 7 | Discussion | Markdown cell with your written analysis |

---

## Quick Reference — Complete File List

| File | Created in | Purpose |
|---|---|---|
| `data_generation.py` | Phase 1 | Generates AES/DES/RSA/plaintext samples |
| `feature_extraction.py` | Phase 2 | Extracts 12 statistical features per sample |
| `model_training.py` | Phase 3 | Trains Random Forest, saves confusion matrix |
| `xai_explanation.py` | Phase 4 | SHAP analysis — 4 types of plots |
| `main.ipynb` | All phases | Master notebook tying everything together |
| `data/dataset.csv` | Phase 1 output | Raw bytes + labels |
| `data/features.csv` | Phase 2 output | Numerical features + labels |
| `models/rf_model.pkl` | Phase 3 output | Saved trained model |
| `report/*.png` | Phase 3–4 output | All evaluation and SHAP plots |

---

## Timeline Suggestion

| Day | Work |
|---|---|
| Day 1 | Phase 1 — setup, install, generate dataset |
| Day 2 | Phase 2 — feature extraction, sanity checks |
| Day 3 | Phase 3 — train model, evaluate, confusion matrix |
| Day 4 | Phase 4 — SHAP analysis, generate all plots |
| Day 5 | Phase 5 — write report, clean up notebook |

---

*Project title: Explainable AI for Cryptographic Behaviour Classification*  
*Subject: Cryptography & Network Security*
