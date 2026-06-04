# Explainable AI for Cryptographic Behaviour Classification

**Subject:** Cryptography & Network Security  
**Date:** June 2026

---

## 1. Introduction

### Problem Statement

Classifying cryptographic algorithms from ciphertext alone is a fundamental challenge in network security and traffic analysis. Traditional machine learning models can achieve high accuracy on this task, but they operate as black boxes — offering no insight into *why* a particular classification was made. This project applies Explainable AI (XAI) techniques to make cryptographic classification transparent and interpretable.

### Motivation for XAI in Cryptography

In real-world security operations, analysts need not only accurate classifications but also understandability. Knowing that a traffic sample was classified as AES encryption is useful; knowing *which statistical features* drove that decision is far more valuable for threat investigation, anomaly detection, and model validation. SHAP (SHapley Additive exPlanations) provides a principled, game-theoretic framework for attributing each prediction to its constituent features.

### Scope

This project generates synthetic ciphertext samples from four classes — AES, DES, RSA, and plaintext — extracts 12 statistical features from raw bytes, trains a Random Forest classifier, and then uses SHAP to explain both global feature importance and individual predictions.

---

## 2. Background

### AES (Advanced Encryption Standard)

AES is a symmetric block cipher operating on 128-bit blocks with key sizes of 128, 192, or 256 bits. In this project, AES-CBC (Cipher Block Chaining) mode with 128-bit keys is used. AES produces ciphertext with near-uniform byte distribution and high entropy (~8 bits), making it statistically indistinguishable from random data.

### DES (Data Encryption Standard)

DES is a symmetric block cipher operating on 64-bit blocks with a 56-bit effective key size. DES-CBC mode with 64-bit keys is used here. DES produces output statistically similar to AES — both generate near-uniform byte distributions — making AES-vs-DES the hardest classification boundary.

### RSA

RSA is an asymmetric cipher. With a 2048-bit key, RSA-OAEP encryption produces 256-byte ciphertext blocks. Unlike AES/DES, RSA's fixed block structure and padding scheme create subtle statistical differences that aid classification.

### What is XAI?

Explainable AI refers to methods that make machine learning model decisions interpretable to humans. Rather than simply reporting a prediction, XAI techniques decompose the decision into feature-level contributions, enabling analysts to understand, trust, and validate model behaviour.

### SHAP (SHapley Additive exPlanations)

SHAP assigns each feature a contribution value for every prediction. Based on Shapley values from cooperative game theory, SHAP provides: (a) local explanations — why a specific sample received a specific label; (b) global explanations — which features are most important across all predictions; (c) consistency guarantees that simpler attribution methods lack.

### Why Random Forest?

Random Forest is chosen because it handles non-linear feature interactions well, requires minimal hyperparameter tuning, provides built-in feature importance measures, and is directly compatible with SHAP's TreeExplainer for exact Shapley value computation.

---

## 3. Methodology

### 3.1 Dataset Generation

A synthetic dataset of 2000 samples (500 per class) was generated using the `pycryptodome` library:

- **AES**: Random 256-byte plaintext encrypted with AES-CBC (128-bit random key, random IV). Ciphertext includes 16-byte IV + padded encrypted blocks.
- **DES**: Random 256-byte plaintext encrypted with DES-CBC (64-bit random key, random IV). Ciphertext includes 8-byte IV + padded encrypted blocks.
- **RSA**: Random 190-byte plaintext encrypted with RSA-OAEP (2048-bit key). Produces fixed 256-byte ciphertext.
- **Plaintext**: Mixed structured data including English text fragments, JSON-like structures, binary headers, and partial random bytes — representing realistic network payloads.

### 3.2 Feature Engineering

12 statistical features were extracted from each raw byte sequence:

| Feature | Description |
|---|---|
| `entropy` | Shannon entropy (bits per byte, 0–8) |
| `chi_square` | Chi-square statistic against uniform distribution |
| `byte_freq_mean` | Mean of byte frequency histogram |
| `byte_freq_std` | Standard deviation of byte frequencies |
| `byte_freq_max` | Maximum byte frequency |
| `byte_freq_min` | Minimum byte frequency |
| `run_mean` | Mean run length of consecutive identical bytes |
| `run_std` | Standard deviation of run lengths |
| `run_max` | Maximum run length |
| `unique_byte_ratio` | Fraction of possible byte values observed (distinct/256) |
| `zero_byte_ratio` | Proportion of 0x00 bytes |
| `serial_corr` | Pearson correlation between adjacent bytes |

### 3.3 Model Training

A Random Forest classifier with 100 estimators was trained on an 80/20 stratified train-test split. 5-fold cross-validation was used to assess generalisation.

---

## 4. Results

### 4.1 Overall Performance

| Metric | Value |
|---|---|
| Test accuracy | 91% |
| Cross-validation accuracy | 92.05% ± 1.13% |

### 4.2 Per-Class Classification Report

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| AES | 0.98 | 0.98 | 0.98 |
| DES | 0.97 | 0.96 | 0.96 |
| RSA | 0.80 | 0.91 | 0.85 |
| Plaintext | 0.89 | 0.77 | 0.82 |

### 4.3 Confusion Matrix Analysis

The confusion matrix (`report/confusion_matrix.png`) reveals:

- **AES and DES** are classified with near-perfect accuracy (96–98%), despite their statistical similarity. The model successfully exploits subtle differences in block size and IV length.
- **RSA** achieves 91% recall but is occasionally confused with AES/DES due to overlapping entropy ranges.
- **Plaintext** has the lowest recall (77%), frequently misclassified as RSA. This is expected — structured plaintext with binary components can produce entropy values close to RSA ciphertext.

### 4.4 Cross-Validation

5-fold cross-validation yields 92.05% ± 1.13% accuracy, confirming the model generalises well beyond the training set and is not overfitting.

---

## 5. XAI Analysis

### 5.1 Global Feature Importance

The SHAP bar plot (`report/shap_bar.png`) ranks features by mean absolute SHAP value across all classes. Entropy and chi-square dominate, confirming they are the primary drivers of classification.

### 5.2 SHAP Summary (Beeswarm)

The beeswarm plot (`report/shap_summary.png`) shows per-sample feature contributions:

- **Entropy**: High entropy values push predictions toward AES/DES/RSA; low entropy pushes toward plaintext. This is the single most discriminative feature.
- **Chi-square**: Low chi-square values (near 256) indicate uniform distributions characteristic of AES/DES; high values indicate non-uniform plaintext.
- **Unique byte ratio**: High values (>0.6) indicate ciphertext; low values (<0.4) indicate plaintext.

### 5.3 Local Explanations (Waterfall Plots)

Waterfall plots for each class (`report/shap_waterfall_*.png`) reveal the model's decision path:

- **AES sample**: Entropy and chi-square are the top two features pushing toward AES. Run-length features are near baseline.
- **DES sample**: Nearly identical explanation to AES, confirming statistical similarity.
- **RSA sample**: Slightly lower entropy than AES/DES, with byte frequency features contributing more.
- **Plaintext sample**: Entropy is the dominant negative contributor (low entropy → plaintext). Chi-square strongly confirms.

### 5.4 Dependence Analysis

The dependence plot (`report/shap_dependence.png`) shows how entropy values relate to SHAP contributions. A clear threshold around entropy = 6.5 separates plaintext (below) from ciphertext (above).

---

## 6. Limitations

1. **Synthetic data**: The dataset is synthetically generated and does not capture real-world TLS behavior, protocol overhead, or network-level artifacts. Real-world classification would face additional challenges including packet fragmentation, protocol headers, and traffic padding.

2. **AES vs. DES similarity**: Both algorithms produce statistically near-identical byte distributions. The model's ability to distinguish them relies on subtle IV length differences rather than fundamental cryptographic properties — a distinction that would not persist in real network traffic where IVs may be stripped or standardised.

3. **Feature set limitations**: The 12 statistical features capture first-order and second-order byte statistics. Higher-order features (autocorrelation at multiple lags, spectral analysis, n-gram distributions) could improve performance.

4. **No key-size variation**: All AES samples use 128-bit keys; all RSA samples use 2048-bit keys. Variation in key sizes or encryption modes could change the statistical fingerprint.

5. **Model limitations**: Random Forest was chosen for SHAP compatibility. Deep learning models might capture different patterns but would be harder to explain.

---

## 7. Conclusion

This project demonstrates that statistical features extracted from raw ciphertext bytes can classify encryption algorithms with 91% accuracy (92% cross-validated). SHAP analysis reveals that Shannon entropy and chi-square statistics are the dominant features, with entropy alone achieving near-perfect separation between plaintext and ciphertext.

AES and DES are the hardest pair to distinguish — consistent with the theoretical properties of strong block ciphers that produce output indistinguishable from random. The model's ability to separate them relies on structural artifacts (IV length, block padding) rather than the cryptographic output itself.

SHAP waterfall plots provide instance-level transparency, enabling analysts to trace any individual prediction back to its contributing features. This interpretability is essential for deploying such models in security operations where trust and explainability are as important as accuracy.

### Future Scope

- Apply to real network traffic datasets (CICIDS-2017, UNSW-NB15)
- Include additional protocols (ChaCha20, 3DES, Blowfish)
- Explore deep learning models with attention mechanisms for interpretability
- Investigate adversarial robustness — can byte-level perturbations evade classification?

---

## 8. References

1. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30.

2. Paar, C., & Pelzl, J. (2010). *Understanding Cryptography*. Springer.

3. National Institute of Standards and Technology. (2001). *Advanced Encryption Standard (AES)*. FIPS 197.

4. scikit-learn developers. (2024). *Random Forest Classifier*. scikit-learn Documentation.

5. SHAP contributors. (2024). *SHAP: SHapley Additive exPlanations*. https://shap.readthedocs.io/

---

**Project files:**

| File | Purpose |
|---|---|
| `data_generation.py` | Generates AES/DES/RSA/plaintext samples |
| `feature_extraction.py` | Extracts 12 statistical features per sample |
| `model_training.py` | Trains Random Forest, saves confusion matrix |
| `xai_explanation.py` | SHAP analysis — generates all explanation plots |
| `main.ipynb` | Master notebook tying all phases together |
| `report/confusion_matrix.png` | Confusion matrix heatmap |
| `report/shap_summary.png` | SHAP beeswarm summary plot |
| `report/shap_bar.png` | SHAP feature importance bar chart |
| `report/shap_waterfall_AES.png` | Local explanation for AES |
| `report/shap_waterfall_DES.png` | Local explanation for DES |
| `report/shap_waterfall_RSA.png` | Local explanation for RSA |
| `report/shap_waterfall_plaintext.png` | Local explanation for plaintext |
| `report/shap_dependence.png` | Entropy dependence scatter plot |
