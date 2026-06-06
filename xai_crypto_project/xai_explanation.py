import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_shap_analysis(model, X_train, X_test, y_test):
    os.makedirs("report", exist_ok=True)
    print("Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # shap_values shape: (n_samples, n_features, n_classes)
    n_samples, n_features, n_classes = shap_values.shape

    # Plot 1 & 2: Global summary beeswarm + bar per class
    for class_idx, class_name in enumerate(model.classes_):
        shap.summary_plot(
            shap_values[:, :, class_idx], X_test,
            show=False
        )
        plt.title(f"SHAP Summary - {class_name}")
        plt.tight_layout()
        plt.savefig(f"report/shap_summary_{class_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> report/shap_summary_{class_name}.png")

        shap.summary_plot(
            shap_values[:, :, class_idx], X_test,
            plot_type="bar",
            show=False
        )
        plt.title(f"SHAP Feature Importance (Mean |SHAP Value|) - {class_name}")
        plt.tight_layout()
        plt.savefig(f"report/shap_bar_{class_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> report/shap_bar_{class_name}.png")

    # Plot 3: Local waterfall — highest-confidence correctly-classified sample per class
    # Rationale: picking the first sample may grab a weakly-classified edge case.
    # The highest-confidence sample gives the clearest, most representative explanation.
    proba = model.predict_proba(X_test)  # shape: (n_samples, n_classes)

    for class_idx, class_name in enumerate(model.classes_):
        # mask: samples whose TRUE label matches class_name
        true_label_mask = (y_test == class_name).values
        candidate_indices = np.where(true_label_mask)[0]

        # among those, pick the one with the highest predicted probability for this class
        candidate_probas  = proba[candidate_indices, class_idx]
        best_local_idx    = np.argmax(candidate_probas)
        sample_idx        = candidate_indices[best_local_idx]
        confidence        = candidate_probas[best_local_idx]

        explanation = shap.Explanation(
            values        = shap_values[sample_idx, :, class_idx],
            base_values   = explainer.expected_value[class_idx],
            data          = X_test.iloc[sample_idx].values,
            feature_names = X_test.columns.tolist()
        )
        shap.waterfall_plot(explanation, show=False)
        plt.gcf().axes[0].set_title(
            f"Local explanation — predicted class: {class_name}  "
            f"(confidence: {confidence:.2f})"
        )
        plt.tight_layout()
        plt.savefig(f"report/shap_waterfall_{class_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> report/shap_waterfall_{class_name}.png  "
              f"(sample #{sample_idx}, p={confidence:.3f})")

    # Plot 4: SHAP dependence — top *continuous* feature for the plaintext class
    # (plaintext has the most interesting continuous variation; avoids discrete seq_len
    #  which only has 3 values and produces a useless 3-column scatter)
    dep_class_name = "plaintext"
    dep_class_idx  = list(model.classes_).index(dep_class_name)
    mean_abs_shap  = np.abs(shap_values[:, :, dep_class_idx]).mean(axis=0)
    sorted_feature_indices = np.argsort(mean_abs_shap)[::-1]

    # Skip features with fewer than 20 unique values (discrete / near-constant)
    top_feature_idx = None
    for fidx in sorted_feature_indices:
        if X_test.iloc[:, fidx].nunique() >= 20:
            top_feature_idx = fidx
            break
    if top_feature_idx is None:
        top_feature_idx = sorted_feature_indices[0]  # fallback

    top_feature = X_test.columns[top_feature_idx]
    x_vals = X_test.iloc[:, top_feature_idx].values
    y_vals = shap_values[:, top_feature_idx, dep_class_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(x_vals, y_vals, c=x_vals, cmap="coolwarm", alpha=0.5, s=12)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel(top_feature, fontsize=12)
    ax.set_ylabel(f"SHAP value (class: {dep_class_name})", fontsize=12)
    ax.set_title(
        f"SHAP Dependence — {top_feature}  (class: {dep_class_name})", fontsize=13
    )
    plt.colorbar(sc, label="Feature value")
    plt.tight_layout()
    plt.savefig("report/shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> report/shap_dependence.png  (feature: {top_feature}, class: {dep_class_name})")

    print("\nAll SHAP plots generated successfully.")
