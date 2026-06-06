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

    # Plot 3: Local waterfall - one sample per class
    for class_idx, class_name in enumerate(model.classes_):
        true_label_mask = (y_test == class_name).values
        sample_idx = np.where(true_label_mask)[0][0]

        explanation = shap.Explanation(
            values      = shap_values[sample_idx, :, class_idx],
            base_values = explainer.expected_value[class_idx],
            data        = X_test.iloc[sample_idx].values,
            feature_names = X_test.columns.tolist()
        )
        shap.waterfall_plot(explanation, show=False)
        plt.gcf().axes[0].set_title(f"Local explanation - predicted class: {class_name}")
        plt.tight_layout()
        plt.savefig(f"report/shap_waterfall_{class_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> report/shap_waterfall_{class_name}.png")

    # Plot 4: SHAP dependence for top feature
    class_idx = 0
    mean_abs_shap = np.abs(shap_values[:, :, class_idx]).mean(axis=0)
    top_feature_idx = np.argmax(mean_abs_shap)
    top_feature = X_test.columns[top_feature_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        X_test.iloc[:, top_feature_idx].values,
        shap_values[:, top_feature_idx, class_idx],
        c=X_test.iloc[:, top_feature_idx].values,
        cmap="coolwarm", alpha=0.6, s=10
    )
    ax.set_xlabel(top_feature)
    ax.set_ylabel(f"SHAP value ({model.classes_[class_idx]})")
    ax.set_title(f"SHAP Dependence - {top_feature} (class: {model.classes_[class_idx]})")
    plt.colorbar(sc, label="Feature value")
    plt.tight_layout()
    plt.savefig("report/shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved -> report/shap_dependence.png")

    print("\nAll SHAP plots generated successfully.")
