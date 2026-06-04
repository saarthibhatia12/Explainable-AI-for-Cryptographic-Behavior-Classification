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

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        cmap="Blues"
    )
    plt.title("Confusion Matrix - Cryptographic Algorithm Classifier")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("report/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved -> report/confusion_matrix.png")

    joblib.dump(model, "models/rf_model.pkl")
    print("Model saved -> models/rf_model.pkl")

    return model, X_train, X_test, y_train, y_test
