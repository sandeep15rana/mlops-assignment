import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(
    input_path: str = "data/processed/heart_cleaned.csv",
    output_dir: str = "report/screenshots"
):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(input_path):
        from src.data.download import download_and_clean_data
        df = download_and_clean_data(processed_path=input_path)
    else:
        df = pd.read_csv(input_path)

    print("=== DATASET OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print("\nMissing values:\n", df.isnull().sum())
    print("\nStatistical Summary:\n", df.describe().T)

    sns.set_theme(style="whitegrid")

    # 1. Feature Distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Key Clinical Feature Distributions", fontsize=16, fontweight="bold")
    features = [("age", "Age (years)"), ("trestbps", "Resting Blood Pressure (mm Hg)"),
                ("chol", "Serum Cholesterol (mg/dl)"), ("thalach", "Max Heart Rate Achieved")]

    for ax, (col, label) in zip(axes.flatten(), features):
        sns.histplot(data=df, x=col, hue="target", kde=True, ax=ax, palette="Set1", bins=20)
        ax.set_title(f"Distribution of {label}", fontsize=12)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")

    plt.tight_layout()
    dist_path = os.path.join(output_dir, "feature_distributions.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"Saved: {dist_path}")

    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 9))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Heart Disease Feature Correlation Matrix", fontsize=15, fontweight="bold")
    plt.tight_layout()
    corr_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"Saved: {corr_path}")

    # 3. Class Balance
    plt.figure(figsize=(6, 5))
    target_counts = df["target"].value_counts()
    ax = sns.barplot(x=target_counts.index, y=target_counts.values, palette="Blues_d")
    plt.title("Target Class Distribution (0=Absence, 1=Presence)", fontsize=13, fontweight="bold")
    plt.xlabel("Heart Disease Status")
    plt.ylabel("Patient Count")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    class_path = os.path.join(output_dir, "class_balance.png")
    plt.savefig(class_path, dpi=300)
    plt.close()
    print(f"Saved: {class_path}")

if __name__ == "__main__":
    run_eda()
