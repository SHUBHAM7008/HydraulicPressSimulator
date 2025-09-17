import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_path = "hydraulic_press_dataset_cleaned.csv" # Update path if needed
df = pd.read_csv(file_path)

# Select numeric features only
num_features = df.select_dtypes(include=["int64", "float64"]).columns

# Correlation matrix
corr_matrix = df[num_features].corr()

# --- 1. Heatmap ---
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.show()

# --- 2. Correlation with targets ---
targets = ["binary_failure_T50", "RUL_cycles"]

for t in targets:
    print(f"\nTop correlations with {t}:")
    print(corr_matrix[t].sort_values(ascending=False).head(10))
