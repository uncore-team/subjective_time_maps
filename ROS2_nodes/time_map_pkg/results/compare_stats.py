import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. Read arguments
# =========================
if len(sys.argv) != 3:
    print("Usage: python compare_stats.py <file1.csv> <file2.csv>")
    sys.exit(1)

csv_path1 = sys.argv[1]
csv_path2 = sys.argv[2]

# =========================
# 2. Load data
# =========================
# CSV columns: 0=date, 1-4=time, 5=Total Time, 6=Idle Time, 7=Num Commands,
#              8=Travelled Distance, 9=Min obs. Distance, 10=Success, 11=Idle Time %
df1 = pd.read_csv(csv_path1, usecols=[5, 6, 7, 8, 9, 10, 11], names=[
    "Total Time",
    "Idle Time",
    "Num Commands",
    "Travelled Distance",
    "Min obs. Distance",
    "Success",
    "Idle Time %"
], header=None)

df2 = pd.read_csv(csv_path2, usecols=[5, 6, 7, 8, 9, 10, 11], names=[
    "Total Time",
    "Idle Time",
    "Num Commands",
    "Travelled Distance",
    "Min obs. Distance",
    "Success",
    "Idle Time %"
], header=None)

numeric_columns = [
    "Total Time",
    "Idle Time",
    "Num Commands",
    "Travelled Distance",
    "Min obs. Distance",
    "Idle Time %"
]

# =========================
# 3. Comparative statistics
# =========================
print("\n==============================")
print("COMPARATIVE STATISTICS")
print("==============================")
print("\n--- Dataset 1 ---")
print(df1[numeric_columns + ["Success"]].describe())
print("\n--- Dataset 2 ---")
print(df2[numeric_columns + ["Success"]].describe())

# =========================
# 4. Success rate comparison
# =========================
print("\n==============================")
print("SUCCESS RATE COMPARISON")
print("==============================")

success_count1 = (df1["Success"] == 1).sum()
failure_count1 = (df1["Success"] == -1).sum()
total1 = success_count1 + failure_count1
success_pct1 = (success_count1 / total1) * 100 if total1 > 0 else 0

success_count2 = (df2["Success"] == 1).sum()
failure_count2 = (df2["Success"] == -1).sum()
total2 = success_count2 + failure_count2
success_pct2 = (success_count2 / total2) * 100 if total2 > 0 else 0

print(f"\nDataset 1: {success_count1}/{total1} successes ({success_pct1:.1f}%)")
print(f"Dataset 2: {success_count2}/{total2} successes ({success_pct2:.1f}%)")
print(f"Difference: {success_pct2 - success_pct1:.1f}%")

# =========================
# 5. Comparative boxplots
# =========================
n = len(numeric_columns)
cols = 3
rows = math.ceil((n + 1) / cols)

plt.figure(figsize=(15, 4 * rows))

for i, col in enumerate(numeric_columns, 1):
    plt.subplot(rows, cols, i)
    data1 = df1[df1["Success"] == 1][col].dropna()
    data2 = df2[df2["Success"] == 1][col].dropna()
    
    plt.boxplot([data1, data2], labels=["Dataset 1", "Dataset 2"])
    plt.title(col)
    plt.ylabel(col)
    plt.grid(axis='y', alpha=0.3)

# Success/failure rate chart
plt.subplot(rows, cols, n + 1)
x = np.arange(2)
width = 0.35

plt.bar(x - width/2, [success_count1, failure_count1], width, label='Dataset 1', color=['green', 'red'])
plt.bar(x + width/2, [success_count2, failure_count2], width, label='Dataset 2', color=['lightgreen', 'lightcoral'])

plt.ylabel('Cantidad')
plt.title('Tasa de Éxito/Fracaso')
plt.xticks(x, ['Success', 'Failure'])
plt.legend()

# Add percentages
plt.text(-width/2, success_count1 + 0.5, f"{success_pct1:.1f}%", ha='center', fontsize=9)
plt.text(width/2, success_count2 + 0.5, f"{success_pct2:.1f}%", ha='center', fontsize=9)

plt.suptitle("Dataset Comparison (only successful cases)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3, w_pad=2)
plt.show()

# =========================
# 6. Comparative boxplots by result
# =========================
plt.figure(figsize=(15, 4 * rows))

for i, col in enumerate(numeric_columns, 1):
    plt.subplot(rows, cols, i)
    
    success_data1 = df1[df1["Success"] == 1][col].dropna()
    failure_data1 = df1[df1["Success"] == -1][col].dropna()
    success_data2 = df2[df2["Success"] == 1][col].dropna()
    failure_data2 = df2[df2["Success"] == -1][col].dropna()
    
    positions = [1, 2, 4, 5]
    plt.boxplot(
        [success_data1, failure_data1, success_data2, failure_data2],
        positions=positions,
        widths=0.6
    )
    
    plt.xticks([1.5, 4.5], ['Dataset 1', 'Dataset 2'])
    plt.title(col)
    plt.ylabel(col)
    
    # Add dividing line
    plt.axvline(x=3, color='gray', linestyle='--', linewidth=0.5)
    
    # Add legend only in the first subplot
    if i == 1:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', label='Success'),
            Patch(facecolor='lightcoral', label='Failure')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

plt.suptitle("Comparison by result (Success vs Failure)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3, w_pad=2)
plt.show()

# =========================
# 7. Means comparison
# =========================
print("\n==============================")
print("MEANS COMPARISON (only successful cases)")
print("==============================")
print(f"\n{'Parameter':<25} {'Dataset 1':<15} {'Dataset 2':<15} {'Difference':<15} {'Change %'}")
print("-" * 85)

for col in numeric_columns:
    mean1 = df1[df1["Success"] == 1][col].mean()
    mean2 = df2[df2["Success"] == 1][col].mean()
    diff = mean2 - mean1
    pct_change = (diff / mean1 * 100) if mean1 != 0 else 0
    
    print(f"{col:<25} {mean1:<15.2f} {mean2:<15.2f} {diff:<15.2f} {pct_change:>6.1f}%")
