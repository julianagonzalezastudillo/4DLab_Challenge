import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import shapiro
from scipy.stats import f_oneway, kruskal
from scipy.stats import ttest_rel
from itertools import combinations


# Load data
A = pd.read_csv('data/A.csv', header=None)

#%% Create heatmap
plt.figure(figsize=(12, 6), dpi=300)
max_val = np.max(np.abs(A))  # Get the maximum absolute value in A
ax = sns.heatmap(A, cmap="RdYlBu_r", center=0, vmin=-max_val, vmax=max_val, xticklabels=False, yticklabels=False)

plt.title("Heatmap of Dataset A (Subjects x Brain Regions)")
plt.xlabel("Brain Regions", fontsize=14)
plt.ylabel("Subjects", fontsize=14)
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')

# Adjust the colorbar
cbar = ax.collections[0].colorbar
cbar.set_label('Statistical Property', fontsize=14)  # Set label for the colorbar
cbar.ax.tick_params(labelsize=12)  # Set the tick labels' font size

plt.savefig("figures/A_heatmap.png", transparent=True, bbox_inches='tight')
plt.show()

#%% Flatten the data and plot the distribution
plt.figure(figsize=(12, 6), dpi=300)
flat_A = A.values.flatten()
sns.histplot(flat_A, kde=True)

plt.title("Distribution of Values in Dataset A")
plt.xlabel("Statistical Property", fontsize=14)
plt.savefig("figures/A_distribution.png", transparent=True, bbox_inches='tight')
plt.show()

#%%Set up the plot for all 148 histograms
n_regions = A.shape[1]
n_cols = 14  # Number of columns for subplots
n_rows = int(np.ceil(n_regions / n_cols))  # Calculate number of rows needed
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15), dpi=300, sharex=True, sharey=True)
axes = axes.flatten()

xlim = (np.min(flat_A), np.max(flat_A))
ylim = (0, 15)

# Plot each brain region in a separate subplot
for i in range(n_regions):
    axes[i].hist(A.values[:, i], bins=15, color='skyblue', edgecolor='black')
    axes[i].set_xlim(xlim)
    axes[i].set_ylim(ylim)
    axes[i].text(xlim[1]-5, ylim[1]-2, f'Region {i+1}', fontsize=8,
                 horizontalalignment='right',
                 verticalalignment='center',
                color='black')
    axes[i].tick_params(axis='both', which='major', labelsize=6)

# Hide any empty subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Add an overarching title
fig.suptitle("Histograms of Each Brain Region", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/A_distribution_per_region.png", transparent=True, bbox_inches='tight')
plt.show()

#%% Test for normality
stat, p_value = shapiro(flat_A)
print("Shapiro-Wilk Test: Statistic =", stat, ", p-value =", p_value)

#%% Testing if differences across brain regions are significant
f_values, p_values = f_oneway(*[A.iloc[:, i] for i in range(A.shape[1])])
print("ANOVA p-values for each region:", p_values)
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Reshape data for Tukey's HSD
A_long = A.melt(var_name='Brain_Region', value_name='Value')  # Convert to long format

# Perform Tukey's HSD post hoc test
tukey_results = pairwise_tukeyhsd(endog=A_long['Value'], groups=A_long['Brain_Region'], alpha=0.05)
print(tukey_results)

#%% List to store p-values
p_values = np.zeros((A.shape[1], A.shape[1]))

# Get all combinations of brain regions (column indices)
combinations_of_regions = combinations(range(A.shape[1]), 2)

# Perform pairwise t-tests
for (i, j) in combinations_of_regions:
    region1 = A.iloc[:, i]
    region2 = A.iloc[:, j]

    t_stat, p_value = ttest_rel(region1, region2)
    p_values[i, j] = p_value
    p_values[j, i] = p_value

# Apply Bonferroni correction (multiply by number of comparisons)
num_comparisons = (A.shape[1] * (A.shape[1] - 1)) / 2
bonferroni_p_values = np.minimum(p_values * num_comparisons, 1)
bonferroni_p_values[bonferroni_p_values > 0.05] = np.nan

# Set all p-values > 0.05 to NaN
bonferroni_p_values_masked = bonferroni_p_values.copy()
bonferroni_p_values_masked[bonferroni_p_values > 0.05] = np.nan

# Plot the Bonferroni corrected p-values as a matrix
plt.figure(figsize=(8, 6), dpi=300)
cmap = plt.cm.YlOrRd
cmap.set_bad(color='gray')  # Set NaN to grey
plt.imshow(bonferroni_p_values_masked, cmap=cmap, interpolation='none', vmin=0, vmax=0.05)
plt.colorbar(label="Bonferroni corrected p-values")
plt.xticks(range(A.shape[1]), A.columns, rotation=45)
plt.yticks(range(A.shape[1]), A.columns)
plt.xticks(range(0, bonferroni_p_values.shape[1], 10), [f'Region {i}' for i in range(0, bonferroni_p_values.shape[1], 10)], rotation=45)
plt.yticks(range(0, bonferroni_p_values.shape[0], 10), [f'Region {i}' for i in range(0, bonferroni_p_values.shape[0], 10)])
plt.title("Bonferroni Corrected Pairwise p-values")
plt.savefig("figures/A_p-values_corrected.png", transparent=True, bbox_inches='tight')
plt.show()

