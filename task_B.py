import copy
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def scale_width(w, min_width=0.5, max_width=8):
    """
        Scale edge width based on weigh value (absolute correlation value).
        Map weight to a width in the range [min_width, max_width]
    :param w: weight
    :param min_width:
    :param max_width:
    :return:
    """
    #
    w_width = min_width + (max_width - min_width) * (abs(w) - abs(min_corr)) / (abs(max_corr) - abs(min_corr))
    return w_width

# Define cognitive variables colors
var_colors = ["#4d2fc1", "#8b65fb", "#09e0fc", "#57fcb2", "#ffc852", "#f69c09", "#f46666"]

# Load data
B = pd.read_csv('data/B.csv', header=None)

#%% PLOT DATA
# Pair plot
plt.figure(figsize=(12, 12), dpi=300)
pair_plot = sns.pairplot(B, plot_kws={'color': 'grey', 'alpha': 0.5, 'edgecolor': 'white', 'linewidth': 0.3},
                         diag_kws={'color': 'grey', 'alpha': 0.2, 'edgecolor': 'grey'})

# Correlation matrix
correlation_matrix = B.corr()

# Set diagonal to NaN
np.fill_diagonal(correlation_matrix.values, np.nan)

# Set fontsize for axis labels
for ax in pair_plot.axes[:, 0]:
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)
for ax in pair_plot.axes[-1, :]:
    ax.set_xlabel(ax.get_xlabel(), fontsize=16)

# Define a colormap for correlation values
cmap = plt.cm.YlOrRd
norm = mcolors.Normalize(vmin=0, vmax=1)

# Add a regression line and correlation value to each plot (excluding diagonal)
for i in range(len(B.columns)):
    for j in range(len(B.columns)):
        pair_plot.axes[j, i].set_xlim(np.min(B)-0.1, np.max(B)+0.1)
        pair_plot.axes[j, i].set_ylim(np.min(B)-0.1, np.max(B)+0.1)

        # Build heatmap in upper diagonal
        if i < j:
            pair_plot.axes[i, j].clear()

            # Get the correlation value and set color
            corr = correlation_matrix.iloc[i, j]
            color = cmap(norm(corr))
            pair_plot.axes[i, j].set_facecolor(color)

            # Add the correlation value as text
            pair_plot.axes[i, j].text(0.5, 0.5, f'{corr:.2f}',
                                      transform=pair_plot.axes[i, j].transAxes,
                                      horizontalalignment='center',
                                      verticalalignment='center',
                                      fontsize=22, color='black', weight='bold')
            pair_plot.axes[j, i].set_xticks([])
            pair_plot.axes[j, i].set_yticks([])

        elif i != j:
            # Add regression line
            sns.regplot(x=B.iloc[:, j], y=B.iloc[:, i], ax=pair_plot.axes[i, j],
                        scatter=False, color='red', line_kws={'color': 'red'})

            # Calculate the correlation coefficient
            corr = correlation_matrix.iloc[i, j]

            # Add the correlation value to the top-right corner of the plot
            pair_plot.axes[i, j].text(0.1, 0.9, f'ρ = {corr:.2f}',
                                      transform=pair_plot.axes[i, j].transAxes,
                                      horizontalalignment='left',
                                      verticalalignment='center',
                                      fontsize=12, color='black', weight='bold')

# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=pair_plot.axes, orientation='vertical', fraction=0.02, pad=0.04, label='Correlation')
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Correlation', fontsize=18)

pair_plot.savefig("figures/B_pair_plot.png", transparent=False, bbox_inches='tight')
plt.show()

#%% VIOLIN PLOT
plt.figure(figsize=(12, 6))
ax = sns.violinplot(data=B, palette=var_colors, width=0.8)
plt.xlabel('Cognitive Variables', fontsize=14)
plt.ylabel('z-score', fontsize=14)
plt.tight_layout()
plt.savefig("figures/B_violin_plot.png", transparent=True, bbox_inches='tight')
plt.show()

#%% PLOT CORRELATION NETWORK
# Find min max
min_corr = correlation_matrix.min().min()
max_corr = correlation_matrix.max().max()

# Create network
G = nx.Graph()
G.add_nodes_from(correlation_matrix.columns)

# Add correlation edges for all pairs of cognitive variables
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        weight = correlation_matrix.iloc[i, j]
        G.add_edge(correlation_matrix.columns[i], correlation_matrix.columns[j], weight=weight)

# Define colormap
cmap = LinearSegmentedColormap.from_list("no_white_greys", ["#d3d3d3", "#333333", "#000000"], N=256)
norm = BoundaryNorm(boundaries=np.linspace(min_corr, max_corr, 256), ncolors=256)

# Get nodes positions for spring layout
pos = nx.spring_layout(G, weight=None, k=0.001, seed=42)

# Plot
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
nx.draw_networkx_nodes(G, pos, node_size=700, node_color=var_colors, alpha=0.95,
                       edgecolors=var_colors, linewidths=1, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black', ax=ax)

# Draw edges with color and thickness based on correlation values
edges = nx.get_edge_attributes(G, 'weight')
edges_width = [scale_width(w) for w in edges.values()]
edges_color = [w for w in edges.values()]
edges_color = [cmap(norm(w)) for w in edges.values()]  # Use the custom greyscale for edge colors

nx.draw_networkx_edges(
    G, pos, edgelist=edges.keys(),
    width=edges_width,
    edge_color=edges_color,
    edge_cmap=cmap,
    edge_vmin=min_corr, edge_vmax=max_corr,
    alpha=1,
    ax=ax
)
ax.axis('off')

# Create a ScalarMappable for the color bar with the custom colormap
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])  # Required for ScalarMappable to work with colorbar

# Display the color bar on the same figure
# cbar = fig.colorbar(sm, ax=ax, label='Correlation', ticks=np.linspace(min_corr, max_corr, 5))
# cbar.ax.set_yticklabels([f"{v:.2f}" for v in np.linspace(min_corr, max_corr, 5)])
# cbar.ax.yaxis.set_tick_params(which='minor', length=0)  # Remove minor ticks if any

plt.savefig("figures/B_network_plot.png", transparent=True, bbox_inches='tight')
plt.show()

#%% DISTRIBUTION
# Create the plot
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
for idx, col in enumerate(B.columns):
    sns.kdeplot(B[col], fill=True, color=var_colors[idx], alpha=0.01, label=col, linewidth=2)
    #sns.histplot(B[col], color=var_colors[idx],fill=False, kde=True, bins=20)

# Add a vertical line at x=0
plt.axvline(x=0, color='grey', linestyle='--', linewidth=2)

plt.xlabel("z-score", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, linestyle=':', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig("figures/B_distribution.png", transparent=True, bbox_inches='tight')
plt.show()

#%% PRINCIPAL COMPONENT ANALYSIS (PCA)
pca = PCA(n_components=3)
pca.fit(B)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

# Plot the explained variance
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
plt.bar(range(1, 4), explained_variance, tick_label=["PC1", "PC2", "PC3"], color='grey')
plt.title("Explained Variance of the First Three Components", fontsize=18)
plt.xlabel("Principal Components", fontsize=14)
plt.ylabel("Variance Explained", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("figures/B_pca_variance.png", transparent=True, bbox_inches='tight')
plt.show()

# Scatter plot of the first two components
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
B_pca = pca.transform(B)
plt.scatter(B_pca[:, 0], B_pca[:, 1], color='grey', alpha=0.5, edgecolor='white', linewidth=0.3)
#plt.scatter(B_pca[:, 1], B_pca[:, 2], color='grey', alpha=0.5, edgecolor='white', linewidth=0.3)
#plt.scatter(B_pca[:, 0], B_pca[:, 2], color='grey', alpha=0.5, edgecolor='white', linewidth=0.3)
plt.title("PCA Scores (PC1 vs. PC2)", fontsize=18)
plt.xlabel("PC1", fontsize=14)
plt.ylabel("PC2", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("figures/B_pca_scores.png", transparent=True, bbox_inches='tight')
plt.show()

#%% CLUSTERING
# Using K means
optimal_k = 3  # or based on pca
kmeans = KMeans(n_clusters=optimal_k)
clusters = kmeans.fit_predict(B_pca)

fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

custom_cmap = ListedColormap(['#805af9', '#ffc852', '#cb4a4a'])
plt.scatter(B_pca[:, 0], B_pca[:, 1], c=clusters, cmap=custom_cmap, alpha=0.5, edgecolor='white', linewidth=0.3)
plt.title("K-means Clustering on PCA Components", fontsize=18)
plt.xlabel("PC1", fontsize=14)
plt.ylabel("PC2", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("figures/B_clustering_k_means.png", transparent=True, bbox_inches='tight')
plt.show()

# Try elbow method
distortions = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(B_pca)
    distortions.append(kmeans.inertia_)

plt.plot(range(1, 10), distortions, marker='o', color='grey')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Distortion")
plt.savefig("figures/B_elbow_method.png", transparent=True, bbox_inches='tight')
plt.show()

#%% Update plot
B_ = copy.deepcopy(B)
B_['cluster'] = clusters

custom_palette = custom_cmap.colors  # This will extract the list of colors from your ListedColormap

plt.figure(figsize=(12, 12), dpi=300)
pair_plot = sns.pairplot(B_, hue='cluster', palette=custom_palette, plot_kws={'alpha': 0.5, 'edgecolor': 'white', 'linewidth': 0.3},
                         diag_kws={'color': 'grey', 'alpha': 0.2, 'edgecolor': 'grey'})

# Set fontsize for axis labels
for ax in pair_plot.axes[:, 0]:
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)
for ax in pair_plot.axes[-1, :]:
    ax.set_xlabel(ax.get_xlabel(), fontsize=16)

# Add a regression line and correlation value to each plot (excluding diagonal)
for i in range(len(B.columns)):
    for j in range(len(B.columns)):
        pair_plot.axes[j, i].set_xlim(np.min(B)-0.1, np.max(B)+0.1)
        pair_plot.axes[j, i].set_ylim(np.min(B)-0.1, np.max(B)+0.1)
        if i < j:
            pair_plot.axes[i, j].set_visible(False)

plt.savefig("figures/B_pair_plot_cluster.png", transparent=True, bbox_inches='tight')
plt.show()

#%%
# from sklearn.cluster import AgglomerativeClustering
# from scipy.cluster.hierarchy import dendrogram, linkage
#
# # Step 1: Perform hierarchical clustering using Ward's method
# ward_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
# cluster_labels = ward_clustering.fit_predict(B)
#
# # Step 2: Add cluster labels to the DataFrame (optional, for reference)
# # B_pca['Cluster'] = cluster_labels
#
# # Step 3: Visualize the dendrogram to see the clustering hierarchy
# # Calculate the linkage matrix for the dendrogram
# Z = linkage(B, method='ward')
#
# # Plot the dendrogram
# plt.figure(figsize=(12, 6))
# dendrogram(Z, truncate_mode='level', p=5, show_leaf_counts=False)
# plt.title("Ward Hierarchical Clustering Dendrogram")
# plt.xlabel("Sample Index or (Cluster Size)")
# plt.ylabel("Distance")
# plt.show()
#
# # Step 4: Visualize clusters with a scatter plot for first two cognitive variables
# plt.figure(figsize=(8, 6))
#
# plt.scatter(B.values[:, 0], B.values[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
# plt.title("Scatter Plot of Subjects Grouped by Ward Clustering Labels")
# plt.xlabel("First Cognitive Variable (Standardized)")
# plt.ylabel("Second Cognitive Variable (Standardized)")
# plt.colorbar(label="Cluster Label")
# plt.show()

