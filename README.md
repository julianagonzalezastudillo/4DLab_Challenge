# Brain Regions and Cognitive Variables Analysis

This repository contains an analysis of two simulated datasets provided for a technical interview. The goal is to analyze and visualize brain region statistical properties (Dataset A) and cognitive test Z-scores (Dataset B) across multiple subjects. 

## Datasets

- **Dataset A**: 80 subjects x 148 brain regions. Each element corresponds to a statistical property of a specific brain region for each subject.
- **Dataset B**: 4000 subjects x 7 cognitive variables. Each element represents a cognitive test Z-score for each subject.

---

## Analysis Overview

The project is structured to address specific analytical tasks across both datasets, divided into two main parts: **Task A** and **Task B**.

### Task A: Analysis of Brain Region Data (Dataset A)

1. **Visualization of Dataset A**:
   - A heatmap shows the values across all subjects and brain regions, providing an overview of the distribution and variability across the matrix.

2. **Distribution Analysis of Dataset A**:
   - The distribution of values across the entire dataset is visualized and assessed for its shape (e.g., normality) using histograms and normality tests. 
   - Considering the variability across regions two visualizations are provided: one of the hole flatten samples, and a second one across each brain region.

4. **Statistical Significance of Brain Region Differences**:
   - Two possible statistical tests (ANOVA, and pairwise t-test Bonferroni corrected for multiple comparisons) are conducted to evaluate if the observed differences between brain regions are statistically significant. Resulting p-values are plot on a matrix.


### Task B: Analysis of Cognitive Data (Dataset B)

1. **Visualization of Dataset B**:
   - A pair plot visualizes relationships between cognitive variables also approximated by a regression line, providing insights into any observable patterns (lower triangular matrix).
   - A heatmap of the pearson correlation across cognitive variables (upper triangular matrix).
   - Each cognitive variable histogram distribution (matrix diagonal).
   - A spring-layout correlation network, in which edges represent the correlation weight across cognitive variables.

2. **Distribution of Cognitive Variables**:
   - KDE plots show the distribution for each cognitive variable.

3. **Principal Component Analysis (PCA)**:
   - PCA is applied to reduce dimensionality and identify key components that capture the majority of the variance in the dataset.
   - **Outputs**:
     - **Explained Variance Plot**: Shows the variance captured by the first three components.
     - **Scatter Plot**: Shows the scores for the first two components, revealing potential clustering or structure.
   - **Goal**: Interpret the underlying structure of cognitive data and understand the main factors that capture data variability.

4. **Clustering Analysis**:
   - A clustering algorithm (K-means) is applied to identify groups of subjects based on cognitive variables.
   - **Output**: 
     - A scatter plot of the first two PCA components, colored by cluster assignments.
     - An updated pair plot between cognitive variables showing the identify clusters and its distribution.
   - **Goal**: Determine if there are distinct groups in the data that could reflect different cognitive profiles.

---

## Repository Structure

- `task_A.py`: python script containing the full analysis for task A.
- `task_B.py`: python script containing the full analysis for task B.
- `README.md`: Project overview and instructions (this file).
- `data/`: Folder where Dataset A and Dataset B should be placed.
- `figures/`: Folder where output plots will be saved.

---

### Requirements

- Python 3.7 or higher
- Recommended libraries:
  - `matplotlib==3.9.2`
  - `numpy==1.26.4`
  - `networkx==3.3`
  - `pandas==2.2.2`
  - `seaborn==0.13.2`
  - `scipy==1.13.1`
  - `sklearn==1.5.1`

