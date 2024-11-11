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
   - **Goal**: Determine if the distribution across all values fairly represents individual distributions within the dataset.

3. **Statistical Significance of Brain Region Differences**:
   - Statistical tests (e.g., ANOVA) are conducted to evaluate if the observed differences between brain regions are statistically significant.
   - **Goal**: Assess whether differences across regions are meaningful or a result of random variability.

### Task B: Analysis of Cognitive Data (Dataset B)

1. **Visualization of Dataset B**:
   - A pair plot visualizes relationships between cognitive variables, providing insights into any observable patterns.

2. **Distribution of Cognitive Variables**:
   - Histograms or box plots show the distribution for each cognitive variable, highlighting any differences in their distributions.
   - **Goal**: Identify if certain cognitive variables have unique characteristics or variability.

3. **Principal Component Analysis (PCA)**:
   - PCA is applied to reduce dimensionality and identify key components that capture the majority of the variance in the dataset.
   - **Outputs**:
     - **Explained Variance Plot**: Shows the variance captured by the first three components.
     - **Scatter Plot**: Shows the scores for the first two components, revealing potential clustering or structure.
   - **Goal**: Interpret the underlying structure of cognitive data and understand the main factors that capture data variability.

4. **Clustering Analysis**:
   - A clustering algorithm (e.g., K-means) is applied to identify groups of subjects based on cognitive variables.
   - **Output**: A scatter plot of the first two PCA components, colored by cluster assignments.
   - **Goal**: Determine if there are distinct groups in the data that could reflect different cognitive profiles.

---

## Repository Structure

- `task_A.py`: python script containing the full analysis for task A.
- `task_B.py`: python script containing the full analysis for task B.
- `README.md`: Project overview and instructions (this file).
- `data/`: Folder where Dataset A and Dataset B should be placed (ensure the datasets are in the correct format for analysis).
- `figures/`: Folder where output plots will be saved.

---

### Requirements

- Python 3.7 or higher
- Recommended libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scipy`
  - `sklearn`

### Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/brain-cognitive-analysis.git
   cd brain-cognitive-analysis
