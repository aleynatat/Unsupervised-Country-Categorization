# Unsupervised-Country-Categorization

# Global Socio-Economic Clustering Analysis 

## Overview
This project applies Unsupervised Machine Learning algorithms to categorize countries based on socio-economic and health factors. The ultimate goal is to group countries into distinct categories (e.g., Developed, Developing, Underdeveloped/Budget Needed) to help identify regions that require immediate financial or humanitarian aid.

## Approach & Methodology
To ensure robust and reliable results, this project features an **A/B Testing approach** comparing models trained on raw high-dimensional data vs. dimensionality-reduced data:
* **Raw Data (9D):** Models trained directly on scaled features.
* **PCA Data (3D):** Models trained on the first 3 Principal Components to handle multicollinearity and visualize the data properly.

### Algorithms Implemented
1. **K-Means Clustering:** Optimized using the Elbow Method (WCSS).
2. **DBSCAN:** Optimized using the K-Distance Graph and extensive Grid Search for `eps` and `min_samples`.
3. **HDBSCAN:** Density-based clustering with dynamic Grid Search for `min_cluster_size`.
4. **Agglomerative (Hierarchical) Clustering:** Built with Ward linkage, validated through Dendrogram analysis and Linkage comparison testing.

## Key Features
* **Comprehensive EDA:** Histograms and Correlation Heatmaps to understand feature distributions.
* **Hyperparameter Tuning:** Automated evaluation of DBSCAN and HDBSCAN parameters based on **Silhouette Score**.
* **Dynamic World Maps:** Interactive Choropleth maps built with `Plotly` to visualize how algorithms geographically distribute the clusters.
* **Model Leaderboard:** A final automated leaderboard and grouped bar chart comparing the Silhouette Scores of all models (PCA vs. Without PCA).

## Dataset
The dataset (`29-Country-data.csv`) includes key indicators for various countries:
* `child_mort`: Child mortality rate
* `exports` / `imports`: Trade indicators
* `health`: Total health spending
* `income`: Net income per person
* `inflation`: Inflation rate
* `life_expec`: Life expectancy
* `total_fer`: Total fertility rate
* `gdpp`: GDP per capita

## Technologies Used
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `scipy` (Hierarchical Clustering)
* **Data Visualization:** `matplotlib`, `seaborn`, `plotly`
