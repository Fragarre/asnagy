# Stylometry Analysis Pipeline

A **Python-based stylometry analysis pipeline** that allows you to analyze authorship of texts using multiple classifiers, visualizations, and a Bootstrap Consensus Tree (BCT).  

This pipeline is suitable for literary analysis, forensic linguistics, and research in text classification.

---

## Features

- **Character n-gram TF-IDF extraction** with customizable n-gram range.
- **Dimensionality reduction** via Truncated SVD (LSA).
- **Multiple classifier comparison**:
  - Passive Aggressive
  - Multinomial Naive Bayes
  - Extra Trees
  - XGBoost (Linear & Tree)
  - SVM (RBF & Linear)
  - Nearest Centroid
- **Cross-validation** with confusion matrix output.
- **Bootstrap Consensus Tree (BCT)** for visualizing nearest-neighbor relationships.
- **Dimensionality visualization** with interactive t-SNE and UMAP plots.
- **Distance tables** showing each text’s distance to class centroids.

---

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/stylometry-pipeline.git
cd stylometry-pipeline
