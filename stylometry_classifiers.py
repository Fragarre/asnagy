#!/usr/bin/env python3
"""
stylometry_classifiers.py

Standalone stylometry analysis pipeline with:
- Multiple classifier comparison (cross-validation)
- Bootstrap Consensus Tree (BCT)
- t-SNE and UMAP visualizations
- Distance tables (NearestCentroid)
- Confusion matrices

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from umap import UMAP
import plotly.express as px
import pandas as pd
import shutil
import argparse
import networkx as nx
import random
import math

# --------------------- Helper functions -------------------

def clear_data_folder():
    """
    Clear the ./data/ folder completely and recreate it.
    Useful to start with a clean workspace before extracting new data.
    """
    data_path = './data/'
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)

def unzip_data(zip_file, dest='./data/'):
    """
    Extract a ZIP file containing text files into a destination folder.

    Parameters
    ----------
    zip_file : str
        Path to the ZIP file
    dest : str
        Destination folder to extract files to
    """
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(dest)
    print(f"[INFO] Data extracted to {dest}")

def load_texts_from_data_folder(corpus_path='./data/'):
    """
    Load .txt files from a folder and extract authors and filenames.

    Assumes filenames are in the format <author>_<title>.txt. 
    If no underscore is found, author is set as 'Desconocido'.

    Parameters
    ----------
    corpus_path : str
        Folder containing .txt files

    Returns
    -------
    texts : list[str]
        List of text contents
    labels : list[str]
        List of author labels
    filenames : list[str]
        List of filenames (without extension)
    """
    texts, labels, filenames = [], [], []
    for filename in sorted(os.listdir(corpus_path)):
        if filename.endswith('.txt'):
            full_path = os.path.join(corpus_path, filename)
            with open(full_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())

            # Extract author from filename
            if "_" in filename:
                author = filename.split("_")[0]
            else:
                author = "Desconocido"

            labels.append(author)
            filenames.append(filename[:-4])
    return texts, labels, filenames

def compute_tfidf_and_svd(texts, ngram_min=2, ngram_max=4, svd_max_components=150, variance_threshold=0.90, random_state=42):
    """
    Compute TF-IDF character n-grams and reduce dimensionality with Truncated SVD (LSA).

    Parameters
    ----------
    texts : list[str]
        List of text documents
    ngram_min, ngram_max : int
        Minimum and maximum character n-grams
    svd_max_components : int
        Maximum number of SVD components
    variance_threshold : float
        Minimum cumulative variance to preserve
    random_state : int
        RNG seed for reproducibility

    Returns
    -------
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer
    svd : TruncatedSVD
        Fitted SVD object
    X_reduced : np.ndarray
        Dimensionality-reduced feature matrix
    var_cum : np.ndarray
        Cumulative explained variance ratio
    optimal_n : int
        Number of SVD components chosen
    """
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(ngram_min, ngram_max))
    X = vectorizer.fit_transform(texts)
    print(f"[INFO] Total n-grams generated (vocabulary size): {len(vectorizer.get_feature_names_out())}")

    # Determine optimal number of SVD components
    max_components = min(svd_max_components, X.shape[1] - 1) if X.shape[1] > 1 else 1
    svd_temp = TruncatedSVD(n_components=max_components, random_state=random_state)
    X_reduced_temp = svd_temp.fit_transform(X)
    var_cum = np.cumsum(svd_temp.explained_variance_ratio_)
    optimal_n = int(np.argmax(var_cum >= variance_threshold) + 1) if np.any(var_cum >= variance_threshold) else max_components
    print(f"[INFO] optimal_n (components to reach {int(variance_threshold*100)}% var): {optimal_n}")

    svd = TruncatedSVD(n_components=optimal_n, random_state=random_state)
    X_reduced = svd.fit_transform(X)
    return vectorizer, svd, X_reduced, var_cum, optimal_n

def plot_and_save_confusion(labels, y_pred, classes, out_path='confusion_matrix.png'):
    """
    Generate and save a confusion matrix plot.

    Parameters
    ----------
    labels : list
        True labels
    y_pred : list
        Predicted labels
    classes : list
        List of class labels
    out_path : str
        Filepath to save the PNG
    """
    cm = confusion_matrix(labels, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    ax.set_xlabel("Etiqueta predicha", fontsize=10)
    ax.set_ylabel("Etiqueta real", fontsize=10)
    ax.tick_params(axis='both', labelsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved to {out_path}")

def compute_and_save_tsne(X_reduced, labels, filenames, out_html='tsne_plot.html', point_size=8, random_state=42):
    """
    Compute t-SNE embedding and save an interactive plot using Plotly.

    Parameters
    ----------
    X_reduced : np.ndarray
        Dimensionality-reduced feature matrix
    labels : list[str]
        Author labels
    filenames : list[str]
        Filenames for hover info
    out_html : str
        Output HTML file
    point_size : int
        Marker size
    random_state : int
        RNG seed
    """
    n_samples = len(X_reduced)
    perplexity = min(30, max(2, n_samples // 3))  # avoid too high perplexity
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X_reduced)
    tsne_df = pd.DataFrame(X_tsne, columns=['x', 'y'])
    tsne_df['author'] = labels
    tsne_df['filename'] = filenames
    fig_tsne = px.scatter(tsne_df, x='x', y='y', color='author', hover_data=['filename'],
                          title="Visualización t-SNE")
    fig_tsne.update_traces(marker=dict(size=point_size))
    fig_tsne.write_html(out_html)
    print(f"[INFO] t-SNE interactive plot saved to {out_html}")

def compute_and_save_umap(X_reduced, labels, filenames, out_html='umap_plot.html', point_size=8, random_state=42):
    """
    Compute UMAP embedding and save an interactive plot using Plotly.

    Parameters are analogous to compute_and_save_tsne.
    """
    n_samples = len(X_reduced)
    n_neighbors = min(15, max(2, n_samples // 3))
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=random_state, n_jobs=1)
    X_umap = reducer.fit_transform(X_reduced)
    umap_df = pd.DataFrame(X_umap, columns=['x', 'y'])
    umap_df['author'] = labels
    umap_df['filename'] = filenames
    fig_umap = px.scatter(umap_df, x='x', y='y', color='author', hover_data=['filename'],
                          title="Visualización UMAP")
    fig_umap.update_traces(marker=dict(size=point_size))
    fig_umap.write_html(out_html)
    print(f"[INFO] UMAP interactive plot saved to {out_html}")
    return X_umap

def compute_and_save_distance_table(X_reduced, clf, filenames, labels, out_csv='distances_table.csv'):
    """
    Compute Euclidean distance from each sample to each class centroid (NearestCentroid).

    Saves the distance table with closest class information.

    Returns
    -------
    df_distances : pd.DataFrame
        DataFrame of distances and closest class
    """
    distances = cdist(X_reduced, clf.centroids_, metric='euclidean')
    clf_classes = clf.classes_
    distance_matrix = []
    for i, fname in enumerate(filenames):
        row = {"Texto": fname, "Autor": labels[i]}
        for j, author in enumerate(clf_classes):
            row[f"Distancia_{author}"] = round(distances[i][j], 5)
        closest_author = clf_classes[np.argmin(distances[i])]
        row["Mas_cercano"] = closest_author
        distance_matrix.append(row)
    df_distances = pd.DataFrame(distance_matrix)
    cols = ["Texto", "Autor", "Mas_cercano"] + [col for col in df_distances.columns if col.startswith("Distancia")]
    df_distances = df_distances[cols]
    df_distances.to_csv(out_csv, index=False)
    print(f"[INFO] Distances table saved to {out_csv}")
    return df_distances


def normalize_features(X):
    """Normalize features to range [0,1] using MinMaxScaler"""
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

def compare_classifiers_cv(X_reduced, labels, output_prefix='clf', n_splits=5, random_state=42):
    """
    Compare multiple classifiers using stratified k-fold cross-validation.

    Saves confusion matrices and a CSV summary of mean CV accuracy.

    Parameters
    ----------
    X_reduced : np.ndarray
        Feature matrix
    labels : list[str]
        Author labels
    output_prefix : str
        Prefix for saving confusion matrices and CSV
    n_splits : int
        Number of folds in StratifiedKFold
    random_state : int
        RNG seed

    Returns
    -------
    results : dict
        Dictionary mapping classifier name to mean CV accuracy
    """
    X_norm = normalize_features(X_reduced)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = le.classes_

    CLASSIFIERS = {
        'PassiveAggressive': PassiveAggressiveClassifier(max_iter=1000, random_state=random_state),
        'MultinomialNB': MultinomialNB(),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=200, random_state=random_state),
        'XGBoost (Linear)': XGBClassifier(booster='gblinear', eval_metric='mlogloss', random_state=random_state),
        'XGBoost (Tree)': XGBClassifier(booster='gbtree', eval_metric='mlogloss', random_state=random_state),
        'SVM': SVC(kernel='rbf', probability=True, random_state=random_state),
        'SVM (Linear)': LinearSVC(max_iter=5000, random_state=random_state),
        'NearestCentroid': NearestCentroid()
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {}

    for name, clf in CLASSIFIERS.items():
        scores = cross_val_score(clf, X_norm, y, cv=skf)
        mean_acc = scores.mean()
        results[name] = mean_acc
        print(f"[INFO][Classifier CV] {name}: mean accuracy={mean_acc:.3f}")

        # Train on full data for confusion matrix
        clf.fit(X_norm, y)
        y_pred = clf.predict(X_norm)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        fig, ax = plt.subplots(figsize=(8,6))
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
        plt.title(f"Confusion Matrix: {name}")
        fig.savefig(f"{output_prefix}_{name}_confusion.png", dpi=150)
        plt.close(fig)

    # Save CSV summary
    df_clf = pd.DataFrame(list(results.items()), columns=['Classifier', 'CV_Accuracy'])
    df_clf = df_clf.sort_values(by='CV_Accuracy', ascending=False)
    df_clf.to_csv(f"{output_prefix}_classifier_comparison.csv", index=False)
    print("[INFO] Classifier comparison CSV saved.")

    return results

# ----------------- Main -------------------

def main(args):
    """
    Main pipeline execution: load data, compute features, compare classifiers,
    build BCT, produce t-SNE and UMAP visualizations, compute distance table.
    """
    if args.clear_data and os.path.exists('./data/'):
        clear_data_folder()
    if args.zip is not None:
        if not os.path.exists('./data/'):
            os.makedirs('./data/')
        unzip_data(args.zip, dest='./data/')
    else:
        if not os.path.exists('./data/') or len([f for f in os.listdir('./data/') if f.endswith('.txt')]) == 0:
            raise SystemExit("[ERROR] No data found. Provide --zip data.zip or put .txt files into ./data/")

    texts, labels, filenames = load_texts_from_data_folder('./data/')
    if len(texts) == 0:
        raise SystemExit("[ERROR] No .txt files found in ./data/")

    vectorizer, svd, X_reduced, var_cum, optimal_n = compute_tfidf_and_svd(
        texts,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        svd_max_components=args.svd_max_components,
        variance_threshold=args.variance_threshold,
        random_state=args.random_state
    )

    # ----------------- Classifier comparison -------------------
    print("[INFO] Running classifier comparison with cross-validation...")
    clf_results = compare_classifiers_cv(X_reduced, labels, output_prefix='classifier', n_splits=5, random_state=args.random_state)

    # ----------------- Visualizations -------------------
    compute_and_save_tsne(X_reduced, labels, filenames, out_html=args.tsne_out, point_size=args.point_size, random_state=args.random_state)
    compute_and_save_umap(X_reduced, labels, filenames, out_html=args.umap_out, point_size=args.point_size, random_state=args.random_state)

    # ----------------- Distance table (NearestCentroid) -------------------
    clf_centroid = NearestCentroid()
    clf_centroid.fit(X_reduced, labels)
    compute_and_save_distance_table(X_reduced, clf_centroid, filenames, labels, out_csv=args.distances_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stylometry pipeline with BCT and classifier comparison")
    parser.add_argument('--zip', type=str, default=None, help='Path to .zip file containing .txt files (optional)')
    parser.add_argument('--clear-data', dest='clear_data', action='store_true', help='Clear ./data/ folder at start')
    parser.add_argument('--ngram-min', type=int, default=2)
    parser.add_argument('--ngram-max', type=int, default=4)
    parser.add_argument('--svd-max-components', type=int, default=150)
    parser.add_argument('--variance-threshold', type=float, default=0.90)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--point-size', type=int, default=8)

    # outputs
    parser.add_argument('--tsne-out', type=str, default='tsne_plot.html')
    parser.add_argument('--umap-out', type=str, default='umap_plot.html')
    parser.add_argument('--distances-out', type=str, default='distances_table.csv')
    parser.add_argument('--consensus-out', type=str, default='consensus_tree.png')

    # BCT params
    parser.add_argument('--bct-iterations', type=int, default=500)
    parser.add_argument('--bct-subset', type=int, default=15)
    parser.add_argument('--bct-k', type=int, default=3)
    parser.add_argument('--bct-weights', nargs='+', type=float, default=[3.0, 2.0, 1.0])
    parser.add_argument('--bct-metric', type=str, default='cosine')
    parser.add_argument('--bct-trim-fraction', type=float, default=0.05)

    args = parser.parse_args()
    main(args)
