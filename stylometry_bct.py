#!/usr/bin/env python3
"""
stylometry_bct.py

Standalone version of your stylometry script with an added
Bootstrap Consensus Tree (BCT) implementation.

Usage:
    python stylometry_bct.py --zip path/to/data.zip
    or
    python stylometry_bct.py    # if ./data/ already contains .txt files

Outputs:
    - confusion_matrix.png
    - tsne_plot.html
    - umap_plot.html
    - distances_table.csv
    - consensus_tree.png
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
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist
from umap import UMAP
import plotly.express as px
import pandas as pd
import shutil
import argparse
import networkx as nx
from sklearn.metrics.pairwise import cosine_distances
import random
import math

# -------- Helper functions (close to your original ones) --------

def clear_data_folder():
    data_path = './data/'
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)

def unzip_data(zip_file, dest='./data/'):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(dest)
    print(f"[INFO] Data extracted to {dest}")

def load_texts_from_data_folder(corpus_path='./data/'):
    texts, labels, filenames = [], [], []
    for filename in sorted(os.listdir(corpus_path)):
        if filename.endswith('.txt'):
            full_path = os.path.join(corpus_path, filename)
            with open(full_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())

            if "_" in filename:
                author = filename.split("_")[0]
            else:
                author = "Desconocido"

            labels.append(author)
            filenames.append(filename[:-4])  # without extension
    return texts, labels, filenames

def compute_tfidf_and_svd(texts, ngram_min=2, ngram_max=4, svd_max_components=150, variance_threshold=0.90, random_state=42):
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(ngram_min, ngram_max))
    X = vectorizer.fit_transform(texts)
    print(f"[INFO] Total n-grams generated (vocabulary size): {len(vectorizer.get_feature_names_out())}")

    # Find number of components enough to explain threshold variance (TruncatedSVD)
    max_components = min(svd_max_components, X.shape[1] - 1) if X.shape[1] > 1 else 1
    svd_temp = TruncatedSVD(n_components=max_components, random_state=random_state)
    X_reduced_temp = svd_temp.fit_transform(X)
    var_cum = np.cumsum(svd_temp.explained_variance_ratio_)
    # choose optimal_n as the minimal components reaching threshold, but at least 1 and at most max_components
    optimal_n = int(np.argmax(var_cum >= variance_threshold) + 1) if np.any(var_cum >= variance_threshold) else max_components
    print(f"[INFO] optimal_n (components to reach {int(variance_threshold*100)}% var): {optimal_n}")

    svd = TruncatedSVD(n_components=optimal_n, random_state=random_state)
    X_reduced = svd.fit_transform(X)
    return vectorizer, svd, X_reduced, var_cum, optimal_n

def train_centroid_classifier(X_reduced, labels):
    clf = NearestCentroid()
    clf.fit(X_reduced, labels)
    return clf

def plot_and_save_confusion(labels, y_pred, classes, out_path='confusion_matrix.png'):
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
    n_samples = len(X_reduced)
    perplexity = min(30, max(2, n_samples // 3))
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
    n_samples = len(X_reduced)
    n_neighbors = min(15, max(2, n_samples // 3))
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=random_state)
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
    # reorder columns
    cols = ["Texto", "Autor", "Mas_cercano"] + [col for col in df_distances.columns if col.startswith("Distancia")]
    df_distances = df_distances[cols]
    df_distances.to_csv(out_csv, index=False)
    print(f"[INFO] Distances table saved to {out_csv}")
    return df_distances

# -------- BCT implementation --------

def bootstrap_consensus_tree(X_reduced,
                             filenames,
                             labels,
                             n_iterations=500,
                             subset_size=15,
                             k=3,
                             weighting=(3,2,1),
                             metric='cosine',
                             trim_fraction=0.05,
                             random_state=42):
    """
    Build a Bootstrap Consensus Tree (BCT) from reduced features (e.g., LSA dims).

    Parameters:
        X_reduced: array-like, shape (n_samples, n_features)
        filenames: list of labels for nodes (used for plotting)
        labels: ground truth labels (not used by BCT but kept for reference)
        n_iterations: how many random subsets to sample (500 typical)
        subset_size: how many features per subset (15 typical)
        k: number of nearest neighbours to consider per node (3 typical)
        weighting: tuple of weights for ranks (length must be k), e.g., (3,2,1)
        metric: distance metric for pairwise distances (cosine recommended)
        trim_fraction: fraction of iterations used to threshold edges (e.g., 0.05 -> weak edges removed)
        random_state: seed
    Returns:
        G: networkx.Graph with weighted undirected edges (aggregated)
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_features = X_reduced.shape

    # sanity adjustments
    subset_size = min(subset_size, n_features)
    k = min(k, n_samples - 1)  # cannot be >= n_samples

    # adjacency accumulator (directed)
    accum = np.zeros((n_samples, n_samples), dtype=float)

    weights = np.array(weighting, dtype=float)
    if weights.shape[0] != k:
        raise ValueError("Length of weighting must equal k")

    print(f"[INFO][BCT] Starting BCT: n_iter={n_iterations}, subset_size={subset_size}, k={k}, metric={metric}")
    for it in range(n_iterations):
        # random subset of feature indices
        chosen = rng.choice(n_features, size=subset_size, replace=False)
        X_sub = X_reduced[:, chosen]

        # compute pairwise distances (n_samples x n_samples)
        # using cosine distance, where smaller means more similar
        dists = cosine_distances(X_sub) if metric == 'cosine' else cdist(X_sub, X_sub, metric=metric)

        # for each sample, find k nearest neighbors (excluding self)
        for i in range(n_samples):
            row = dists[i].copy()
            row[i] = np.inf  # exclude self
            nn_idx = np.argpartition(row, k)[:k]  # indices of k smallest distances (unsorted)
            # sort the chosen neighbors by distance
            nn_idx = nn_idx[np.argsort(row[nn_idx])]
            # add weights: nearest gets weights[0], second weights[1], ...
            for rank, nbr in enumerate(nn_idx):
                accum[i, nbr] += weights[rank]

        # optional progress print
        if (it+1) % (max(1, n_iterations//10)) == 0:
            print(f"[BCT] iteration {it+1}/{n_iterations} ...")

    # make undirected weights: sum symmetric entries
    undirected_weights = accum + accum.T

    # create graph and add edges with weights
    G = nx.Graph()
    for i in range(n_samples):
        G.add_node(i, label=filenames[i], author=labels[i])

    # determine trimming threshold
    # The maximum possible per-undirected-edge weight per iteration is at most (weights.sum())
    # (in practice directional contributions vary). As a simple heuristic we compute threshold:
    # keep edges present in at least (trim_fraction * n_iterations) iterations worth of weight.
    # Convert trim_fraction to an absolute weight threshold using typical per-iteration max weight
    per_iter_max = weights.sum()  # e.g., 3+2+1 = 6
    threshold = trim_fraction * n_iterations * per_iter_max
    # Ensure threshold at least > 0
    threshold = max(threshold, 0.0)
    print(f"[BCT] trimming threshold (weight) = {threshold:.3f} (trim_fraction={trim_fraction})")

    for i in range(n_samples):
        for j in range(i+1, n_samples):
            w = undirected_weights[i, j]
            if w > 0:
                if w >= threshold:
                    G.add_edge(i, j, weight=float(w))
                else:
                    # optionally skip adding edges below threshold - here we skip
                    pass

    # If graph is too sparse (e.g., no edges), relax threshold
    if G.number_of_edges() == 0:
        print("[BCT] Warning: no edges after trimming. Relaxing threshold to include all edges > 0.")
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                w = undirected_weights[i, j]
                if w > 0:
                    G.add_edge(i, j, weight=float(w))

    print(f"[BCT] Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def draw_and_save_consensus_graph(G, filenames, labels, out_png='consensus_tree.png', figsize=(10, 8), seed=42):
    """
    Layout and draw the undirected weighted graph using Fruchterman-Reingold (spring_layout).
    Edge thickness proportional to weight.
    Node color by label.
    """
    # prepare positions (Fruchterman-Reingold)
    pos = nx.spring_layout(G, seed=seed, k=None, iterations=200)

    # node coloring: map unique labels to ints
    unique_labels = list(sorted(set(labels)))
    label_to_idx = {lab: idx for idx, lab in enumerate(unique_labels)}
    node_colors = [label_to_idx[labels[i]] if i < len(labels) else 0 for i in G.nodes()]

    # edge widths proportional to weight
    weights = np.array([edata['weight'] for _,_,edata in G.edges(data=True)])
    if len(weights) > 0:
        # normalize widths into a reasonable range
        minw, maxw = weights.min(), weights.max()
        if math.isclose(maxw, minw):
            widths = [2.0 for _ in weights]
        else:
            widths = 1.0 + 4.0 * (weights - minw) / (maxw - minw)  # between 1 and 5
    else:
        widths = []

    plt.figure(figsize=figsize)
    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                           node_size=300,
                           cmap=plt.cm.tab10,
                           node_color=node_colors)
    # Draw edges
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.8)
    # Draw labels
    labels_dict = {i: filenames[i] for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=8)

    plt.title("Bootstrap Consensus Tree (aggregated kNN)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[INFO] Consensus tree image saved to {out_png}")

# -------- Main method (CLI) --------

def main(args):
    if args.clear_data and os.path.exists('./data/'):
        clear_data_folder()
    if args.zip is not None:
        # ensure data folder exists
        if not os.path.exists('./data/'):
            os.makedirs('./data/')
        unzip_data(args.zip, dest='./data/')
    else:
        if not os.path.exists('./data/') or len([f for f in os.listdir('./data/') if f.endswith('.txt')]) == 0:
            raise SystemExit("[ERROR] No data found. Provide --zip data.zip or put .txt files into ./data/")

    texts, labels, filenames = load_texts_from_data_folder('./data/')
    if len(texts) == 0:
        raise SystemExit("[ERROR] No .txt files found in ./data/")

    # Compute TF-IDF and SVD reduction (preserve original logic)
    vectorizer, svd, X_reduced, var_cum, optimal_n = compute_tfidf_and_svd(
        texts,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        svd_max_components=args.svd_max_components,
        variance_threshold=args.variance_threshold,
        random_state=args.random_state
    )

    # Train NearestCentroid classifier and compute confusion matrix
    clf = train_centroid_classifier(X_reduced, labels)
    y_pred = clf.predict(X_reduced)
    plot_and_save_confusion(labels, y_pred, clf.classes_, out_path=args.confusion_out)

    # t-SNE and UMAP visualizations (saved as html)
    compute_and_save_tsne(X_reduced, labels, filenames, out_html=args.tsne_out, point_size=args.point_size, random_state=args.random_state)
    compute_and_save_umap(X_reduced, labels, filenames, out_html=args.umap_out, point_size=args.point_size, random_state=args.random_state)

    # distances table
    df_distances = compute_and_save_distance_table(X_reduced, clf, filenames, labels, out_csv=args.distances_out)

    # Build Bootstrap Consensus Tree
    G = bootstrap_consensus_tree(X_reduced,
                                 filenames=filenames,
                                 labels=labels,
                                 n_iterations=args.bct_iterations,
                                 subset_size=args.bct_subset,
                                 k=args.bct_k,
                                 weighting=tuple(args.bct_weights),
                                 metric=args.bct_metric,
                                 trim_fraction=args.bct_trim_fraction,
                                 random_state=args.random_state)

    draw_and_save_consensus_graph(G, filenames, labels, out_png=args.consensus_out, figsize=(12, 10), seed=args.random_state)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stylometry pipeline with Bootstrap Consensus Tree")
    parser.add_argument('--zip', type=str, default=None, help='Path to .zip file containing .txt files (optional)')
    parser.add_argument('--clear-data', dest='clear_data', action='store_true', help='Clear ./data/ folder at start')
    parser.add_argument('--ngram-min', type=int, default=2)
    parser.add_argument('--ngram-max', type=int, default=4)
    parser.add_argument('--svd-max-components', type=int, default=150)
    parser.add_argument('--variance-threshold', type=float, default=0.90)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--point-size', type=int, default=8)

    # outputs
    parser.add_argument('--confusion-out', type=str, default='confusion_matrix.png')
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
