#!/usr/bin/env python3
"""
stylometry_bct_pipeline.py

Full stylometry pipeline with:
- Classifier comparison
- Best classifier selection
- Distance/probability table
- t-SNE/UMAP visualizations (true and predicted labels with confidence)
- Bootstrap Consensus Tree (BCT)
"""
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from umap import UMAP
import pandas as pd
import shutil
import argparse
import networkx as nx
import random
import math

# ------------------ Data Handling ---------------------

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
            with open(os.path.join(corpus_path, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
            if "_" in filename:
                author = filename.split("_")[0]
            else:
                author = "Desconocido"
            labels.append(author)
            filenames.append(filename[:-4])
    return texts, labels, filenames

# ------------------ TF-IDF and SVD ---------------------

def compute_tfidf_and_svd(texts, ngram_min=2, ngram_max=4, svd_max_components=150,
                          variance_threshold=0.90, random_state=42):
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(ngram_min, ngram_max))
    X = vectorizer.fit_transform(texts)
    max_components = min(svd_max_components, X.shape[1] - 1) if X.shape[1] > 1 else 1
    svd_temp = TruncatedSVD(n_components=max_components, random_state=random_state)
    X_reduced_temp = svd_temp.fit_transform(X)
    var_cum = np.cumsum(svd_temp.explained_variance_ratio_)
    optimal_n = int(np.argmax(var_cum >= variance_threshold) + 1) if np.any(var_cum >= variance_threshold) else max_components
    print(f"[INFO] optimal_n (components to reach {int(variance_threshold*100)}% var): {optimal_n}")
    svd = TruncatedSVD(n_components=optimal_n, random_state=random_state)
    X_reduced = svd.fit_transform(X)
    return vectorizer, svd, X_reduced, var_cum, optimal_n

# ------------------ Classifiers ---------------------

def get_classifiers(random_state=42):
    pa = PassiveAggressiveClassifier(random_state=random_state)
    mnb = MultinomialNB()
    et = ExtraTreesClassifier(random_state=random_state)
    xgbl = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state)
    xgbt = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', booster='gbtree', random_state=random_state)
    svm = SVC(probability=True, random_state=random_state)
    svml = LinearSVC(random_state=random_state)
    nc = NearestCentroid()
    return {
        'PassiveAggressive': pa,
        'MultinomialNB': mnb,
        'ExtraTrees': et,
        'XGBoost (Linear)': xgbl,
        'XGBoost (Tree)': xgbt,
        'SVM': svm,
        'SVM (Linear)': svml,
        'NearestCentroid': nc
    }

def compare_classifiers_cv(X, y, cv_folds=5, random_state=42):
    CLASSIFIERS = get_classifiers(random_state=random_state)
    results = {}
    for name, clf in CLASSIFIERS.items():
        try:
            scores = cross_val_score(clf, X, y, cv=StratifiedKFold(cv_folds, shuffle=True, random_state=random_state))
            results[name] = scores.mean()
            print(f"[CV] {name}: mean accuracy = {scores.mean():.3f}")
        except Exception as e:
            print(f"[CV] {name} failed: {e}")
            results[name] = 0.0
    return results

# ------------------ Distance table ---------------------

def compute_distance_table(X, clf, filenames, labels, out_csv='distances_table.csv'):
    if hasattr(clf, "centroids_"):
        distances = cdist(X, clf.centroids_, metric='euclidean')
        clf_classes = clf.classes_
        rows = []
        for i, fname in enumerate(filenames):
            row = {"Texto": fname, "Autor": labels[i]}
            for j, author in enumerate(clf_classes):
                row[f"Distancia_{author}"] = round(distances[i][j], 5)
            row["Mas_cercano"] = clf_classes[np.argmin(distances[i])]
            rows.append(row)
        df = pd.DataFrame(rows)
        cols = ["Texto", "Autor", "Mas_cercano"] + [col for col in df.columns if col.startswith("Distancia")]
        df = df[cols]
    elif hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        classes = clf.classes_
        df = pd.DataFrame(proba, columns=[f"P_{c}" for c in classes])
        df["Texto"] = filenames
        df["Autor"] = labels
        df["Mas_cercano"] = classes[np.argmax(proba, axis=1)]
    else:
        y_pred = clf.predict(X)
        df = pd.DataFrame({"Texto": filenames, "Autor": labels, "Mas_cercano": y_pred})
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Distance/probability table saved to {out_csv}")
    return df

# ------------------ Confusion Matrix ---------------------

def plot_confusion(labels, y_pred, classes, out_path='confusion_matrix.png'):
    cm = confusion_matrix(labels, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.tick_params(axis='both', labelsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {out_path}")

# ------------------ t-SNE / UMAP embeddings ---------------------

def plot_embedding(X, true_labels, predicted_labels=None, confidences=None,
                   filenames=None, out_html='embedding.html', method='tsne',
                   point_size=8, random_state=42, title_extra=''):
    import plotly.express as px

    # --- Compute embedding ---
    if method.lower() == 'tsne':
        X_emb = TSNE(n_components=2, random_state=random_state).fit_transform(X)
        title = "t-SNE"
    elif method.lower() == 'umap':
        X_emb = UMAP(n_components=2, random_state=random_state).fit_transform(X)
        title = "UMAP"
    else:
        raise ValueError("method must be 'tsne' or 'umap'")

    # --- Prepare dataframe ---
    df_emb = pd.DataFrame(X_emb, columns=['x','y'])
    df_emb['TrueAuthor'] = true_labels

    if predicted_labels is not None:
        df_emb['PredictedAuthor'] = predicted_labels
    if confidences is not None:
        df_emb['Confidence'] = confidences
    if filenames is not None:
        df_emb['Filename'] = filenames

    # --- Determine color column ---
    color_col = 'PredictedAuthor' if predicted_labels is not None else 'TrueAuthor'

    # --- Determine marker size ---
    # Case 1: no confidence → fixed size via size_max
    if confidences is None:
        fig = px.scatter(
            df_emb, x='x', y='y', color=color_col,
            hover_data=['Filename'] if filenames is not None else None,
            title=f"{title} {title_extra}",
        )
        fig.update_traces(marker=dict(size=point_size, opacity=0.8))

    # Case 2: confidence exists → pass vector to 'size'
    else:
        # Normalize confidence to marker sizes
        sizes = 4 + 12 * (confidences - np.min(confidences)) / (np.ptp(confidences) + 1e-8)

        fig = px.scatter(
            df_emb, x='x', y='y', color=color_col,
            hover_data=['Filename'] if filenames is not None else None,
            title=f"{title} {title_extra}",
            size=sizes,
        )
        fig.update_traces(marker=dict(opacity=0.8))

    # --- Save HTML ---
    fig.write_html(out_html)
    print(f"[INFO] {title} interactive plot saved to {out_html}")

    return X_emb

# ------------------ BCT functions (builder + drawer) ---------------------

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
    Build a Bootstrap Consensus Tree (BCT) from reduced features (X_reduced).

    Parameters
    ----------
    X_reduced : array-like, shape (n_samples, n_features)
        Feature matrix (e.g. LSA/SVD reduced).
    filenames : list[str]
        Names for each sample (used as node labels).
    labels : list[str]
        Ground-truth author labels.
    n_iterations : int
        Number of bootstrap iterations (typical 500).
    subset_size : int
        Number of feature dims to sample per iteration.
    k : int
        Number of nearest neighbours to consider per sample.
    weighting : tuple
        Weights for ranks (length must be k), e.g. (3,2,1).
    metric : str
        Distance metric (default 'cosine').
    trim_fraction : float
        Fraction used to convert to a threshold for trimming weak edges.
    random_state : int
        RNG seed.

    Returns
    -------
    G : networkx.Graph
        Undirected weighted graph aggregated over iterations.
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_features = X_reduced.shape

    # safety adjustments
    subset_size = min(subset_size, n_features)
    k = min(k, n_samples - 1)
    weights = np.array(weighting, dtype=float)
    if weights.shape[0] != k:
        raise ValueError("Length of weighting must equal k")

    # accumulator for directed vote weights
    accum = np.zeros((n_samples, n_samples), dtype=float)
    print(f"[BCT] Starting: n_iter={n_iterations}, subset_size={subset_size}, k={k}, metric={metric}")
    for it in range(n_iterations):
        chosen = rng.choice(n_features, size=subset_size, replace=False)
        X_sub = X_reduced[:, chosen]

        # compute pairwise distances (cosine returns 0..2 where smaller = more similar)
        if metric == 'cosine':
            from sklearn.metrics.pairwise import cosine_distances
            dists = cosine_distances(X_sub)
        else:
            from scipy.spatial.distance import cdist as scipy_cdist
            dists = scipy_cdist(X_sub, X_sub, metric=metric)
        for i in range(n_samples):
            row = dists[i].copy()
            row[i] = np.inf
            # get k nearest (unsorted), then sort them
            nn_idx = np.argpartition(row, k)[:k]
            nn_idx = nn_idx[np.argsort(row[nn_idx])]
            for rank, nbr in enumerate(nn_idx):
                accum[i, nbr] += weights[rank]
        if (it + 1) % max(1, (n_iterations // 10)) == 0:
            print(f"[BCT] iteration {it+1}/{n_iterations}")

    # make undirected edge weights by summing symmetric entries
    undirected_weights = accum + accum.T

    # prepare graph
    G = nx.Graph()
    for i in range(n_samples):
        G.add_node(i, label=filenames[i], author=labels[i])

    # compute threshold: keep edges present at least trim_fraction * n_iterations (scaled by per-iteration max)
    per_iter_max = weights.sum()
    threshold = trim_fraction * n_iterations * per_iter_max
    threshold = max(threshold, 0.0)
    print(f"[BCT] trimming threshold (weight) = {threshold:.3f} (trim_fraction={trim_fraction})")
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            w = undirected_weights[i, j]
            if w > 0 and w >= threshold:
                G.add_edge(i, j, weight=float(w))

    # if edges empty, relax and include any positive edge
    if G.number_of_edges() == 0:
        print("[BCT] Warning: no edges after trimming. Including all positive-weight edges.")
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                w = undirected_weights[i, j]
                if w > 0:
                    G.add_edge(i, j, weight=float(w))

    print(f"[BCT] Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def draw_and_save_consensus_graph(G, filenames, labels, out_png='consensus_tree.png', figsize=(10, 8), seed=42):
    """
    Draw an undirected weighted graph G and save as PNG.

    Node color by true label; edge width proportional to aggregated weight.
    """
    # compute layout
    pos = nx.spring_layout(G, seed=seed, k=None, iterations=200)

    # color mapping
    unique_labels = list(sorted(set(labels)))
    label_to_idx = {lab: idx for idx, lab in enumerate(unique_labels)}
    node_colors = [label_to_idx[labels[i]] if i < len(labels) else 0 for i in G.nodes()]

    # compute edge widths
    weights = np.array([edata.get('weight', 1.0) for _, _, edata in G.edges(data=True)])
    if weights.size > 0:
        minw, maxw = weights.min(), weights.max()
        if math.isclose(maxw, minw):
            widths = [2.0 for _ in weights]
        else:
            widths = 1.0 + 4.0 * (weights - minw) / (maxw - minw)
    else:
        widths = []
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos,
                           node_size=300,
                           cmap=plt.cm.tab10,
                           node_color=node_colors)
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.8)
    labels_dict = {i: filenames[i] for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=8)

    plt.title("Bootstrap Consensus Tree (aggregated kNN)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[BCT] Consensus tree image saved to {out_png}")

# ------------------ Main ---------------------

def main(args):
    if args.clear_data:
        clear_data_folder()
    if args.zip:
        if not os.path.exists('./data/'):
            os.makedirs('./data/')
        unzip_data(args.zip)
    texts, labels, filenames = load_texts_from_data_folder()
    if len(texts) == 0:
        raise SystemExit("[ERROR] No .txt files found.")

    vectorizer, svd, X_reduced, var_cum, optimal_n = compute_tfidf_and_svd(
        texts, ngram_min=args.ngram_min, ngram_max=args.ngram_max,
        svd_max_components=args.svd_max_components,
        variance_threshold=args.variance_threshold,
        random_state=args.random_state
    )

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    X_norm = normalize(X_reduced)

    # Compare classifiers
    clf_results = compare_classifiers_cv(X_norm, y_encoded, cv_folds=5, random_state=args.random_state)
    best_clf_name = max(clf_results, key=clf_results.get)
    print(f"[INFO] Best classifier: {best_clf_name} (CV accuracy = {clf_results[best_clf_name]:.3f})")

    # Fit best classifier
    best_clf = get_classifiers()[best_clf_name]
    best_clf.fit(X_norm, y_encoded)

    # Predictions & confidence
    y_pred = best_clf.predict(X_norm)
    pred_labels = le.inverse_transform(y_pred)
    if hasattr(best_clf, "predict_proba"):
        conf = best_clf.predict_proba(X_norm).max(axis=1)
    elif hasattr(best_clf, "decision_function"):
        df = best_clf.decision_function(X_norm)
        df_min, df_max = df.min(), df.max()
        conf = (df.max(axis=1) - df_min) / (df_max - df_min + 1e-8)
    else:
        conf = np.ones(len(y_encoded))

    # Confusion matrix
    plot_confusion(labels, pred_labels, le.classes_, out_path=args.confusion_out)

    # Distance/probability table
    compute_distance_table(X_norm, best_clf, filenames, labels, out_csv=args.distances_out)

    # t-SNE & UMAP embeddings
    plot_embedding(X_reduced, labels, filenames=filenames, out_html=args.tsne_out, method='tsne', point_size=args.point_size)
    plot_embedding(X_reduced, labels, filenames=filenames, out_html=args.umap_out, method='umap', point_size=args.point_size)

    # Embeddings with predicted labels + confidence
    plot_embedding(X_reduced, labels, predicted_labels=pred_labels, confidences=conf,
                   filenames=filenames, out_html='tsne_predicted.html', method='tsne',
                   point_size=args.point_size, title_extra=f"(Best: {best_clf_name})")
    plot_embedding(X_reduced, labels, predicted_labels=pred_labels, confidences=conf,
                   filenames=filenames, out_html='umap_predicted.html', method='umap',
                   point_size=args.point_size, title_extra=f"(Best: {best_clf_name})")

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
# ------------------ CLI ---------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stylometry pipeline with classifier comparison + BCT + embeddings")
    parser.add_argument('--zip', type=str, default=None)
    parser.add_argument('--clear-data', default=True, action='store_true')
    parser.add_argument('--ngram-min', type=int, default=2)
    parser.add_argument('--ngram-max', type=int, default=4)
    parser.add_argument('--svd-max-components', type=int, default=150)
    parser.add_argument('--variance-threshold', type=float, default=0.90)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--point-size', type=int, default=8)
    parser.add_argument('--confusion-out', type=str, default='confusion_matrix.png')
    parser.add_argument('--tsne-out', type=str, default='tsne_plot.html')
    parser.add_argument('--umap-out', type=str, default='umap_plot.html')
    parser.add_argument('--distances-out', type=str, default='distances_table.csv')
    parser.add_argument('--consensus-out', type=str, default='consensus_tree.png')
    parser.add_argument('--bct-iterations', type=int, default=500)
    parser.add_argument('--bct-subset', type=int, default=15)
    parser.add_argument('--bct-k', type=int, default=3)
    parser.add_argument('--bct-weights', nargs='+', type=float, default=[3.0, 2.0, 1.0])
    parser.add_argument('--bct-metric', type=str, default='cosine')
    parser.add_argument('--bct-trim-fraction', type=float, default=0.05)

    args = parser.parse_args()
    main(args)
