#!/usr/bin/env python3
"""
stylometry_classifiers.py

Standalone version of your stylometry script with:
- Multiple classifier comparison (with cross-validation)
- Confusion matrices
"""

import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import shutil
import argparse
import time


# --------------------- Helper functions -------------------

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
            filenames.append(filename[:-4])
    from collections import Counter

    print("\n[DEBUG] Distribución de textos por clase:")
    for cls, count in Counter(labels).items():
        print(f"  {cls}: {count}")

    return texts, labels, filenames

def compute_tfidf_and_svd(texts, ngram_min=2, ngram_max=4, svd_max_components=150, variance_threshold=0.90, random_state=42):
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(ngram_min, ngram_max))
    X = vectorizer.fit_transform(texts)
    print(f"[INFO] Total n-grams generated (vocabulary size): {len(vectorizer.get_feature_names_out())}")

    max_components = min(svd_max_components, X.shape[1] - 1) if X.shape[1] > 1 else 1
    svd_temp = TruncatedSVD(n_components=max_components, random_state=random_state)
    X_reduced_temp = svd_temp.fit_transform(X)
    var_cum = np.cumsum(svd_temp.explained_variance_ratio_)
    optimal_n = int(np.argmax(var_cum >= variance_threshold) + 1) if np.any(var_cum >= variance_threshold) else max_components
    print(f"[INFO] optimal_n (components to reach {int(variance_threshold*100)}% var): {optimal_n}")

    svd = TruncatedSVD(n_components=optimal_n, random_state=random_state)
    X_reduced = svd.fit_transform(X)
    return vectorizer, svd, X_reduced, var_cum, optimal_n

# ----------------- New classifier comparison -------------------
random_state = 42
CLASSIFIERS = {
        "PassiveAggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=random_state),
    # Probabilísticos
        "SVM": SVC(kernel='rbf', probability=True, random_state=random_state),
    # No probabilísticos directos (pero calibrables)
        "SVM (Linear)": LinearSVC(max_iter=5000, random_state=random_state),
    # Basado en centroides: luego convertimos distancias en probabilidades normalizadas
        "NearestCentroid": NearestCentroid(),
    # --- NUEVOS MODELOS ---
    # 1. Logistic Regression multinomial
        "LogReg": LogisticRegression(
            max_iter=2000, 
            random_state=random_state
        ),
    # 2. Random Forest
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state
        ),
    # 3. ExtraTrees (más rápido y muy eficaz)
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=400,
            random_state=random_state
        ),
    # 4. LinearSVC calibrado → sí produce probabilidades
        "CalibratedLinearSVC": CalibratedClassifierCV(cv=5
        )
    }

def normalize_features(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

def compare_classifiers_cv(X_reduced, labels, output_prefix='clf', n_splits=5, random_state=42):
    """
    Entrena todos los clasificadores, calcula accuracy por CV,
    devuelve: results (accuracy), clasificadores entrenados y LabelEncoder.
    """

    X_norm = normalize_features(X_reduced)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = le.classes_

    from collections import Counter
    min_class = min(Counter(labels).values())
    if min_class < n_splits:
        n_splits = min_class
        print(f"[INFO] Adjusting n_splits to {n_splits}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = {}
    fitted_classifiers = {}

    for name, clf in CLASSIFIERS.items():
        print(f"[INFO] Training classifier: {name}")

        scores = cross_val_score(clf, X_norm, y, cv=skf)
        mean_acc = scores.mean()
        results[name] = mean_acc
        print(f"[INFO][CV] {name}: accuracy={mean_acc:.4f}")

        clf.fit(X_norm, y)
        fitted_classifiers[name] = clf

    return results, fitted_classifiers, le

def pick_best_probabilistic_clf(results, fitted_classifiers):
    """
    Elige el mejor clasificador que pueda dar probabilidades.
    """

    probabilistic_clfs = []

    for name, clf in fitted_classifiers.items():
        if hasattr(clf, "predict_proba"):   # SVM, LogReg, RF, ExtraTrees, CalibratedLinearSVC
            probabilistic_clfs.append(name)
        elif name == "NearestCentroid":      # lo tratamos aparte
            probabilistic_clfs.append(name)

    if not probabilistic_clfs:
        raise RuntimeError("No classifier with probability support found!")

    best_name = max(probabilistic_clfs, key=lambda n: results[n])
    best_clf = fitted_classifiers[best_name]

    print(f"[INFO] Best probabilistic classifier: {best_name}")

    return best_name, best_clf

def centroid_probabilities(clf, x_vec, classes):
    distances = clf.decision_function([x_vec])[0]
    inv = 1 / (distances + 1e-9)
    probs = inv / inv.sum()
    return dict(zip(classes, probs))

def classify_new_text(new_txt_path, vectorizer, svd, scaler, best_clf, le, output_csv="probabilities.csv"):
    with open(new_txt_path, "r", encoding="utf8") as f:
        new_text = f.read()

    X_new = vectorizer.transform([new_text])
    X_new = svd.transform(X_new)
    X_new = scaler.transform(X_new)

    classes = le.classes_

    if hasattr(best_clf, "predict_proba"):
        probs = best_clf.predict_proba(X_new)[0]
        result = dict(zip(classes, probs))

    else:
        result = centroid_probabilities(best_clf, X_new[0], classes)

    df = pd.DataFrame({
        "Author": list(result.keys()),
        "Probability": [round(p, 4) for p in result.values()]
    })
    df = df.sort_values(by="Probability", ascending=False)
    df.to_csv(output_csv, index=False, float_format="%.4f")


    df = df.sort_values(by="Probability", ascending=False)
    df.to_csv(output_csv, index=False)

    print(f"[INFO] Probabilities saved to {output_csv}")



# ----------------- Main -------------------

def main(args):
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

    results, fitted_clfs, le = compare_classifiers_cv(
    X_reduced, labels, output_prefix="classifier",
    n_splits=5, random_state=args.random_state
    )

    best_name, best_clf = pick_best_probabilistic_clf(results, fitted_clfs)

    scaler = MinMaxScaler()
    scaler.fit(X_reduced)

    if args.classify:
        classify_new_text(args.classify, vectorizer, svd, scaler, best_clf, le)

   # ----------------- Classifier comparison -------------------
    print("[INFO] Running classifier comparison with cross-validation...")
    clf_results = compare_classifiers_cv(
        X_reduced,
        labels,
        output_prefix='classifier',
        n_splits=5,
        random_state=args.random_state
    )

# ============================================================
# 📌 NUEVO: Tabla comparativa de clasificadores
# ============================================================

    print("[INFO] Building detailed classifier comparison table...")

    results_table = []

    # Usamos los mismos clasificadores que compare_classifiers_cv
    CLASSIFIERS_DETAILED = {
        'PassiveAggressive': PassiveAggressiveClassifier(max_iter=1000, random_state=args.random_state),
        'SVM_rbf': SVC(kernel='rbf', probability=True, random_state=args.random_state),
        'LinearSVC_Calibrated': CalibratedClassifierCV(
            LinearSVC(max_iter=5000, random_state=args.random_state),
            cv=5
        ),
        'LogisticRegression': LogisticRegression(max_iter=5000, n_jobs=-1, random_state=args.random_state),
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.random_state
        )
    }

    # Entrenar todos sobre datos reducidos normalizados
    X_norm = normalize_features(X_reduced)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    for name, clf in CLASSIFIERS_DETAILED.items():
        print(f"[INFO] Fitting {name}...")

        start_train = time.time()
        clf.fit(X_norm, y)
        train_time = time.time() - start_train

        start_pred = time.time()
        y_pred = clf.predict(X_norm)
        pred_time = time.time() - start_pred

        acc = (y_pred == y).mean()
        n_features = X_norm.shape[1]

        results_table.append({
            "classifier": name,
            "accuracy": round(acc, 4),
            "train_time": round(train_time, 4),
            "predict_time": round(pred_time, 4),
            "n_features": round(n_features, 4)  # aunque es entero, se mantiene consistente
        })

    df_results = pd.DataFrame(results_table).sort_values(by="accuracy", ascending=False)
    df_results.to_csv("classifier_comparison.csv", index=False, float_format="%.4f")

    print("[INFO] classifier_comparison.csv saved successfully.")
    # ============================================================


    # ----------------- Distance table (NearestCentroid) -------------------
    clf_centroid = NearestCentroid()
    clf_centroid.fit(X_reduced, labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stylometry pipeline Classifier comparison")
    parser.add_argument('--zip', type=str, default=None, help='Path to .zip file containing .txt files (optional)')
    parser.add_argument('--clear-data', default=True, dest='clear_data', action='store_true', help='Clear ./data/ folder at start')
    parser.add_argument('--ngram-min', type=int, default=2)
    parser.add_argument('--ngram-max', type=int, default=4)
    parser.add_argument('--svd-max-components', type=int, default=150)
    parser.add_argument('--variance-threshold', type=float, default=0.90)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--point-size', type=int, default=8)
    parser.add_argument('--classify', type=str, default=None,
                    help='Path to a .txt file to classify with best classifier')
    args = parser.parse_args()
    main(args)
