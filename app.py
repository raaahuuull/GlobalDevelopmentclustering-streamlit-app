"""
Clustering Models Evaluation App

This Streamlit application allows users to upload a dataset (CSV or Excel),
applies multiple clustering algorithms, evaluates them using Silhouette Score
and Davies-Bouldin Index, and visualizes the clusters using PCA.

No hardcoded file paths are used; data is provided via file upload.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score


st.set_page_config(page_title="Clustering Models Evaluation", layout="wide")
st.title("Clustering Models Evaluation & Deployment")


uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)


if uploaded_file is not None:

    # Load file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Original Dataset Preview")
    st.dataframe(df.head())

    # ---------------- DATA CLEANING ----------------
    df_numeric = df.apply(pd.to_numeric, errors="coerce")
    df_numeric = df_numeric.dropna(axis=1, how="all")
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(df_numeric)

    if X.shape[1] == 0:
        st.error("No valid numeric columns found.")
        st.stop()

    st.success(f"Cleaned dataset shape: {X.shape}")

   
    models = {
        "KMeans": KMeans(n_clusters=3, n_init=10, random_state=42),
        "Agglomerative": AgglomerativeClustering(n_clusters=3),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        "Gaussian Mixture": GaussianMixture(n_components=3, random_state=42),
        "MeanShift": MeanShift()
    }

   
    st.subheader("Clustering Model Evaluation")

    progress = st.progress(0)
    status = st.empty()

    results = []
    total_models = len(models)

    for i, (model_name, model) in enumerate(models.items(), start=1):
        status.text(f"Evaluating {model_name}...")

        try:
            if model_name == "MeanShift" and X.shape[0] > 1500:
                results.append({
                    "Model": model_name,
                    "Clusters": "Skipped (Slow)",
                    "Silhouette Score": "NA",
                    "Davies–Bouldin Index": "NA"
                })
            else:
                if model_name == "Gaussian Mixture":
                    labels = model.fit(X).predict(X)
                else:
                    labels = model.fit_predict(X)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                if n_clusters > 1 and -1 not in labels:
                    sil = round(silhouette_score(X, labels), 3)
                    dbi = round(davies_bouldin_score(X, labels), 3)
                else:
                    sil, dbi = "NA", "NA"

                results.append({
                    "Model": model_name,
                    "Clusters": n_clusters,
                    "Silhouette Score": sil,
                    "Davies–Bouldin Index": dbi
                })

        except Exception:
            results.append({
                "Model": model_name,
                "Clusters": "Error",
                "Silhouette Score": "Error",
                "Davies–Bouldin Index": "Error"
            })

        progress.progress(i / total_models)

    status.text("Evaluation completed")

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    st.subheader("Cluster Visualization (PCA)")

    selected_model = st.selectbox(
        "Select model for visualization",
        list(models.keys())
    )

    model = models[selected_model]

    if selected_model == "Gaussian Mixture":
        labels = model.fit(X).predict(X)
    else:
        labels = model.fit_predict(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    ax.set_title(f"{selected_model} Clustering (PCA)")
    st.pyplot(fig)


    df_output = df_numeric.copy()
    df_output["Cluster"] = labels

    st.subheader("Clustered Output Preview")
    st.dataframe(df_output.head())

    st.download_button(
        "Download Clustered Dataset",
        df_output.to_csv(index=False),
        "clustered_output.csv"
    )

