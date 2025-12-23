# 🌍 Global Development Clustering & Model Evaluation

An end-to-end **unsupervised machine learning project** that analyzes global development indicators, compares multiple clustering algorithms using internal validation metrics, and deploys the workflow as an **interactive Streamlit web application**.

---

## 🎯 Project Objective

The objective of this project is to:

- Analyze country-level global development indicators  
- Apply and compare multiple **unsupervised clustering algorithms**  
- Evaluate clustering performance using **internal validation metrics**  
- Deploy the complete pipeline as an **interactive Streamlit application**

This project focuses on **model comparison, evaluation, and interpretability** rather than prediction.

---

## 📊 Dataset Description

The dataset contains **country-level global development indicators**, including:

- Birth Rate  
- CO₂ Emissions  
- GDP  
- Energy Usage  
- Health Expenditure  
- Infant Mortality Rate  
- Life Expectancy  
- Internet Usage  
- Business and Tax Indicators  

The application supports both **CSV and Excel (.xlsx)** formats via file upload.

---

## 🧹 Data Preprocessing

The following preprocessing steps are automatically applied:

- Conversion of all columns to numeric values  
- Removal of non-numeric and empty columns  
- Replacement of infinite values  
- Mean imputation for missing values  

This ensures robustness when handling **real-world, noisy datasets**.

---

## 🤖 Clustering Models Implemented

The project evaluates the following clustering algorithms:

- **KMeans**
- **Agglomerative Clustering**
- **DBSCAN**
- **Gaussian Mixture Model (GMM)**
- **MeanShift**  
  *(Skipped automatically for large datasets due to performance constraints)*

---

## 📈 Model Evaluation Metrics

Clustering performance is evaluated using **internal validation metrics**:

### 🔹 Silhouette Score  
Measures how well data points fit within their assigned cluster.  
Higher values indicate better clustering quality.

### 🔹 Davies–Bouldin Index  
Measures average similarity between clusters.  
Lower values indicate better clustering quality.

Metrics are computed only when **valid clusters** are formed.

---

## 🧪 Sample Evaluation Results

| Model | Clusters | Silhouette Score | Davies–Bouldin Index |
|------|----------|------------------|----------------------|
| KMeans | 3 | 0.864 | 0.288 |
| Agglomerative | 3 | 0.866 | 0.282 |
| DBSCAN | 0 | NA | NA |
| Gaussian Mixture | 3 | 0.631 | 0.950 |
| MeanShift | Skipped (Slow) | NA | NA |

**Best performing model in this setup:**  
➡️ **Agglomerative Clustering**

---

## 📊 Visualization

- Dimensionality reduction using **PCA (2 components)**
- Interactive 2D cluster visualization
- Visual comparison across clustering algorithms

---

## 🚀 Application Features

- Upload CSV or Excel datasets  
- Automatic preprocessing and cleaning  
- Model-wise clustering evaluation  
- PCA-based visualization  
- Downloadable clustered dataset  

---

## 🖥️ Deployment

The application is deployed using **Streamlit Cloud** and integrated with GitHub.

- The app accepts datasets via **file upload**
- No hardcoded dataset paths are used
- Fully portable and reproducible

🔗 **Live Application:**  
https://clustering-app-app-b9u8nekxiyp9izbszxqs7w.streamlit.app/

---

## ▶️ How to Run Locally

git clone https://github.com/raaahuull/GlobalDevelopmentclustering-streamlit-app.git
cd GlobalDevelopmentclustering-streamlit-app
pip install -r requirements.txt
streamlit run app.py

---

## 📂 Project Structure

GlobalDevelopmentClustering-streamlit-app/
├── assets/        # Screenshots and visuals
├── data/          # Sample datasets (CSV, Excel)
├── models/        # Saved clustering models (.pkl)
├── notebooks/     # EDA and model-building notebooks
├── app.py         # Streamlit application
├── requirements.txt
└── README.md

---

## Conclusion

This project demonstrates a complete unsupervised learning workflow, including preprocessing, clustering, evaluation, visualization, and deployment. It emphasizes the importance of model evaluation metrics and scalable deployment for real-world data science applications.

---

## Author

Rahul Raj  
B.E – Artificial Intelligence and Machine Learning
