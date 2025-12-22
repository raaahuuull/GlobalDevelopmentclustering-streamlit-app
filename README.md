# Global Development Clustering and Model Evaluation

This project implements and evaluates multiple unsupervised clustering algorithms on a Global Development Measurement dataset and deploys the complete workflow using Streamlit.

---

## Project Objective

The objective of this project is to analyze global development indicators using unsupervised machine learning techniques, compare multiple clustering models using internal evaluation metrics, and deploy the solution as an interactive web application.

---

## Dataset Description

The dataset contains country-level development indicators such as:
- Birth Rate
- CO₂ Emissions
- GDP
- Energy Usage
- Health Expenditure
- Infant Mortality Rate
- Life Expectancy
- Internet Usage
- Business and Tax Indicators

Both CSV and Excel formats are supported.

---

## Data Preprocessing

The following preprocessing steps are applied automatically:
- Conversion of all columns to numeric values
- Removal of non-numeric and empty columns
- Mean imputation for missing values
- Replacement of infinite values

This ensures robust handling of real-world datasets.

---

## Clustering Models Used

The following clustering algorithms are implemented:
- KMeans
- Agglomerative Clustering
- DBSCAN
- Gaussian Mixture Model (GMM)
- MeanShift (skipped for large datasets due to performance constraints)

---

## Model Evaluation Metrics

Clustering models are evaluated using internal validation metrics:

### Silhouette Score
Measures cluster cohesion and separation. Higher values indicate better clustering.

### Davies–Bouldin Index
Measures average similarity between clusters. Lower values indicate better clustering.

Metrics are computed only when valid clusters are formed.

---

## Sample Evaluation Results

| Model | Clusters | Silhouette Score | Davies–Bouldin Index |
|-----|--------|------------------|----------------------|
| KMeans | 3 | 0.864 | 0.288 |
| Agglomerative | 3 | 0.866 | 0.282 |
| DBSCAN | 0 | NA | NA |
| Gaussian Mixture | 3 | 0.631 | 0.950 |
| MeanShift | Skipped (Slow) | NA | NA |

Best performing model in this setup: **Agglomerative Clustering**.

---

## Application Features

- Upload CSV or Excel datasets
- Automatic preprocessing and cleaning
- Model-wise clustering evaluation
- PCA-based 2D visualization
- Downloadable clustered output

---

## Deployment

The application is deployed using Streamlit Cloud and integrated with GitHub.

Live App:  
(https://clustering-app-app-b9u8nekxiyp9izbszxqs7w.streamlit.app/)

---

## Conclusion

This project demonstrates a complete clustering pipeline including preprocessing, model evaluation, visualization, and deployment. It highlights the importance of internal validation metrics and scalable deployment for real-world unsupervised learning applications.

---

## Author

Rahul Raj  
B.E – Artificial Intelligence and Machine Learning
