# Problem Statement : Classification Using Multiple ML Models
Implement multiple classification models - Build an interactive Streamlit web application to demonstrate your models - Deploy the app on Streamlit Community Cloud 
This project aims to implement multiple machine learning classification models to predict whether a tumour is malignant or benign using the Breast Cancer dataset.

# Dataset Description
•	Source: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

•	Breast Cancer Wisconsin (Diagnostic) dataset

•	Number of Instances: 569

•	Number of Features: 30 numeric features (diagnostic measurements)

•	Target Variable: diagnosis (M = Malignant, B = Benign)

•	Dataset Characteristics:

o	Features include radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension (mean, standard error, worst).

# Models Used & Evaluation Metrics
| ML Model Name        | Accuracy  | AUC       | Precision | Recall    | F1        | MCC       |
|----------------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Logistic Regression  | 0.973684  | 0.997380  | 0.972222  | 0.985915  | 0.979021  | 0.943898  |
| Decision Tree        | 0.947368  | 0.943990  | 0.957746  | 0.957746  | 0.957746  | 0.887979  |
| KNN                  | 0.947368  | 0.981985  | 0.957746  | 0.957746  | 0.957746  | 0.887979  |
| Naive Bayes          | 0.964912  | 0.997380  | 0.958904  | 0.985915  | 0.972222  | 0.925285  |
| Random Forest        | 0.964912  | 0.995087  | 0.958904  | 0.985915  | 0.972222  | 0.925285  |
| XGBoost              | 0.956140  | 0.990829  | 0.958333  | 0.971831  | 0.965035  | 0.906379  |

# Observations on Model Performance
| ML Model Name        | Observation about model performance |
|----------------------|-------------------------------------|
| Logistic Regression  | Highest accuracy and AUC; performs well on linearly separable features. |
| Decision Tree        | Slightly lower accuracy; may overfit on small datasets.                 |
| KNN                  | Good performance; sensitive to feature scaling.                         |
| Naive Bayes          | High AUC; independence assumption works well here.                      |
| Random Forest        | Stable ensemble performance; reduces overfitting.                       |
| XGBoost              | High accuracy; robust gradient boosting method.                         |

# Streamlit App Features
•	Upload dataset (CSV) for testing : Please use this data [ else format mismatch might happen ]
o	https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
•	Select ML model from dropdown
•	Display metrics: Accuracy, AUC, Precision, Recall, F1 Score, MCC
•	Visualize Confusion Matrix
