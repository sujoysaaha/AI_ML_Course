#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Streamlit App Code
# Import dependent libs

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")

st.title("Breast Cancer Wisconsin Classification App")


# In[ ]:


# Load models
with open("model/saved_models.pkl", "rb") as f:
    models = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
model = models[model_name]


# In[ ]:


# Upload Test Data Set
uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])


# In[ ]:


# ----------------------------
# Inference / Usecase
# ----------------------------
if uploaded_file:

    data = pd.read_csv(uploaded_file)

    # Remove unnecessary columns
    data = data.drop(columns=["id"], errors="ignore")
    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

    # Check target column
    if "diagnosis" not in data.columns:
        st.error("CSV must contain 'diagnosis' column.")
        st.stop()

    # Extract target BEFORE dropping
    y = data["diagnosis"].str.upper().map({"M": 0, "B": 1})

    # Extract features
    X = data.drop("diagnosis", axis=1)

    # Scale features (important since you trained manually)
    X_scaled = scaler.transform(X)

    # Predict
    predictions = model.predict(X_scaled)

    # Display Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y, predictions))

    # Display Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    fig, ax = plt.subplots()
    ax.imshow(cm)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    # Show numbers inside matrix
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)


# In[ ]:




