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


# Inference/ Usecase

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    if "diagnosis" not in data.columns:
        st.error("CSV must contain 'diagnosis' column.")
        st.stop()

    # Remove unwanted columns
    data = data.drop(columns=["id", "diagnosis"], errors="ignore")
    
    X = data.drop("diagnosis", axis=1)
    y = data["diagnosis"].map({"M": 0, "B": 1})
    
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    st.subheader("Classification Report")
    st.text(classification_report(y, predictions))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, predictions)
    st.write(cm)


# In[ ]:


print("completed")


# In[ ]:






