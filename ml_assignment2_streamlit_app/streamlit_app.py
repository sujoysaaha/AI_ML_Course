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

    X = data.drop("target", axis=1)
    y = data["target"]

    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    st.subheader("Classification Report")
    st.text(classification_report(y, predictions))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, predictions)

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


# In[ ]:


print("completed")


# In[ ]:




