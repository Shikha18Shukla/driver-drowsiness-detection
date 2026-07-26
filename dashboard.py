import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torchvision.transforms as transforms
from PIL import Image
import plotly.express as px
import numpy as np

# === Title ===
st.title("🧠 Driver Drowsiness Detection Dashboard")

st.markdown("""
Welcome to the Driver Drowsiness Detection Dashboard.  
This dashboard provides insights into our image classification model trained to detect whether a driver is **drowsy** or **alert**.  
Using deep learning with EfficientNet-B0, this project aims to improve road safety by enabling real-time monitoring systems.
""")

# === Section 1: Dataset Distribution ===
st.header("📊 Dataset Distribution")

st.markdown("""
Understanding the balance of the dataset is crucial for evaluating model performance.  
Below is a bar chart representing the number of images in each category — **Drowsy** and **Non Drowsy** — used during training and evaluation.
""")

drowsy_path = "dataset/Drowsy"
non_drowsy_path = "dataset/Non Drowsy"

drowsy_count = len(os.listdir(drowsy_path))
non_drowsy_count = len(os.listdir(non_drowsy_path))

df_dist = pd.DataFrame({
    "Class": ["Drowsy", "Non Drowsy"],
    "Count": [drowsy_count, non_drowsy_count]
})

fig = px.bar(df_dist, x="Class", y="Count", color="Class", title="Image Count per Class", text="Count")
fig.update_layout(xaxis_title="Driver State", yaxis_title="Number of Images")
st.plotly_chart(fig)

# === Section 2: Confusion Matrix ===
st.header("🧪 Confusion Matrix")

st.markdown("""
The confusion matrix below shows the number of correct and incorrect predictions made by the model.  
This visualization helps identify how well the model distinguishes between **drowsy** and **alert** drivers.
""")

if os.path.exists("outputs/y_true.npy") and os.path.exists("outputs/y_pred.npy"):
    y_true = np.load("outputs/y_true.npy")
    y_pred = np.load("outputs/y_pred.npy")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Drowsy", "Non Drowsy"])

    fig3, ax = plt.subplots()
    disp.plot(ax=ax, cmap="Blues", values_format='d')
    ax.set_title("Confusion Matrix")
    st.pyplot(fig3)
else:
    st.warning("outputs/y_true.npy and outputs/y_pred.npy not found. Please save true/pred labels during evaluation.")

# === Section 3: Summary ===
st.header("📝 Summary")

st.markdown(f"""
- The dataset contains a total of **{drowsy_count + non_drowsy_count} images**, with:
  - **{drowsy_count} Drowsy** images
  - **{non_drowsy_count} Non Drowsy** images
- The model architecture used is **EfficientNet-B0**, fine-tuned for binary classification.
- Drowsiness detection was implemented using **PyTorch** for modeling and **MediaPipe** for real-time webcam input.
- This dashboard helps stakeholders understand dataset balance, model behavior, and how well it performs on real-world data.
""")

st.markdown("---")
st.caption("Developed for Data Visualization & Interpretation Final Submission.")
