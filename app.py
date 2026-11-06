import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Classification Model Evaluation", layout="wide")
st.title("Breast Cancer Classification Dashboard")
st.markdown("Evaluate pre-trained machine learning classifiers on test data.")

# model mapping
model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl"
}

# sidebar controls
st.sidebar.header("Configuration")
selected_model_name = st.sidebar.selectbox("Select Model", list(model_files.keys()))
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

# determine dataset source
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Uploaded file loaded.")
elif os.path.exists("test_data.csv"):
    df = pd.read_csv("test_data.csv")
    st.sidebar.info("Using repository test_data.csv")
else:
    st.warning("Please upload test_data.csv to proceed.")

if df is not None:
    # check target column
    target_col = None
    for col in ['target', 'Target', 'class', 'Class']:
        if col in df.columns:
            target_col = col
            break
            
    if not target_col:
        st.error("Uploaded CSV must contain a 'target' column.")
    else:
        X_test = df.drop(columns=[target_col])
        y_test = df[target_col]

        # load scaler and model
        scaler = joblib.load('model/scaler.pkl')
        model = joblib.load(os.path.join('model', model_files[selected_model_name]))

        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

        # calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader(f"Evaluation Metrics: {selected_model_name}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("AUC", f"{auc:.4f}")
        c3.metric("Precision", f"{prec:.4f}")
        c4.metric("Recall", f"{rec:.4f}")
        c5.metric("F1 Score", f"{f1:.4f}")
        c6.metric("MCC", f"{mcc:.4f}")

        # visualizations
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)

        with col_right:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"))
