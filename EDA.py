import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(page_title="EDA App", layout="wide")
st.title("Exploratory Data Analysis App")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Missing Value Summary")
    st.write(df.isnull().sum())

    st.subheader("🧾 Data Types")
    st.write(df.dtypes)

    st.subheader("Correlation Heatmap (Numeric Columns Only)")
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for correlation heatmap.")

    st.subheader("Distribution Plot")
    numeric_cols = numeric_df.columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox("Select column for distribution", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("Outlier Detection (IQR Method)")
    outlier_col = st.selectbox("Select column to detect outliers", numeric_cols, key="outlier")
    if outlier_col:
        Q1 = df[outlier_col].quantile(0.25)
        Q3 = df[outlier_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[outlier_col] < Q1 - 1.5 * IQR) | (df[outlier_col] > Q3 + 1.5 * IQR)]
        st.write(f"Outliers detected in `{outlier_col}`: {outliers.shape[0]}")
        st.dataframe(outliers)

