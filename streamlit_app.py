import streamlit as st
import pandas as pd
from model import train_model

# Title of the app
st.title(" Dataset Explorer ")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel) for model predictions", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the dataset
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Show basic info
        st.subheader("Preview of Dataset")
        st.write(df.head())

        # Summary statistics
        st.subheader("Summary Statistics")
        st.write(df.describe(include="all"))

        target_column = st.selectbox("Select target column", df.columns)

        if st.button("Train Model"):
            train_model = train_model(uploaded_file, target_column)
            model, metrics = train_model.feature_engineering()

            st.success("Model trained successfully!")
            st.write(metrics)

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a dataset to get started.")
