import streamlit as st
import pandas as pd

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

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a dataset to get started.")
