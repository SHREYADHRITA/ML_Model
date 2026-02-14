import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from model import LogisticRegressionModel, DecisionTreeModel, KNNModel, NaiveBayesModel, RandomForestModel, XGBoostModel
from utils import evaluate_model, load_sample_data
import joblib


st.set_page_config(page_title="Streamlit ML Demo", layout="wide")
st.title("Streamlit ML — Train and Compare Models")
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Load sample data if no file uploaded
if uploaded_file is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

# Safe CSV read
import io
try:
    content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    if not content.strip():
        st.error("The uploaded file appears to be empty. Please upload a valid CSV file.")
        st.stop()

    try:
        df = pd.read_csv(io.StringIO(content))
    except pd.errors.ParserError:
        try:
            df = pd.read_csv(io.StringIO(content), sep=";")
        except pd.errors.ParserError:
            df = pd.read_csv(io.StringIO(content), sep="\\t")

    if df.empty or len(df.columns) == 0:
        st.error("No valid columns found in the uploaded file. Ensure it’s a proper CSV with headers.")
        st.stop()

except pd.errors.EmptyDataError:
    st.error("No columns to parse from file. Please upload a valid CSV (check for headers).")
    st.stop()
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.success("File uploaded successfully ✅")
st.write("### Data Preview")
st.dataframe(df.head())

# Load uploaded data
df = pd.read_csv(uploaded_file)

# Encode categorical features
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    st.info(f"Encoding categorical columns: {', '.join(categorical_cols)}")
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Target column selection
target_column = st.sidebar.selectbox("Select target column", df.columns)
X = df.drop(columns=[target_column])
y = df[target_column]

# Encode target if categorical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
if y.dtype == 'object' or y.dtype.name == 'category':
    y = le.fit_transform(y)


# Ensure y is properly encoded for all models (especially XGBoost)
from sklearn.preprocessing import LabelEncoder
if y.dtype == "object" or y.dtype.name == "category" or y.min() != 0 or not set(y.unique()) == set(range(len(y.unique()))):
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), name=target_column)


# Split configuration
test_size = st.sidebar.slider("Test set proportion", 0.1, 0.5, 0.2)
random_state = st.sidebar.number_input("Random seed", value=42)


# Handle stratify only if each class has >=2 samples
if y.value_counts().min() < 2:
    stratify_y = None
else:
    stratify_y = y

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=test_size, random_state=int(random_state), stratify=stratify_y
)

# Model selection
model_name = st.sidebar.selectbox(
"Select Model",
["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

model_mapping = {
    "Logistic Regression": LogisticRegressionModel,
    "Decision Tree": DecisionTreeModel,
    "KNN": KNNModel,
    "Naive Bayes": NaiveBayesModel,
    "Random Forest": RandomForestModel,
    "XGBoost": XGBoostModel,
}

ModelClass = model_mapping[model_name]

# show some hyperparameters (simple)
st.sidebar.write("### Hyperparameters")
if model_name == "Logistic Regression":
    C = st.sidebar.number_input("Inverse regularization (C)", 0.0001, 10.0, value=1.0, format="%f")
    max_iter = st.sidebar.number_input("max_iter", 50, 2000, value=200)
    model = ModelClass(C=float(C), max_iter=int(max_iter))
elif model_name == "KNN":
    n_neighbors = st.sidebar.slider("n_neighbors", 1, 50, 5)
    model = ModelClass(n_neighbors=int(n_neighbors))
elif model_name == "Decision Tree":
    max_depth = st.sidebar.slider("max_depth", 1, 50, value=2)
    model = ModelClass(max_depth=int(max_depth))
elif model_name == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 10, 1000, value=100)
    model = ModelClass(n_estimators=int(n_estimators))
elif model_name == "XGBoost":
    n_estimators = st.sidebar.slider("n_estimators", 10, 1000, value=100)
    learning_rate = st.sidebar.number_input("learning_rate", 0.001, 1.0, value=0.01)
    model = ModelClass(n_estimators=int(n_estimators), learning_rate=float(learning_rate))
else:
    model = ModelClass()


if st.button("Train model"):
    with st.spinner("Training..."):
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

    st.success("Training finished")
    st.write("### Metrics")
    st.json(metrics)

    # Save model
    save_name = st.text_input("Save model filename (optional)", value=f"{model_name.replace(' ', '_').lower()}.pkl")
    if st.button("Save model"):
        joblib.dump(model, save_name)
    st.info(f"Saved to {save_name}")
