import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# PAGE SETTINGS
st.set_page_config(page_title="Smart Property Valuation Platform", layout="wide")

st.title("🏢 Smart Property Valuation Platform")
st.write("Upload datasets, explore patterns, train ML models, and predict property valuations instantly.")

# FILE UPLOAD
uploaded_file = st.file_uploader("Upload Real Estate Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # FIX: Handle missing values automatically
    df = df.fillna(df.mean(numeric_only=True))

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.write("### Dataset Shape:", df.shape)

    # EDA SECTION
    st.header("🔍 Exploratory Data Analysis")

    # Missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Numeric summary
    st.subheader("Statistical Summary")
    st.write(df.describe())

    # Numerical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Distribution plots
    st.subheader("📊 Numerical Feature Distributions")
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        plt.title(f"Distribution of {col}")
        st.pyplot(fig)

    # Correlation heatmap
    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig)

    # Scatterplots
    st.subheader("📈 Feature vs Target Scatter Plots")
    target = st.selectbox("Select Target Column (Sale Amount)", df.columns)

    for col in num_cols:
        if col != target:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[col], y=df[target])
            plt.title(f"{col} vs {target}")
            st.pyplot(fig)

    # MODEL TRAINING
    st.header("🤖 Model Training")

    selected_features = st.multiselect("Select Feature Columns", num_cols, default=list(num_cols))

    # ensure target isn't in feature list
    if target in selected_features:
        selected_features.remove(target)

    if len(selected_features) >= 1:
        X = df[selected_features]
        y = df[target]

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # ML Models
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
        }

        results = []

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            results.append([name, r2, mae, rmse])

        results_df = pd.DataFrame(results, columns=["Model", "R² Score", "MAE", "RMSE"])

        st.subheader("📌 Model Comparison")
        st.dataframe(results_df)

        # best model selection
        best_model_name = results_df.sort_values("R² Score", ascending=False).iloc[0, 0]
        best_model = models[best_model_name]

        st.success(f"🏆 Best Model Selected: {best_model_name}")

        best_model.fit(X_scaled, y)

        # PREDICTION FORM
        st.header("🧮 Predict Property Value")

        input_data = {}

        st.subheader("Enter Property Details:")

        for col in selected_features:
            default_val = float(df[col].mean()) if not df[col].isnull().all() else 0
            input_data[col] = st.number_input(f"{col}", value=default_val)

        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        if st.button("Predict Valuation"):
            prediction = best_model.predict(input_scaled)[0]
            st.success(f"💰 Estimated Property Value: {prediction:,.2f} USD")

else:
    st.info("Upload a CSV file to begin.")
