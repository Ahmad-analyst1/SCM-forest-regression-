
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="SCM Random Forest Regression", page_icon="🚚", layout="wide")

FEATURES = [
    "fuel_consumption_rate", "traffic_congestion_level",
    "weather_condition_severity", "warehouse_inventory_level",
    "loading_unloading_time", "handling_equipment_availability",
    "port_congestion_level", "shipping_costs",
    "supplier_reliability_score", "lead_time_days",
    "route_risk_level", "driver_behavior_score",
    "fatigue_monitoring_score"
]
TARGET = "delivery_time_deviation"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dynamic_supply_chain_logistics_dataset.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df[FEATURES + [TARGET]].dropna()

@st.cache_resource
def train_model(n_estimators=300, max_depth=18, min_samples_leaf=2):
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R²": r2_score(y_test, pred)
    }
    return model, X_train, X_test, y_train, y_test, pred, metrics

st.title("🚚 Supply Chain Management — Random Forest Regression")
st.caption("Predicting Delivery Time Deviation from logistics and operational risk factors")

df = load_data()

with st.sidebar:
    st.header("Model Settings")
    n_estimators = st.slider("Number of trees", 100, 500, 300, 50)
    max_depth = st.slider("Max depth", 5, 30, 18)
    min_leaf = st.slider("Minimum samples per leaf", 1, 10, 2)
    st.markdown("---")
    st.write(f"Dataset rows: **{len(df):,}**")
    st.write(f"Predictors: **{len(FEATURES)}**")

model, X_train, X_test, y_train, y_test, y_pred, metrics = train_model(
    n_estimators, max_depth, min_leaf
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "🎯 Prediction", "🔎 Feature Importance", "📈 Model Diagnostics"
])

with tab1:
    st.subheader("Dataset & Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("MAE", f"{metrics['MAE']:.3f}")
    c3.metric("RMSE", f"{metrics['RMSE']:.3f}")
    c4.metric("R²", f"{metrics['R²']:.3f}")

    st.markdown("**Target:** `delivery_time_deviation`")
    st.dataframe(df.head(20), use_container_width=True)

with tab2:
    st.subheader("Predict Delivery Time Deviation")
    st.info("Enter operational conditions below. Values are constrained to the ranges observed in the training dataset.")

    inputs = {}
    cols = st.columns(2)
    for i, feature in enumerate(FEATURES):
        lo = float(df[feature].min())
        hi = float(df[feature].max())
        med = float(df[feature].median())
        step = (hi - lo) / 100 if hi > lo else 1.0
        with cols[i % 2]:
            inputs[feature] = st.number_input(
                feature.replace("_", " ").title(),
                min_value=lo, max_value=hi, value=med, step=step,
                format="%.3f"
            )

    if st.button("Predict Delivery Time Deviation", type="primary"):
        row = pd.DataFrame([inputs])[FEATURES]
        prediction = float(model.predict(row)[0])
        st.success(f"Predicted delivery time deviation: **{prediction:.3f}**")

with tab3:
    st.subheader("Random Forest Feature Importance")
    imp = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp["Feature"], imp["Importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance — Random Forest")
    ax.grid(axis="x", alpha=0.25)
    st.pyplot(fig, use_container_width=True)

    st.dataframe(imp.sort_values("Importance", ascending=False).reset_index(drop=True),
                 use_container_width=True)

with tab4:
    st.subheader("Actual vs Predicted")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(y_test, y_pred, alpha=0.35)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax1.plot([mn, mx], [mn, mx], linestyle="--")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Actual vs Predicted Delivery Time Deviation")
    ax1.grid(alpha=0.25)
    st.pyplot(fig1, use_container_width=True)

    st.subheader("Residual Distribution")
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(residuals, bins=40)
    ax2.set_xlabel("Residual (Actual − Predicted)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Prediction Residuals")
    ax2.grid(alpha=0.25)
    st.pyplot(fig2, use_container_width=True)

st.markdown("---")
st.caption("Educational SCM analytics project • Random Forest Regressor • Train/Test split = 80/20 • random_state = 42")
