# SCM Random Forest Regression — Streamlit

A GitHub/Streamlit-deployable Supply Chain Management analytics application based on the supplied logistics dataset.

## Objective
Predict `delivery_time_deviation` using 13 operational, transportation, supplier, route-risk and driver-related features.

## Model
- Algorithm: Random Forest Regressor
- Train/test split: 80/20
- Random state: 42
- Default trees: 300
- Default maximum depth: 18
- Minimum samples per leaf: 2
- Metrics: MAE, RMSE, R²

## Features
`fuel_consumption_rate`, `traffic_congestion_level`, `weather_condition_severity`, `warehouse_inventory_level`, `loading_unloading_time`, `handling_equipment_availability`, `port_congestion_level`, `shipping_costs`, `supplier_reliability_score`, `lead_time_days`, `route_risk_level`, `driver_behavior_score`, `fatigue_monitoring_score`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and the `data/` folder.
3. Open Streamlit Community Cloud.
4. Select the GitHub repository and `app.py`.
5. Deploy.

## Project structure

```text
scm_random_forest_streamlit/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── dynamic_supply_chain_logistics_dataset.csv
```

## Note
The supplied notebook used Linear Regression. This deployable version keeps the same dataset, predictors, target and 80/20 evaluation structure, but replaces the estimator with `RandomForestRegressor` as requested.
