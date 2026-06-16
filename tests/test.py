import pandas as pd
import numpy as np
import json
from xai import XAIExplainer

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    import warnings
    warnings.filterwarnings('ignore')

    print("--- Starting Local XAI Test ---")

    base_features = ["temp_c", "humidity_pct", "wind_speed_ms", "precipitation", "hour", "day_of_week", "is_weekend", "is_holiday", "month"]
    traffic_features = base_features + ["district_id"]
    aqi_features = base_features + ["traffic_index"]
    
    # Fake background data
    X_train_traffic = pd.DataFrame(np.random.rand(100, 10) * 10, columns=traffic_features)
    X_train_aqi = pd.DataFrame(np.random.rand(100, 10) * 10, columns=aqi_features)
    
    # Train fake models
    traffic_model = RandomForestRegressor(n_estimators=10).fit(X_train_traffic, np.random.rand(100) * 100)
    aqi_model = RandomForestRegressor(n_estimators=10).fit(X_train_aqi, np.random.rand(100, 5) * 50)

    # Initialize TWO separate Explainer classes
    xai_traffic = XAIExplainer(model=traffic_model, X_train=X_train_traffic, features=traffic_features)
    xai_aqi = XAIExplainer(model=aqi_model, X_train=X_train_aqi, features=aqi_features)

    # Live Data for Traffic
    live_df_traffic = pd.DataFrame({
        "temp_c": [15], "humidity_pct": [60], "wind_speed_ms": [5], "precipitation": [0], 
        "hour": [8], "day_of_week": [1], "is_weekend": [0], "is_holiday": [0], "month": [3], 
        "district_id": [1] # District is here!
    })
    
    # Live Data for AQI
    live_df_aqi = live_df_traffic.drop(columns=["district_id"])
    live_df_aqi["traffic_index"] = [85.5] # Traffic is here!

    record_ids = [1001]
    district_names = ["Kadikoy"]

    # --- TEST TRAFFIC ---
    print("\n[TEST 1] Running Traffic...")
    traffic_results = xai_traffic.explain(
        df_input=live_df_traffic, 
        prediction_values=traffic_model.predict(live_df_traffic), 
        prediction_type='traffic', 
        district_names_list=district_names, 
        record_ids_list=record_ids
    )
    print(json.dumps(traffic_results[0], indent=2))

    # --- TEST AQI ---
    print("\n[TEST 2] Running AQI...")
    aqi_results = xai_aqi.explain(
        df_input=live_df_aqi, 
        prediction_values=aqi_model.predict(live_df_aqi), 
        prediction_type='aqi', 
        district_names_list=district_names, 
        record_ids_list=record_ids
    )
    print(json.dumps(aqi_results[0], indent=2))