import pandas as pd
from psycopg2.extras import execute_values
from typing import Dict, Any, List, Union

def generate_explanation_traffic(shap_values: Dict[str, float], prediction_values: float, base_value: float, district_name: str) -> str:
        '''
        Generates a narrative explanation for traffic predictions based on SHAP values, prediction values, and the district name. Identifies the main contributing features and constructs an explanation that highlights the key drivers of the predicted traffic congestion level.
        
        it follows a structured approach:
        1. Defines narrative labels for each feature.
        2. Contains predefined mechanism explanations for certain features that can be included in the narrative.
        3. Determines the direction of the prediction (increase or decrease) compared to the base value.
        4. Ranks the features based on the absolute value of their SHAP contributions to identify the main and secondary contributors.
        5. Constructs a narrative explanation that incorporates the main contributing feature, optionally includes a secondary feature, and adds a mechanism clause if applicable.

        the secondary feature takes from the mechanism explanations if available, otherwise it uses a predefined sentence structure. The mechanism clause is added if there is a third feature with a known mechanism that hasn't already been used as the main or secondary feature.
        used features are added to a list so that the same feature is not referenced multiple times in the explanation.
        
        Args:
            shap_values: A dictionary mapping feature names to their corresponding SHAP values for a single prediction instance.
            prediction_values: The predicted traffic congestion level for the instance.
            base_value: The base value for the SHAP explanation.
            district_name: The name of the district for which to generate the explanation.
        Returns:
            A string containing the narrative explanation for the traffic prediction.
        '''
        target_name = "traffic congestion level"

        feature_labels = {
            "district_id": f"traffic conditions in {district_name.capitalize()}",
            "hour": "the time of day",
            "day_of_week": "the day of the week",
            "is_weekend": "weekend travel patterns",
            "is_holiday": "holiday travel patterns",
            "month": "seasonal travel patterns",
            "temp_c": "temperature",
            "humidity_pct": "humidity",
            "wind_speed_ms": "wind speed",
            "precipitation": "the rainfall/snowfall level",
        }

        # Mechanism explanations
        feature_mechanisms = {
            "hour": {
                "positive": "the current time coincides with typical commuting hours",
                "negative": "the current time falls outside typical peak commuting periods"
            },
            "is_weekend": {
                "positive": "weekend travel activity can increase traffic in some districts",
                "negative": "weekend conditions often reduce commuter traffic"
            },
            "is_holiday": {
                "positive": "holiday travel activity can increase traffic in some districts",
                "negative": "holiday conditions often reduce commuter traffic"
            },
            "district_id": {
                "positive": f"overall traffic conditions across {district_name.capitalize()} contribute to higher congestion",
                "negative": f"overall traffic conditions in {district_name.capitalize()} are typically less congested"
            }
        }

        direction = "increase" if prediction_values > base_value else "decrease"

        if direction == "increase":
            candidates = [f for f in shap_values if shap_values[f] > 0]
        else:
            candidates = [f for f in shap_values if shap_values[f] < 0]

        candidates.sort(key=lambda f: abs(shap_values[f]), reverse=True)

        # main feature
        main_feature = candidates[0]
        main_label = feature_labels[main_feature]

        # secondary feature
        secondary_feature = candidates[1] if len(candidates) > 1 else None
        secondary_label = feature_labels[secondary_feature] if secondary_feature else None

        used_features = {main_feature}
        if secondary_feature:
            used_features.add(secondary_feature)

        # secondary clause
        secondary_clause = ""
        if secondary_feature:
            if secondary_feature in feature_mechanisms:
                mech = feature_mechanisms[secondary_feature]
                clause = mech["positive"] if direction == "increase" else mech["negative"]
                secondary_clause = f" {clause.capitalize()}."
            else:
                if direction == "increase":
                    secondary_clause = f" {secondary_label.capitalize()} also contributes to increased congestion."
                else:
                    secondary_clause = f" {secondary_label.capitalize()} also contributes to lighter traffic."

        # mechanism clause (third feature not already used)
        mechanism_clause = ""
        for f in candidates:
            if f not in used_features and f in feature_mechanisms:
                mech = feature_mechanisms[f]
                clause = mech["positive"] if direction == "increase" else mech["negative"]
                mechanism_clause = f" Moreover, {clause}."
                break

        # build narrative
        if direction == "increase":
            narrative = f"The predicted {target_name} is higher than usual, primarily due to {main_label}.{secondary_clause}{mechanism_clause}"
        else:
            narrative = f"The predicted {target_name} is lower than usual, mainly due to {main_label}.{secondary_clause}{mechanism_clause}"

        return narrative

def generate_explanation_aqi(row: Union[pd.DataFrame, pd.Series], shap_values: Dict[str, float], prediction_values: float, base_value: float, pollutant: str) -> str:
    '''
    Generates a narrative explanation for AQI predictions based on SHAP values, prediction values, the specific pollutant, and the input feature values. Identifies the main contributing features and constructs an explanation that highlights the key drivers of the predicted pollutant concentration level.
    
    it follows a structured approach:
    1. Defines narrative labels for each feature.
    2. Contains predefined mechanism explanations for certain features that can be included in the narrative.
    3. Determines the direction of the prediction (increase or decrease) compared to the base value.
    4. Ranks the features based on the absolute value of their SHAP contributions to identify the main and secondary contributors.
    5. Checks for the rush hour clause condition.
    6. Constructs a narrative explanation that incorporates the main contributing feature, optionally includes a secondary feature, adds mechanism clause and rush hour clause if applicable.
    
    the secondary feature takes from the mechanism explanations if available, otherwise it uses a predefined sentence structure. The mechanism clause is added if there is a third feature with a known mechanism that hasn't already been used as the main or secondary feature.
    the rush hour clause is added if the main feature is traffic and the time of day indicates rush hour, but only if hour isn't already used as a main or secondary feature.
    used features are added to a list so that the same feature is not referenced multiple times in the explanation.

    Args:
        row: The input feature values for the prediction instance, provided as a DataFrame or Series. Used to extract the hour of day for the rush hour clause from the actual row.
        shap_values: A dictionary mapping feature names to their corresponding SHAP values for a single prediction instance.
        prediction_values: The predicted pollutant concentration level for the instance.
        base_value: The base value for the SHAP explanation.
        pollutant: The specific pollutant for which to generate the explanation.

    Returns:
        A string containing the narrative explanation for the AQI prediction.
    '''
    
    feature_labels = {
        "traffic_index": "traffic congestion",
        "temp_c": "temperature",
        "humidity_pct": "humidity",
        "wind_speed_ms": "wind speed",
        "hour": "time of day",
        "day_of_week": "day of the week",
        "is_weekend": "weekend traffic patterns",
        "is_holiday": "holiday traffic patterns",
        "precipitation": "the rainfall/snowfall level",
        "month": "seasonal travel patterns"
    }

    feature_mechanisms = {
        "wind_speed_ms": {
            "positive": "higher wind speed helps disperse pollutants",
            "negative": "relatively low wind speed limits pollutant dispersion"
        },
        "traffic_index": {
            "positive": "heavy traffic increases emissions",
            "negative": "lighter traffic contributes to lower emissions"
        }
    }

    direction = "increase" if prediction_values > base_value else "decrease"

    if direction == "increase":
        candidates = [f for f in shap_values if shap_values[f] > 0]
    else:
        candidates = [f for f in shap_values if shap_values[f] < 0]

    candidates.sort(key=lambda f: abs(shap_values[f]), reverse=True)

    main_feature = candidates[0]
    main_label = feature_labels[main_feature]
    
    # secondary feature
    secondary_feature = candidates[1] if len(candidates) > 1 else None
    secondary_label = feature_labels[secondary_feature] if secondary_feature else None

    used_features = {main_feature}
    if secondary_feature:
        used_features.add(secondary_feature)

    rush_clause = ""
    if direction == "increase" and main_feature in ["traffic_index"] and "hour" not in used_features and "is_weekend" == 0 and "is_holiday" == 0:
        if isinstance(row, pd.DataFrame):
            hour_val = int(row.iloc[0]["hour"])
        elif isinstance(row, pd.Series):
            hour_val = int(row["hour"])
        else: 
            hour_val = int(row[shap_values.keys().index("hour")])
        if 7 <= hour_val <= 9:
            rush_clause = " Morning rush-hour traffic further increases emissions."
        elif 16 <= hour_val <= 19:
            rush_clause = " Evening rush-hour traffic further increases emissions."

    if rush_clause:
        used_features.update(["hour"])  # skip these for mechanism

    # secondary clause
    secondary_clause = ""
    if secondary_feature:
        if secondary_feature in feature_mechanisms:
            mech = feature_mechanisms[secondary_feature]
            clause = mech["positive"] if direction == "increase" else mech["negative"]
            secondary_clause = f" {clause.capitalize()}."
        else:
            if direction == "increase":
                secondary_clause = f" {secondary_label.capitalize()} also contributes positively to emissions."
            else:
                secondary_clause = f" {secondary_label.capitalize()} also helps reduce pollutant concentrations."

    # mechanism clause (third feature not already used)
    mechanism_clause = ""
    for f in candidates:
        if f not in used_features and f in feature_mechanisms:
            mech = feature_mechanisms[f]
            clause = mech["positive"] if direction == "increase" else mech["negative"]
            mechanism_clause = f" Furthermore, {clause}."
            break

    narrative = ""
    if direction == "increase":
        narrative = f"The predicted {pollutant.upper()} level is higher than usual, primarily due to {main_label}.{secondary_clause}{mechanism_clause}{rush_clause}"
    else:
        narrative = f"The predicted {pollutant.upper()} level is lower than usual, mainly due to {main_label}.{secondary_clause}{mechanism_clause}"

    return narrative

def save_explanations_to_db(db_connection: Any, explanations: List[Dict[str, Any]]) -> None:
        """
        Takes the list of dictionaries generated by explain() and generate_explanation functions, and performs a bulk insert into a PostgreSQL database.
        Args:
            db_connection: An active connection object to a PostgreSQL database.
            explanations: A list of dictionaries containing the explanations to be saved. Each dictionary should have the keys 'record_id', 'target_feature', 'shap_values', and 'explanation_text'.
        
        Returns:
            None
        """

        insert_query = """
            INSERT INTO xai_explanations (record_id, target_feature, shap_values, explanation_text)
            VALUES %s
            ON CONFLICT (record_id, target_feature) DO NOTHING;
        """
        
        data_tuples = [
            (
                item['record_id'], 
                item['target_feature'], 
                item['shap_values'], 
                item['explanation_text']
            ) 
            for item in explanations
        ]
        
        cursor = None

        try:
            cursor = db_connection.cursor()
            execute_values(cursor, insert_query, data_tuples)
            db_connection.commit()
            print(f"Successfully saved {len(data_tuples)} XAI explanations to the database.")
            
        except Exception as e:
            db_connection.rollback()
            print(f"Failed to save XAI explanations: {e}")
            
        finally:
            if cursor:
                cursor.close()