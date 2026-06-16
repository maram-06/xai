import shap
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any, Union
from .utils import generate_explanation_traffic, generate_explanation_aqi, save_explanations_to_db

class XAIExplainer:
    def __init__(self, model: Any, X_train: pd.DataFrame, features: List[str]) -> None:
        '''
        Initializes the XAIExplainer with the given model, training data, and feature names. Sets up the SHAP explainer using a sample of the training data as background.
        Args:
            model: The trained machine learning model for which explanations will be generated.
            X_train: The training data used to create the background dataset for SHAP. Should be a DataFrame.
            features: A list of feature names corresponding to the columns in X_train and the model's expected input.
        
        Returns:
            None
        '''
        self.model = model
        self.features = features

        self.background = X_train.sample(n=min(200, len(X_train)), random_state=0)

        def predict_fn(X_input):
            if isinstance(X_input, np.ndarray):
                X_input = pd.DataFrame(X_input, columns=self.features)
            return self.model.predict(X_input)

        self.explainer = shap.Explainer(predict_fn, self.background, feature_names=self.features)

        self.aqi_targets = ['pm10', 'no2', 'so2', 'o3', 'co']

    def explain(self, df_input: pd.DataFrame, prediction_values: Union[List[float], np.ndarray], prediction_type: str, district_names_list: List[str], record_ids_list: List[int]) -> List[Dict[str, Any]]:
        '''
        Generates SHAP explanations and narrative text for each row in the input DataFrame. Handles both traffic and AQI prediction types.
        Args:
            df_input: DataFrame containing the input features for which explanations are to be generated.
            prediction_values: The predicted values from the model for each row in df_input. Can be a list or numpy array.
            prediction_type: A string indicating the type of prediction ('traffic' or 'aqi').
            district_names_list: A list of district names corresponding to each row in df_input.
            record_ids_list: A list of record IDs corresponding to each row in df_input.

        Returns:
            A list of dictionaries, each containing 'record_id', 'target_feature', 'shap_values', and 'explanation_text' for the corresponding row and prediction type.
        '''
        
        shap_expl_batch = self.explainer(df_input)
        all_results = []

        for row_index in range(len(df_input)):
            row_data = df_input.iloc[row_index]
            district_name = district_names_list[row_index]
            record_id = record_ids_list[row_index]

            if prediction_type == 'traffic':
                base_value = float(shap_expl_batch.base_values[row_index])
                single_pred = float(prediction_values[row_index]) if isinstance(prediction_values, (list, np.ndarray)) else float(prediction_values[row_index])
                shap_dict = dict(zip(self.features, shap_expl_batch.values[row_index]))
                shap_dict = {k: round(float(v), 4) for k, v in shap_dict.items()}
                explanation = generate_explanation_traffic(shap_dict, single_pred, base_value, district_name)

                all_results.append({
                    'record_id': record_id,
                    'target_feature': 'traffic_index',
                    'shap_values': json.dumps(shap_dict),
                    'explanation_text': explanation})
            
            elif prediction_type == 'aqi':
                single_row_pred = prediction_values[row_index] if isinstance(prediction_values, (list, np.ndarray)) else prediction_values[row_index]

                for i, target in enumerate(self.aqi_targets):
                    base_value = shap_expl_batch.base_values[row_index][i] if isinstance(shap_expl_batch.base_values[row_index], (list, np.ndarray)) else shap_expl_batch.base_values[row_index]
                    shap_dict = dict(zip(self.features, shap_expl_batch.values[row_index,:,i]))
                    shap_dict = {k: round(float(v), 4) for k, v in shap_dict.items()}
                    current_pred = float(single_row_pred[i]) if isinstance(single_row_pred, (list, np.ndarray)) else float(single_row_pred)
                    explanation = generate_explanation_aqi(row_data, shap_dict, current_pred, base_value, target)

                    all_results.append({
                        'record_id': record_id,
                        'target_feature': target,
                        'shap_values': json.dumps(shap_dict),
                        'explanation_text': explanation})
        
        return all_results

    def save_to_db(self, db_connection: Any, explanations: List[Dict[str, Any]]) -> None:
            """
            Library entry point wrapper to expose the utility database tool.
            Args:
                db_connection: An active connection object to a PostgreSQL database.
                explanations: A list of dictionaries containing the explanations to be saved. Each dictionary should have the keys 'record_id', 'target_feature', 'shap_values', and 'explanation_text'.
            
            Returns:
                None
            """
            return save_explanations_to_db(db_connection, explanations)