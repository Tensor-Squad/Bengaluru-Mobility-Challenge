import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

def predict_counts(tracker_history):
    # Convert tracker_history to a DataFrame
    data = []
    for tracker_id, info in tracker_history.items():
        data.append([info['timestamp'], info['class_id'], info['pattern']])

    df = pd.DataFrame(data, columns=['timestamp', 'class_id', 'pattern'])

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Aggregate data by 3 minutes
    df.set_index('timestamp', inplace=True)
    df = df.groupby([pd.Grouper(freq='3T'), 'class_id', 'pattern']).size().reset_index(name='count')

    # Label encode categorical columns
    label_encoder_class_id = LabelEncoder()
    label_encoder_pattern = LabelEncoder()
    df['class_id'] = label_encoder_class_id.fit_transform(df['class_id'])
    df['pattern'] = label_encoder_pattern.fit_transform(df['pattern'])

    # Function to create lag features
    def create_lag_features(df, lags):
        for lag in lags:
            df[f'lag_{lag}'] = df['count'].shift(lag)
        df.dropna(inplace=True)
        return df

    # Prepare data for XGBoost model
    predictions = {}
    skipped_combinations = {}
    for (class_id, pattern), group in df.groupby(['class_id', 'pattern']):
        # Set timestamp as index
        group.set_index('timestamp', inplace=True)
        
        # Ensure the time series is complete by filling missing timestamps with 0 count
        group = group.resample('3T').sum().fillna(0)
        
        # Check if the time series has enough data points
        if len(group) < 10:  # Arbitrary threshold, can be adjusted
            print(f"Skipping class_id: {class_id}, pattern: {pattern} due to insufficient data points.")
            pattern_str = label_encoder_pattern.inverse_transform([pattern])[0]
            class_id_str = label_encoder_class_id.inverse_transform([class_id])[0]
            skipped_combinations.setdefault(pattern_str, {})[class_id_str] = group['count'].sum()
            continue
        
        # Create lag features
        group = create_lag_features(group, lags=[1, 2, 3, 4, 5])
        
        # Use the entire dataset for training
        X = group.drop(columns=['count'])
        y = group['count']
        
        # Train XGBoost model
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X, y)
        
        # Make predictions for the next 30 minutes
        last_row = X.iloc[-1].values.reshape(1, -1)
        forecast = []
        for _ in range(10):  # 10 intervals of 3 minutes each to cover 30 minutes
            pred = model.predict(last_row)
            forecast.append(pred[0])
            last_row = last_row[:, 1:]  # Remove the oldest lag
            last_row = np.append(last_row, pred).reshape(1, -1)  # Add the new prediction as the newest lag
        
        # Aggregate predictions
        total_forecast = sum(forecast)
        
        # Store predictions
        pattern_str = label_encoder_pattern.inverse_transform([pattern])[0]
        class_id_str = label_encoder_class_id.inverse_transform([class_id])[0]
        if pattern_str not in predictions:
            predictions[pattern_str] = {}
        predictions[pattern_str][class_id_str] = round(total_forecast)  # Round to the nearest integer

    # Merge skipped combinations into predictions
    for pattern_str, class_ids in skipped_combinations.items():
        if pattern_str not in predictions:
            predictions[pattern_str] = {}
        for class_id_str, count in class_ids.items():
            predictions[pattern_str][class_id_str] = int(count)

    return predictions