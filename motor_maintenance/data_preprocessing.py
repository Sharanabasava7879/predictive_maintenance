import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """Load the dataset"""
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape}")
        print(f"\nMachine Failure Distribution:\n{df['Machine failure'].value_counts()}")
        return df
    
    def create_features(self, df):
        """Feature engineering"""
        df_processed = df.copy()
        
        # Temperature difference
        df_processed['temp_diff'] = df_processed['Process temperature [K]'] - df_processed['Air temperature [K]']
        
        # Power estimate
        df_processed['power'] = df_processed['Torque [Nm]'] * df_processed['Rotational speed [rpm]']
        
        # Temperature ratio
        df_processed['temp_ratio'] = df_processed['Process temperature [K]'] / df_processed['Air temperature [K]']
        
        # Torque to speed ratio
        df_processed['torque_speed_ratio'] = df_processed['Torque [Nm]'] / (df_processed['Rotational speed [rpm]'] + 1)
        
        # Tool wear categories
        df_processed['tool_wear_category'] = pd.cut(df_processed['Tool wear [min]'], 
                                                      bins=[0, 50, 150, 300],
                                                      labels=['Low', 'Medium', 'High'])
        df_processed['tool_wear_low'] = (df_processed['tool_wear_category'] == 'Low').astype(int)
        df_processed['tool_wear_medium'] = (df_processed['tool_wear_category'] == 'Medium').astype(int)
        df_processed['tool_wear_high'] = (df_processed['tool_wear_category'] == 'High').astype(int)
        
        return df_processed
    
    def prepare_data(self, df):
        """Prepare features and target"""
        feature_columns = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]',
            'temp_diff',
            'power',
            'temp_ratio',
            'torque_speed_ratio',
            'tool_wear_low',
            'tool_wear_medium',
            'tool_wear_high'
        ]
        
        X = df[feature_columns]
        y = df['Machine failure']
        
        return X, y, feature_columns
    
    def split_and_scale(self, X, y, test_size=0.2, random_state=42):
        """Split data and scale features"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_scaler(self, filepath='models/scaler.pkl'):
        """Save the scaler"""
        joblib.dump(self.scaler, filepath)
        print(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath='models/scaler.pkl'):
        """Load the scaler"""
        self.scaler = joblib.load(filepath)
        print(f"Scaler loaded from {filepath}")
        return self.scaler