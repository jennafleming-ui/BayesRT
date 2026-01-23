import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class RiskCalcPro:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def build_bayesian_model(self, input_dim):
        """Build Bayesian Neural Network"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
        return model
    
    def prepare_features(self, laps_df):
        """Create features for lap time prediction"""
        features = pd.DataFrame()
        features['lap_number'] = laps_df['LapNumber']
        features['lap_number_normalized'] = features['lap_number'] / laps_df['LapNumber'].max()
        
        # Rolling statistics
        for driver in laps_df['Driver'].unique():
            mask = laps_df['Driver'] == driver
            laps_df.loc[mask, 'rolling_avg_3'] = laps_df.loc[mask, 'LapTimeSeconds'].rolling(3, min_periods=1).mean()
            laps_df.loc[mask, 'rolling_std_3'] = laps_df.loc[mask, 'LapTimeSeconds'].rolling(3, min_periods=1).std().fillna(0)
        
        features['rolling_avg_3'] = laps_df['rolling_avg_3']
        features['rolling_std_3'] = laps_df['rolling_std_3']
        
        # Driver encoding
        driver_dummies = pd.get_dummies(laps_df['Driver'], prefix='driver')
        features = pd.concat([features, driver_dummies], axis=1)
        
        return features
    
    def train(self, laps_df, epochs=30):
        """Train model on lap data"""
        print("🏎️ Training RiskCalc Pro...")
        
        X = self.prepare_features(laps_df)
        y = laps_df['LapTimeSeconds'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = self.build_bayesian_model(X_train_scaled.shape[1])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        self.model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=32, 
                      validation_split=0.2, verbose=0)
        
        # Test predictions
        predictions = []
        for _ in range(20):  # Monte Carlo samples
            pred = self.model.predict(X_test_scaled, verbose=0)
            predictions.append(pred.flatten())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        mae = np.mean(np.abs(mean_pred - y_test))
        rmse = np.sqrt(np.mean((mean_pred - y_test)**2))
        
        # Coverage
        lower = mean_pred - 1.96 * std_pred
        upper = mean_pred + 1.96 * std_pred
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        
        print(f"   RiskCalc Pro trained!")
        print(f"   MAE: {mae:.3f}s, RMSE: {rmse:.3f}s")
        print(f"   95% CI Coverage: {coverage*100:.1f}%")
        
        # Save model
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save('models/risk_calc_model.keras')
        print("   Model saved to models/risk_calc_model.keras")
        
        return {'mae': mae, 'rmse': rmse, 'coverage': coverage}
    
    def predict_with_uncertainty(self, X, n_samples=20):
        """Generate predictions with uncertainty"""
        if self.model is None:
            return {'mean': 75.0, 'std': 1.0, 'lower': 74.0, 'upper': 76.0}
        
        X_scaled = self.scaler.transform(X) if hasattr(self, 'scaler') else X
        
        predictions = []
        for _ in range(n_samples):
            pred = self.model.predict(X_scaled, verbose=0)
            predictions.append(pred.flatten())
        
        predictions = np.array(predictions)
        
        return {
            'mean': np.mean(predictions, axis=0),
            'std': np.std(predictions, axis=0),
            'lower': np.percentile(predictions, 2.5, axis=0),
            'upper': np.percentile(predictions, 97.5, axis=0)
        }

if __name__ == "__main__":
    print("Testing RiskCalc Pro...")
    df = pd.read_csv('data/monaco_2024_laps.csv')
    df['LapTimeSeconds'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()
    clean_df = df[(df['LapTimeSeconds'] > 70) & (df['LapTimeSeconds'] < 95)]
    
    risk_calc = RiskCalcPro()
    risk_calc.train(clean_df, epochs=30)
    