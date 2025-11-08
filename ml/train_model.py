import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_and_save_model(data_path: str):
    """Train and save model with feature names"""
    
    print("📂 Loading data...")
    df = pd.read_csv(data_path)
    
    # Feature engineering
    print("🔧 Creating features...")
    
    # Time features
    if 'Time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Time'])
        df['Hour'] = df['timestamp'].dt.hour
        df['QuarterHour'] = df['Hour'] * 4 + df['timestamp'].dt.minute // 15
    
    # Temperature features
    df['DependentTemperature'] = df['ModuleTemperature'] - df['AmbientTemperature']
    df['TemperatureIrradiation'] = df['AmbientTemperature'] * df['Irradiation']
    
    # Time interaction
    if 'Hour' in df.columns:
        df['TemperatureTimeOfDay'] = df['AmbientTemperature'] * df['Hour']
    
    # Efficiency metrics
    df['PowerPerIrradiation'] = df['DC_POWER'] / (df['Irradiation'] + 1e-6)
    df['IrradiationSquared'] = df['Irradiation'] ** 2
    
    # Lag features
    if 'DailyYield' in df.columns:
        df['DailyYieldLag1'] = df['DailyYield'].shift(1)
        df['DailyYieldSameTimeYesterdayDiff'] = df['DailyYield'] - df['DailyYieldLag1']
        df['AverageDailyYieldSummary'] = df['DailyYield'].rolling(window=7, min_periods=1).mean()
    
    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Define features in EXACT order
    feature_cols = [
        'QuarterHour',
        'DailyYieldSameTimeYesterdayDiff',
        'AverageDailyYieldSummary',
        'DependentTemperature',
        'TemperatureIrradiation',
        'TemperatureTimeOfDay',
        'PowerPerIrradiation',
        'IrradiationSquared'
    ]
    
    # Filter existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"📋 Features: {feature_cols}")
    
    X = df[feature_cols]
    y = df['DC_POWER']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("🤖 Training Polynomial Regression...")
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('linear', LinearRegression())
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n✅ Model Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAE: {mae:.2f} W")
    print(f"   RMSE: {rmse:.2f} W")
    
    # Save model and feature names
    joblib.dump(model, "solar_model.pkl")
    joblib.dump(feature_cols, "feature_names.pkl")
    
    print(f"\n💾 Saved:")
    print(f"   - solar_model.pkl")
    print(f"   - feature_names.pkl")
    
    return model, feature_cols

if __name__ == "__main__":
    # Update with your data path
    train_and_save_model("Training_Data.csv")