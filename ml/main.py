# ============================================
# REAL-TIME SOLAR PANEL CLEANING API
# FastAPI Backend for Live Sensor Data
# ============================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime
import uvicorn

# ============================================
# 1. DATA MODELS (Request/Response)
# ============================================

class SensorData(BaseModel):
    """Real-time sensor input"""
    timestamp: Optional[str] = Field(default=None, description="ISO format timestamp")
    ambient_temp: float = Field(..., description="Ambient temperature (°C)", ge=-50, le=60)
    module_temp: float = Field(..., description="Module temperature (°C)", ge=-50, le=100)
    irradiation: float = Field(..., description="Solar irradiation (W/m²)", ge=0, le=1500)
    dc_power: float = Field(..., description="Current DC power output (W)", ge=0)
    daily_yield: Optional[float] = Field(default=0, description="Daily yield so far (kWh)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-11-08T10:30:00",
                "ambient_temp": 28.5,
                "module_temp": 45.2,
                "irradiation": 850.0,
                "dc_power": 12500.0,
                "daily_yield": 45.3
            }
        }

class CleaningRecommendation(BaseModel):
    """API Response"""
    status: str = Field(..., description="normal, yellow, orange, red")
    needs_cleaning: bool
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    predicted_power: float = Field(..., description="Expected power output (W)")
    actual_power: float = Field(..., description="Current power output (W)")
    power_loss_percentage: float = Field(..., description="Performance loss (%)")
    message: str
    recommendation: str
    estimated_energy_loss_kwh: float
    timestamp: str

class BatchSensorData(BaseModel):
    """For batch predictions"""
    readings: List[SensorData]

class HealthCheck(BaseModel):
    """API health status"""
    status: str
    model_loaded: bool
    model_type: str
    model_accuracy: float
    timestamp: str

# ============================================
# 2. MODEL MANAGER CLASS
# ============================================

class SolarPanelModel:
    """Manages ML model and predictions"""
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.model_accuracy = 0.0
        self.feature_names = []
        self.scaler = None
        self.history = []  # Store last 7 readings for rolling average
        
    def load_model(self, model_path: str, feature_names_path: str = None):
        """Load trained model and feature names"""
        try:
            self.model = joblib.load(model_path)
            
            # Load feature names if provided
            if feature_names_path:
                self.feature_names = joblib.load(feature_names_path)
            
            print(f"✅ Model loaded successfully")
            print(f"📋 Expected features: {self.feature_names}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def create_features(self, data: SensorData) -> pd.DataFrame:
        """Engineer features from raw sensor data - FIXED VERSION"""
        
        # Parse timestamp
        if data.timestamp:
            ts = pd.to_datetime(data.timestamp)
        else:
            ts = pd.Timestamp.now()
        
        hour = ts.hour
        minute = ts.minute
        
        # Calculate derived features
        quarter_hour = hour * 4 + minute // 15
        dependent_temp = data.module_temp - data.ambient_temp
        temp_irradiation = data.ambient_temp * data.irradiation
        temp_time_of_day = data.ambient_temp * hour
        power_per_irradiation = data.dc_power / (data.irradiation + 1e-6) if data.irradiation > 0 else 0
        irradiation_squared = data.irradiation ** 2
        
        # Handle inf/nan
        if np.isinf(power_per_irradiation) or np.isnan(power_per_irradiation):
            power_per_irradiation = 0
        
        # Create features in EXACT order from training
        # These must match your training feature order!
        features = {
            'QuarterHour': quarter_hour,
            'DailyYieldSameTimeYesterdayDiff': 0,  # Default for real-time
            'AverageDailyYieldSummary': data.daily_yield,
            'DependentTemperature': dependent_temp,
            'TemperatureIrradiation': temp_irradiation,
            'TemperatureTimeOfDay': temp_time_of_day,
            'PowerPerIrradiation': power_per_irradiation,
            'IrradiationSquared': irradiation_squared
        }
        
        # If we have stored feature names, ensure exact match
        if self.feature_names:
            # Create DataFrame with exact feature order
            df = pd.DataFrame([features])[self.feature_names]
        else:
            df = pd.DataFrame([features])
        
        return df
    
    def predict(self, data: SensorData) -> dict:
        """Make prediction and return cleaning recommendation"""
        
        if self.model is None:
            raise ValueError("Model not loaded!")
        
        # Create features
        X = self.create_features(data)
        
        print(f"🔍 Input features: {X.columns.tolist()}")  # Debug
        print(f"🔍 Feature values: {X.values[0]}")  # Debug
        
        # Predict expected power
        predicted_power = self.model.predict(X)[0]
        
        # Handle array output
        if isinstance(predicted_power, np.ndarray):
            predicted_power = predicted_power[0]
        
        actual_power = data.dc_power
        
        # Calculate error
        residual = actual_power - predicted_power
        
        # Calculate percentage error (handle division by zero)
        if predicted_power > 0:
            error_percentage = (residual / predicted_power) * 100
        else:
            error_percentage = 0
        
        # Add to history for rolling average
        self.history.append(error_percentage)
        if len(self.history) > 7:
            self.history.pop(0)
        
        rolling_avg_error = np.mean(self.history) if self.history else error_percentage
        
        # Determine alert level
        alert_level = self._determine_alert_level(error_percentage, rolling_avg_error)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(alert_level, error_percentage)
        
        # Calculate confidence
        confidence = self._calculate_confidence(data.irradiation)
        
        # Estimate energy loss
        energy_loss = self._estimate_energy_loss(error_percentage, data.irradiation)
        
        return {
            'status': alert_level,
            'needs_cleaning': alert_level in ['orange', 'red'],
            'confidence': confidence,
            'predicted_power': float(predicted_power),
            'actual_power': float(actual_power),
            'power_loss_percentage': float(abs(error_percentage)),
            'message': self._get_status_message(alert_level),
            'recommendation': recommendation,
            'estimated_energy_loss_kwh': float(energy_loss),
            'timestamp': datetime.now().isoformat()
        }
    
    def _determine_alert_level(self, error_pct: float, rolling_avg: float) -> str:
        """Determine alert level based on error"""
        if rolling_avg < -15:
            return 'red'
        elif error_pct < -15:
            return 'orange'
        elif error_pct < -10:
            return 'yellow'
        else:
            return 'normal'
    
    def _get_status_message(self, alert_level: str) -> str:
        """Get human-readable status message"""
        messages = {
            'normal': '✅ Panels operating normally',
            'yellow': '⚠️ Minor performance drop detected',
            'orange': '🟠 Significant underperformance - consider cleaning',
            'red': '🔴 URGENT: Immediate cleaning required!'
        }
        return messages.get(alert_level, 'Unknown status')
    
    def _generate_recommendation(self, alert_level: str, error_pct: float) -> str:
        """Generate detailed recommendation"""
        if alert_level == 'red':
            return f"Immediate cleaning required! Panels are underperforming by {abs(error_pct):.1f}%. Schedule cleaning within 24 hours to prevent further energy loss."
        elif alert_level == 'orange':
            return f"Cleaning recommended within 2-3 days. Current performance drop: {abs(error_pct):.1f}%."
        elif alert_level == 'yellow':
            return f"Monitor closely. Performance drop: {abs(error_pct):.1f}%. Consider cleaning if condition persists."
        else:
            return "Panels are clean and operating efficiently. No action needed."
    
    def _calculate_confidence(self, irradiation: float) -> float:
        """Calculate prediction confidence based on conditions"""
        # Higher confidence during good sunlight conditions
        if irradiation > 600:
            return 0.95
        elif irradiation > 300:
            return 0.85
        elif irradiation > 100:
            return 0.70
        else:
            return 0.50  # Low confidence at night/cloudy
    
    def _estimate_energy_loss(self, error_pct: float, irradiation: float) -> float:
        """Estimate daily energy loss in kWh"""
        if error_pct >= 0:
            return 0.0
        
        # Assume 10kW system, 5 peak sun hours
        system_capacity_kw = 10
        peak_sun_hours = 5
        daily_generation = system_capacity_kw * peak_sun_hours
        
        loss_kwh = daily_generation * (abs(error_pct) / 100)
        return loss_kwh

# ============================================
# 3. FASTAPI APPLICATION
# ============================================

app = FastAPI(
    title="Solar Panel Cleaning Detection API",
    description="Real-time ML-powered cleaning recommendation system",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model_manager = SolarPanelModel()

# ============================================
# 4. API ENDPOINTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        # Load model and feature names
        success = model_manager.load_model(
            "solar_model.pkl",
            "feature_names.pkl"
        )
        
        if success:
            print("✅ API ready to accept requests")
        else:
            print("⚠️ Warning: Model not loaded. Train model first!")
            
    except Exception as e:
        print(f"❌ Startup error: {e}")

@app.get("/", response_model=HealthCheck)
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "model_loaded": model_manager.model is not None,
        "model_type": model_manager.model_type or "none",
        "model_accuracy": 0.9977,  # From your training results
        "timestamp": datetime.now().isoformat()
    }

@app.get("/debug/features")
async def debug_features():
    """Debug endpoint to check feature names"""
    return {
        "model_loaded": model_manager.model is not None,
        "expected_features": model_manager.feature_names,
        "feature_count": len(model_manager.feature_names)
    }

@app.post("/predict", response_model=CleaningRecommendation)
async def predict_cleaning(data: SensorData):
    """
    Real-time cleaning prediction from sensor data
    
    Returns cleaning recommendation with confidence score
    """
    try:
        if model_manager.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        result = model_manager.predict(data)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(data: BatchSensorData):
    """
    Batch prediction for multiple readings
    """
    try:
        if model_manager.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        results = []
        for reading in data.readings:
            result = model_manager.predict(reading)
            results.append(result)
        
        return {
            "total_readings": len(results),
            "predictions": results,
            "summary": {
                "normal": sum(1 for r in results if r['status'] == 'normal'),
                "yellow": sum(1 for r in results if r['status'] == 'yellow'),
                "orange": sum(1 for r in results if r['status'] == 'orange'),
                "red": sum(1 for r in results if r['status'] == 'red'),
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/history")
async def get_history():
    """Get recent prediction history"""
    return {
        "recent_errors": model_manager.history,
        "rolling_average": np.mean(model_manager.history) if model_manager.history else 0,
        "readings_count": len(model_manager.history)
    }

@app.post("/reset")
async def reset_history():
    """Reset prediction history"""
    model_manager.history = []
    return {"message": "History reset successfully"}

# ============================================
# 5. MODEL TRAINING & SAVING SCRIPT
# ============================================

def train_and_save_model(data_path: str, 
                         model_output: str = "solar_model.pkl",
                         features_output: str = "feature_names.pkl"):
    """
    Train model and save with feature names
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    
    print("📂 Loading training data...")
    df = pd.read_csv(data_path)
    
    # Feature engineering
    print("🔧 Engineering features...")
    
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
    
    # Lag features (if available)
    if 'DailyYield' in df.columns:
        df['DailyYieldLag1'] = df['DailyYield'].shift(1)
        df['DailyYieldSameTimeYesterdayDiff'] = df['DailyYield'] - df['DailyYieldLag1']
        df['AverageDailyYieldSummary'] = df['DailyYield'].rolling(window=7, min_periods=1).mean()
    
    # Replace inf values
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Define features - EXACT ORDER MATTERS!
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
    
    # Filter to only existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"📋 Using features: {feature_cols}")
    
    X = df[feature_cols]
    y = df['DC_POWER']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Training set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")
    
    # Train polynomial model
    print("🤖 Training Polynomial Regression...")
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('linear', LinearRegression())
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n{'='*60}")
    print(f"MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.2f} W")
    print(f"RMSE: {rmse:.2f} W")
    print(f"{'='*60}\n")
    
    # Save model
    joblib.dump(model, model_output)
    print(f"💾 Model saved to: {model_output}")
    
    # Save feature names - IMPORTANT!
    joblib.dump(feature_cols, features_output)
    print(f"💾 Feature names saved to: {features_output}")
    
    return model, feature_cols

# ============================================
# 6. RUN SERVER
# ============================================

if __name__ == "__main__":
    # Uncomment to train model first:
    # train_and_save_model("Training_Data.csv", "solar_model.pkl")
    
    # Run API server
    uvicorn.run(app, host="0.0.0.0", port=8000)