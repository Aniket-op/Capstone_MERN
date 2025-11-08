# ============================================
# REAL-TIME SOLAR PANEL CLEANING API
# FastAPI Backend for Live Sensor Data
# ============================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
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
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2024-11-08T10:30:00",
                "ambient_temp": 28.5,
                "module_temp": 45.2,
                "irradiation": 850.0,
                "dc_power": 12500.0,
                "daily_yield": 45.3
            }
        }
    )

class CleaningRecommendation(BaseModel):
    """API Response"""
    status: str = Field(..., description="normal, yellow, orange, red, unknown")
    needs_cleaning: bool
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    predicted_power: Optional[float] = Field(default=None, description="Expected power output (W)")
    actual_power: float = Field(..., description="Current power output (W)")
    power_loss_percentage: float = Field(..., description="Performance loss (%)")
    message: str
    recommendation: str
    estimated_energy_loss_kwh: float
    consecutive_bad_readings: Optional[int] = Field(default=0)
    timestamp: str

class BatchSensorData(BaseModel):
    """For batch predictions"""
    readings: List[SensorData]

class HealthCheck(BaseModel):
    """API health status"""
    status: str
    ml_model_loaded: bool  # Changed from model_loaded
    ml_model_type: str     # Changed from model_type
    ml_model_accuracy: float  # Changed from model_accuracy
    timestamp: str
    
    model_config = ConfigDict(protected_namespaces=())  # Allow model_ prefix

# ============================================
# 2. MODEL MANAGER CLASS
# ============================================

class SolarPanelModel:
    """Manages ML model and predictions"""
    
    def __init__(self):
        self.model = None
        self.model_type = "polynomial"
        self.model_accuracy = 0.9977
        self.feature_names = []
        self.history = []
        self.maintenance_mode = False
        
    def load_model(self, model_path: str, feature_names_path: str = None):
        """Load trained model and feature names"""
        try:
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
            
            if feature_names_path:
                self.feature_names = joblib.load(feature_names_path)
                print(f"📋 Expected features ({len(self.feature_names)}): {self.feature_names}")
            else:
                print("⚠️ No feature names file provided")
            
            return True
        except FileNotFoundError as e:
            print(f"❌ File not found: {e}")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def create_features(self, data: SensorData) -> pd.DataFrame:
        """Engineer features from raw sensor data"""
        
        if data.timestamp:
            try:
                ts = pd.to_datetime(data.timestamp)
            except:
                ts = pd.Timestamp.now()
        else:
            ts = pd.Timestamp.now()
        
        hour = ts.hour
        minute = ts.minute
        
        quarter_hour = hour * 4 + minute // 15
        dependent_temp = data.module_temp - data.ambient_temp
        temp_irradiation = data.ambient_temp * data.irradiation
        temp_time_of_day = data.ambient_temp * hour
        
        if data.irradiation > 0:
            power_per_irradiation = data.dc_power / data.irradiation
        else:
            power_per_irradiation = 0
        
        if np.isinf(power_per_irradiation) or np.isnan(power_per_irradiation):
            power_per_irradiation = 0
        
        irradiation_squared = data.irradiation ** 2
        
        features = {
            'QuarterHour': quarter_hour,
            'DailyYieldSameTimeYesterdayDiff': 0,
            'AverageDailyYieldSummary': data.daily_yield,
            'DependentTemperature': dependent_temp,
            'TemperatureIrradiation': temp_irradiation,
            'TemperatureTimeOfDay': temp_time_of_day,
            'PowerPerIrradiation': power_per_irradiation,
            'IrradiationSquared': irradiation_squared
        }
        
        if self.feature_names:
            df = pd.DataFrame([features])[self.feature_names]
        else:
            df = pd.DataFrame([features])
        
        return df
    
    def _validate_conditions(self, data: SensorData) -> dict:
        """Validate if conditions are suitable for prediction"""
        
        if self.maintenance_mode:
            return {
                'valid': False,
                'reason': 'System in maintenance mode. Predictions disabled.'
            }
        
        if data.irradiation < 100:
            return {
                'valid': False,
                'reason': 'Insufficient sunlight for accurate prediction (irradiation < 100 W/m²)'
            }
        
        if data.dc_power < 100:
            return {
                'valid': False,
                'reason': 'Power output too low for meaningful analysis'
            }
        
        if data.timestamp:
            try:
                ts = pd.to_datetime(data.timestamp)
                hour = ts.hour
                if hour < 8 or hour > 17:
                    return {
                        'valid': False,
                        'reason': 'Outside optimal prediction hours (8 AM - 5 PM)'
                    }
            except:
                pass
        
        if data.module_temp < data.ambient_temp - 5:
            return {
                'valid': False,
                'reason': 'Sensor error: Module temp lower than ambient (possible sensor fault)'
            }
        
        if data.irradiation > 1500:
            return {
                'valid': False,
                'reason': 'Irradiation reading exceeds physical limits (sensor error)'
            }
        
        return {'valid': True, 'reason': None}
    
    def predict(self, data: SensorData) -> dict:
        """Enhanced prediction with false case handling"""
        
        if self.model is None:
            raise ValueError("Model not loaded!")
        
        validation_result = self._validate_conditions(data)
        if not validation_result['valid']:
            return {
                'status': 'unknown',
                'needs_cleaning': False,
                'confidence': 0.0,
                'predicted_power': None,
                'actual_power': float(data.dc_power),
                'power_loss_percentage': 0.0,
                'message': validation_result['reason'],
                'recommendation': 'Cannot determine cleaning status under current conditions',
                'estimated_energy_loss_kwh': 0.0,
                'consecutive_bad_readings': 0,
                'timestamp': datetime.now().isoformat()
            }
        
        X = self.create_features(data)
        
        try:
            predicted_power = self.model.predict(X)[0]
            if isinstance(predicted_power, np.ndarray):
                predicted_power = predicted_power[0]
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
        
        max_system_capacity = 15000
        if predicted_power > max_system_capacity:
            predicted_power = max_system_capacity
        
        actual_power = data.dc_power
        residual = actual_power - predicted_power
        
        if predicted_power > 0:
            error_percentage = (residual / predicted_power) * 100
        else:
            error_percentage = 0
        
        error_percentage = self._apply_temperature_correction(
            error_percentage, 
            data.module_temp
        )
        
        self.history.append(error_percentage)
        if len(self.history) > 7:
            self.history.pop(0)
        
        rolling_avg_error = np.mean(self.history) if self.history else error_percentage
        consecutive_bad = self._count_consecutive_bad_readings()
        
        alert_level = self._determine_alert_level_enhanced(
            error_percentage, 
            rolling_avg_error,
            consecutive_bad
        )
        
        confidence = self._calculate_confidence(data.irradiation)
        
        if confidence < 0.6 and alert_level in ['orange', 'red']:
            alert_level = 'yellow'
        
        recommendation = self._generate_recommendation(alert_level, error_percentage)
        energy_loss = self._estimate_energy_loss(error_percentage, data.irradiation)
        
        return {
            'status': alert_level,
            'needs_cleaning': alert_level in ['orange', 'red'] and consecutive_bad >= 3,
            'confidence': confidence,
            'predicted_power': float(predicted_power),
            'actual_power': float(actual_power),
            'power_loss_percentage': float(abs(error_percentage)),
            'message': self._get_status_message(alert_level),
            'recommendation': recommendation,
            'estimated_energy_loss_kwh': float(energy_loss),
            'consecutive_bad_readings': consecutive_bad,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_status_message(self, alert_level: str) -> str:
        messages = {
            'normal': '✅ Panels operating normally',
            'yellow': '⚠️ Minor performance drop detected',
            'orange': '🟠 Significant underperformance - consider cleaning',
            'red': '🔴 URGENT: Immediate cleaning required!',
            'unknown': '❓ Unable to determine status'
        }
        return messages.get(alert_level, 'Unknown status')
    
    def _generate_recommendation(self, alert_level: str, error_pct: float) -> str:
        if alert_level == 'red':
            return f"Immediate cleaning required! Panels are underperforming by {abs(error_pct):.1f}%. Schedule cleaning within 24 hours to prevent further energy loss."
        elif alert_level == 'orange':
            return f"Cleaning recommended within 2-3 days. Current performance drop: {abs(error_pct):.1f}%."
        elif alert_level == 'yellow':
            return f"Monitor closely. Performance drop: {abs(error_pct):.1f}%. Consider cleaning if condition persists."
        else:
            return "Panels are clean and operating efficiently. No action needed."
    
    def _calculate_confidence(self, irradiation: float) -> float:
        if irradiation > 600:
            return 0.95
        elif irradiation > 300:
            return 0.85
        elif irradiation > 100:
            return 0.70
        else:
            return 0.50
    
    def _estimate_energy_loss(self, error_pct: float, irradiation: float) -> float:
        if error_pct >= 0:
            return 0.0
        
        system_capacity_kw = 10
        peak_sun_hours = 5
        daily_generation = system_capacity_kw * peak_sun_hours
        
        loss_kwh = daily_generation * (abs(error_pct) / 100)
        return loss_kwh
    
    def _apply_temperature_correction(self, error_pct: float, module_temp: float) -> float:
        temp_coefficient = -0.4
        standard_temp = 25.0
        
        if module_temp > standard_temp:
            temp_loss = (module_temp - standard_temp) * temp_coefficient
            corrected_error = error_pct + temp_loss
            return corrected_error
        
        return error_pct
    
    def _count_consecutive_bad_readings(self) -> int:
        count = 0
        for error in reversed(self.history):
            if error < -10:
                count += 1
            else:
                break
        return count
    
    def _determine_alert_level_enhanced(self, error_pct: float, 
                                       rolling_avg: float,
                                       consecutive_bad: int) -> str:
        if rolling_avg < -15 and consecutive_bad >= 3:
            return 'red'
        
        if error_pct < -15 or (rolling_avg < -12 and consecutive_bad >= 2):
            return 'orange'
        
        if error_pct < -7:
            return 'yellow'
        
        return 'normal'

# ============================================
# 3. INITIALIZE MODEL MANAGER
# ============================================

model_manager = SolarPanelModel()

# ============================================
# 4. LIFESPAN EVENT HANDLER (REPLACES on_event)
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    print("🚀 Starting up...")
    try:
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
    
    yield  # Application runs here
    
    # Shutdown
    print("🛑 Shutting down...")

# ============================================
# 5. FASTAPI APPLICATION
# ============================================

app = FastAPI(
    title="Solar Panel Cleaning Detection API",
    description="Real-time ML-powered cleaning recommendation system",
    version="1.0.0",
    lifespan=lifespan  # Use lifespan instead of on_event
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# 6. API ENDPOINTS
# ============================================

@app.get("/", response_model=HealthCheck)
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "ml_model_loaded": model_manager.model is not None,
        "ml_model_type": model_manager.model_type,
        "ml_model_accuracy": model_manager.model_accuracy,
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
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(data: BatchSensorData):
    """Batch prediction for multiple readings"""
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
                "unknown": sum(1 for r in results if r['status'] == 'unknown'),
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/history")
async def get_history():
    """Get recent prediction history"""
    return {
        "recent_errors": model_manager.history,
        "rolling_average": float(np.mean(model_manager.history)) if model_manager.history else 0.0,
        "readings_count": len(model_manager.history)
    }

@app.post("/reset")
async def reset_history():
    """Reset prediction history"""
    model_manager.history = []
    return {"message": "History reset successfully"}

@app.post("/maintenance/start")
async def start_maintenance():
    """Put system in maintenance mode"""
    model_manager.maintenance_mode = True
    return {"message": "Maintenance mode enabled. Predictions disabled."}

@app.post("/maintenance/end")
async def end_maintenance():
    """Exit maintenance mode"""
    model_manager.maintenance_mode = False
    model_manager.history = []
    return {"message": "Maintenance mode disabled. History reset."}

@app.post("/cleaning/confirmed")
async def confirm_cleaning():
    """Reset history after confirmed cleaning"""
    model_manager.history = []
    return {"message": "Cleaning confirmed. History reset."}

# ============================================
# 7. RUN SERVER
# ============================================

if __name__ == "__main__":
    # Run with reload disabled to avoid the warning
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)