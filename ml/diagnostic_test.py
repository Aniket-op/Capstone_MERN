# diagnostic_test.py
import requests

BASE_URL = "http://localhost:8000"

# Reset history
requests.post(f"{BASE_URL}/reset")

print("="*70)
print("MODEL DIAGNOSTIC TEST")
print("="*70)

# Test with heavy dirt
data = {
    "timestamp": "2024-11-08T12:00:00",
    "ambient_temp": 28.5,
    "module_temp": 45.2,
    "irradiation": 850.0,
    "dc_power": 9400.0,
    "daily_yield": 34.0
}

for i in range(7):
    r = requests.post(f"{BASE_URL}/predict", json=data)
    result = r.json()
    print(f"Reading {i+1}: "
          f"Predicted={result.get('predicted_power', 0):.0f}W, "
          f"Loss={result['power_loss_percentage']:.1f}%, "
          f"Status={result['status']}, "
          f"Consecutive={result['consecutive_bad_readings']}")

print("="*70)