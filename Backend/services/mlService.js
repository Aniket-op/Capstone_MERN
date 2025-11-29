import axios from "axios";
import SolarData from "../models/SolarData.js";
import { io } from "../server.js"; // Socket.io instance
import { createNotification } from "./notificationService.js";
import mqttService from "./mqttService.js";
import { fetchWeatherData } from "./weatherService.js";

const ML_SERVER_URL = process.env.ML_SERVER_URL || "http://localhost:8000";

// Process sensor data: Save to DB + Get ML prediction
export const processSensorData = async (sensorData) => {
  try {
    console.log("🔄 Processing sensor data from hardware...");
    console.log("📦 Hardware data received:", JSON.stringify(sensorData, null, 2));

    // Step 1: Extract hardware data (module_temp, current, voltage, humidity, power)
    const hardwareData = {
      module_temp: sensorData.module_temp || sensorData.moduleTemperature,
      current: sensorData.current,
      voltage: sensorData.voltage,
      humidity: sensorData.humidity,
      power: sensorData.power,
    };

    // Step 2: Fetch weather data (ambientTemp, solarIrradiance)
    console.log("🌤️ Fetching weather data...");
    const weatherData = await fetchWeatherData();
    const ambientTemp = weatherData.ambientTemp;
    const solarIrradiance = weatherData.solarIrradiance;

    // Step 3: Calculate powerGeneration = power * 0.75
    const powerGeneration = (hardwareData.power || 0) * 0.75;
    console.log(`⚡ Calculated powerGeneration: ${powerGeneration.toFixed(2)} W (power * 0.75)`);

    // Step 4: Map to ML server format
    const mlPayload = {
      timestamp: new Date().toISOString(),
      ambient_temp: ambientTemp,
      module_temp: hardwareData.module_temp,
      irradiation: solarIrradiance,
      dc_power: powerGeneration, // Calculated powerGeneration
      daily_yield: sensorData.dailyYield || 0,
    };

    console.log("📤 Sending to ML server:", JSON.stringify(mlPayload, null, 2));

    // Step 2: Get prediction from ML server
    const mlResponse = await axios.post(
      `${ML_SERVER_URL}/predict`,
      mlPayload,
      { timeout: 10000 }
    );

    const prediction = mlResponse.data;
    console.log("📥 ML Prediction received:");
    console.log(JSON.stringify(prediction, null, 2));

    // Step 5: Save to MongoDB with all data
    const solarDataRecord = new SolarData({
      temperature: ambientTemp, // From weather API
      humidity: hardwareData.humidity,
      current: hardwareData.current,
      voltage: hardwareData.voltage,
      solarIrradiance: solarIrradiance, // From weather API
      powerGeneration: powerGeneration, // Calculated: power * 0.75
      powerActual: prediction.actual_power,
      powerPredicted: prediction.predicted_power,
      panelEfficiency: calculateEfficiency({ solarIrradiance, powerGeneration }),
      dailyYield: sensorData.dailyYield || 0,
      cleaningDays: sensorData.cleaningDays || 0,
      moduleTemp: hardwareData.module_temp,
    });

    const savedData = await solarDataRecord.save();
    console.log("💾 Data saved to MongoDB successfully!");
    console.log("   Document ID:", savedData._id);
    console.log("   Temperature:", savedData.temperature);
    console.log("   Power Generation:", savedData.powerGeneration);
    console.log("   Timestamp:", savedData.createdAt);

    // Step 6: Emit real-time update to frontend with full ML message
    io.emit("solarDataUpdate", {
      sensorData: savedData,
      prediction: prediction,
    });

    // Step 7: Emit ML server full message for toast notification
    io.emit("mlMessage", {
      message: prediction.message,
      recommendation: prediction.recommendation,
      status: prediction.status,
      needs_cleaning: prediction.needs_cleaning,
      confidence: prediction.confidence,
      power_loss_percentage: prediction.power_loss_percentage,
      estimated_energy_loss_kwh: prediction.estimated_energy_loss_kwh,
      timestamp: prediction.timestamp,
      fullMessage: `${prediction.message} - ${prediction.recommendation}`,
    });

    // Step 5: Handle cleaning alerts
    if (prediction.needs_cleaning) {
      console.log("🚨 Cleaning required!");
      
      // Create notification with full ML message
      await createNotification({
        message: `${prediction.message} - ${prediction.recommendation}`,
        status: prediction.status,
        powerLoss: prediction.power_loss_percentage,
        fullMLMessage: JSON.stringify(prediction, null, 2), // Store full ML response
      });

      // Emit alert to frontend
      io.emit("cleaningAlert", {
        status: prediction.status,
        message: prediction.message,
        recommendation: prediction.recommendation,
        powerLoss: prediction.power_loss_percentage,
        estimatedLoss: prediction.estimated_energy_loss_kwh,
      });

      // If Red alert, send command to hardware (optional)
      if (prediction.status === "red") {
        mqttService.publishCommand({
          action: "ALERT",
          level: "RED",
          message: "Immediate cleaning required",
        });
      }
    }

    return {
      saved: savedData,
      prediction: prediction,
    };

  } catch (error) {
    console.error("❌ Error processing sensor data:", error.message);
    
    // Save data even if ML prediction fails
    try {
      // Fetch weather data for fallback
      const weatherData = await fetchWeatherData();
      const powerGeneration = (sensorData.power || 0) * 0.75;
      
      const fallbackData = new SolarData({
        temperature: weatherData.ambientTemp,
        humidity: sensorData.humidity,
        current: sensorData.current,
        voltage: sensorData.voltage,
        solarIrradiance: weatherData.solarIrradiance,
        powerGeneration: powerGeneration,
        dailyYield: sensorData.dailyYield || 0,
        moduleTemp: sensorData.module_temp,
      });
      const saved = await fallbackData.save();
      console.log("💾 Data saved (without ML prediction)");
      console.log("   Document ID:", saved._id);
      console.log("   Temperature:", saved.temperature);
      console.log("   Power Generation:", saved.powerGeneration);
    } catch (dbError) {
      console.error("❌ Failed to save data:", dbError);
    }

    throw error;
  }
};

// Calculate panel efficiency
const calculateEfficiency = (sensorData) => {
  if (!sensorData.solarIrradiance || sensorData.solarIrradiance === 0) {
    return 0;
  }
  
  const panelArea = 1.6; // m² (adjust for your panel)
  const maxPower = sensorData.solarIrradiance * panelArea;
  const efficiency = (sensorData.powerGeneration / maxPower) * 100;
  
  return Math.min(efficiency, 100); // Cap at 100%
};

// Manual prediction request
export const getPrediction = async (sensorData) => {
  try {
    const response = await axios.post(
      `${ML_SERVER_URL}/predict`,
      sensorData,
      { timeout: 10000 }
    );
    return response.data;
  } catch (error) {
    console.error("❌ ML prediction error:", error.message);
    throw new Error("ML server unavailable");
  }
};

// Check ML server health
export const checkMLServerHealth = async () => {
  try {
    const response = await axios.get(`${ML_SERVER_URL}/`, { timeout: 5000 });
    return {
      status: "online",
      data: response.data,
    };
  } catch (error) {
    return {
      status: "offline",
      error: error.message,
    };
  }
};