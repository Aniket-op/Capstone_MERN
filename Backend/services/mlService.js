import axios from "axios";
import SolarData from "../models/SolarData.js";
import { io } from "../server.js"; // Socket.io instance
import { createNotification } from "./notificationService.js";
import mqttService from "./mqttService.js";

const ML_SERVER_URL = process.env.ML_SERVER_URL || "http://localhost:8000";

// Process sensor data: Save to DB + Get ML prediction
export const processSensorData = async (sensorData) => {
  try {
    console.log("🔄 Processing sensor data...");

    // Step 1: Map hardware data to ML server format
    const mlPayload = {
      timestamp: new Date().toISOString(),
      ambient_temp: sensorData.temperature || 28.5,
      module_temp: sensorData.moduleTemperature || sensorData.temperature + 15,
      irradiation: sensorData.solarIrradiance || 850.0,
      dc_power: sensorData.powerGeneration || sensorData.current * 12, // V*I
      daily_yield: sensorData.dailyYield || 0,
    };

    console.log("📤 Sending to ML server:", mlPayload);

    // Step 2: Get prediction from ML server
    const mlResponse = await axios.post(
      `${ML_SERVER_URL}/predict`,
      mlPayload,
      { timeout: 10000 }
    );

    const prediction = mlResponse.data;
    console.log("📥 ML Prediction:", prediction);

    // Step 3: Save to MongoDB
    const solarDataRecord = new SolarData({
      temperature: sensorData.temperature,
      humidity: sensorData.humidity,
      current: sensorData.current,
      solarIrradiance: sensorData.solarIrradiance,
      powerGeneration: sensorData.powerGeneration,
      powerActual: prediction.actual_power,
      powerPredicted: prediction.predicted_power,
      panelEfficiency: calculateEfficiency(sensorData),
      dailyYield: sensorData.dailyYield,
      cleaningDays: sensorData.cleaningDays || 0,
    });

    const savedData = await solarDataRecord.save();
    console.log("💾 Data saved to MongoDB:", savedData._id);

    // Step 4: Emit real-time update to frontend
    io.emit("solarDataUpdate", {
      sensorData: savedData,
      prediction: prediction,
    });

    // Step 5: Handle cleaning alerts
    if (prediction.needs_cleaning) {
      console.log("🚨 Cleaning required!");
      
      // Create notification
      await createNotification({
        message: `${prediction.message} - ${prediction.recommendation}`,
        status: prediction.status,
        powerLoss: prediction.power_loss_percentage,
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
      const fallbackData = new SolarData({
        temperature: sensorData.temperature,
        humidity: sensorData.humidity,
        current: sensorData.current,
        solarIrradiance: sensorData.solarIrradiance,
        powerGeneration: sensorData.powerGeneration,
        dailyYield: sensorData.dailyYield,
      });
      await fallbackData.save();
      console.log("💾 Data saved (without ML prediction)");
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