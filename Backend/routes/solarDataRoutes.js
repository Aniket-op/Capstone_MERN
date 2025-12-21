// import express from "express";
// import { getSolarData , postSolarData } from "../controllers/solarController.js";
// import {protect}  from "../middlewares/authMiddleware.js";

// const router = express.Router();

// router.get("/", protect, getSolarData);
// router.post("/", protect, postSolarData);


// export default router;

import express from "express";
import {
  getSolarData,
  getLatestSolarData,
  getSolarDataStats,
  getTodayStats,
  postSolarData,
  getPredictionForData,
  deleteOldSolarData,
  getMLServerStatus,
  getChartData,
  exportSolarData,
  getLatestMQTTMessage,
  sendCommandToESP32,
} from "../controllers/solarDataController.js";
import { protect } from "../middlewares/authMiddleware.js";

const router = express.Router();

// Public routes
router.get("/", getSolarData);
router.get("/latest", getLatestSolarData);
router.get("/stats", getSolarDataStats);
router.get("/today", getTodayStats);
router.get("/chart", getChartData);
router.get("/export", exportSolarData);

// MQTT routes for ESP32 hardware
router.get("/mqtt/latest", getLatestMQTTMessage); // Get latest MQTT message from ESP32
router.post("/mqtt/command", sendCommandToESP32); // Send command to ESP32

// Protected routes
router.post("/", protect, postSolarData);
router.post("/predict", protect, getPredictionForData);
router.delete("/cleanup", protect, deleteOldSolarData);

// ML server status
router.get("/ml-status", getMLServerStatus);

export default router;