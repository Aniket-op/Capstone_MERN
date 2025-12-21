// import express from "express";
// import dotenv from "dotenv";
// import cors from "cors";
// import connectDB from "./config/db.js"; // <-- note the .js
// import authRoutes from "./routes/authRoutes.js";
// import solarRoutes from "./routes/solarRoutes.js";
// import solar from "./routes/solar.js";
// import { errorHandler } from "./middlewares/errorMiddleware.js";
// import notificationRoutes from "./routes/notificationRoutes.js";
// dotenv.config();
// connectDB();

// const app = express();
// app.use(cors());
// app.use(express.json());

// app.use("/api/auth", authRoutes);
// app.use("/api/solar", solarRoutes);
// app.use("/api/demo/solar", solar);
// app.use("/api/notifications", notificationRoutes);



// app.use(errorHandler);

// const PORT = process.env.PORT || 5000;
// app.listen(PORT, () => console.log(`Server running on port ${PORT}`));

import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import { createServer } from "http";
import { Server } from "socket.io";
import connectDB from "./config/db.js";
import mqttService from "./services/mqttService.js";

// Routes
import authRoutes from "./routes/authRoutes.js";
import solarDataRoutes from "./routes/solarDataRoutes.js";
import notificationRoutes from "./routes/notificationRoutes.js";

dotenv.config();

// Increase max listeners to avoid warning from nodemon
process.setMaxListeners(15);

const app = express();
const httpServer = createServer(app);

// Socket.IO setup
export const io = new Server(httpServer, {
  cors: {
    origin: process.env.FRONTEND_URL || "http://localhost:3000",
    methods: ["GET", "POST"],
  },
});

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Connect to MongoDB
connectDB();

// Connect to MQTT
mqttService.connect();

// Scheduled task to request hardware data every hour
let dataRequestInterval = null;

const requestHardwareData = () => {
  if (mqttService.isConnected) {
    mqttService.publishCommand({
      action: "REQUEST_DATA",
      timestamp: new Date().toISOString(),
      source: "backend_scheduler"
    });
    console.log("📡 Requested hardware data via MQTT");
  } else {
    console.log("⚠️ MQTT not connected, skipping data request");
  }
};

// Start requesting data every hour (3600000 ms = 1 hour)
// Wait 30 seconds after server starts to ensure MQTT is connected
setTimeout(() => {
  // Request data immediately on first connection
  requestHardwareData();
  
  // Then request every hour
  dataRequestInterval = setInterval(() => {
    requestHardwareData();
  }, 36000); // 1 hour = 3600000 milliseconds
  
  console.log("⏰ Scheduled data requests: Every 1 hour");
}, 30000); // Wait 30 seconds before starting

// Socket.IO connection
io.on("connection", (socket) => {
  console.log(`✅ Client connected: ${socket.id}`);
  
  socket.on("disconnect", () => {
    console.log(`❌ Client disconnected: ${socket.id}`);
  });
});

// Routes
app.use("/api/auth", authRoutes);
app.use("/api/solar", solarDataRoutes);
app.use("/api/notifications", notificationRoutes);

// Root-level MQTT routes (matching test code for ESP32 compatibility)
app.get("/latest", (req, res) => {
  const latestMessage = mqttService.getLatestMessage();
  if (!latestMessage) {
    return res.json({ message: "No data received yet from ESP32" });
  }
  res.status(200).json({
    status: "success",
    timestamp: latestMessage.timestamp,
    data: latestMessage.data,
  });
});

app.post("/command", (req, res) => {
  const { command } = req.body;
  if (!command) {
    return res.status(400).json({ error: "Missing command" });
  }

  const result = mqttService.publishCommand(command);
  if (!result) {
    return res.status(500).json({
      status: "error",
      message: "Failed to send command",
    });
  }

  res.json({ status: "success", sent: command });
});

// Health check
app.get("/", (req, res) => {
  res.json({
    message: "Solar Panel Backend API",
    status: "running",
    mqtt: mqttService.isConnected ? "connected" : "disconnected",
  });
});

// Error handling
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    success: false,
    error: err.message || "Internal server error",
  });
});

const PORT = process.env.PORT || 5000;

httpServer.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});

// Graceful shutdown
process.on("SIGINT", () => {
  console.log("\n🛑 Shutting down gracefully...");
  
  // Clear the data request interval
  if (dataRequestInterval) {
    clearInterval(dataRequestInterval);
    console.log("🛑 Stopped scheduled data requests");
  }
  
  mqttService.disconnect();
  process.exit(0);
});