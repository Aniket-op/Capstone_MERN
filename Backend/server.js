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
  mqttService.disconnect();
  process.exit(0);
});