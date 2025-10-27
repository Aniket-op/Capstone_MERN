import express from "express";
import SolarData from "../models/solarData.js";

const router = express.Router();

// 🔹 Demo route - returns mock data if DB is empty
router.get("/", async (req, res) => {
  try {
    // Fetch last 20 records
    let data = await SolarData.find().sort({ createdAt: 1 }).limit(20);

    // If no data exists, create mock demo data
    if (data.length === 0) {
      const demoData = [];
      for (let i = 0; i < 20; i++) {
        demoData.push({
          temperature: (20 + Math.random() * 15).toFixed(1), // 20–35 °C
          humidity: (30 + Math.random() * 50).toFixed(1), // 30–80 %
          current: (5 + Math.random() * 10).toFixed(2), // 5–15 A
          solarIrradiance: (600 + Math.random() * 400).toFixed(1), // 600–1000 W/m²
          powerGeneration: (Math.random() * 5).toFixed(2), // 0–5 kW
          powerActual: (Math.random() * 5).toFixed(2),
          powerPredicted: (Math.random() * 5).toFixed(2),
          panelEfficiency: (90 + Math.random() * 5).toFixed(2), // 90–95 %
          dailyYield: (10 + Math.random() * 15).toFixed(2), // 10–25 kWh
          cleaningDays: Math.floor(Math.random() * 5), // 0–5 kW
        });
      }
      data = await SolarData.insertMany(demoData);
    }

    res.json(data);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Server error" });
  }
});
// server/routes/demo.js
router.get("/notifications", async (req, res) => {
  res.json([
    {
      title: "Panel Cleaned",
      message: "Solar panel cleaning completed successfully.",
      createdAt: new Date(),
    },
    {
      title: "Low Power Alert",
      message: "Power generation dropped below threshold.",
      createdAt: new Date(Date.now() - 3600000),
    },
  ]);
});

export default router;
