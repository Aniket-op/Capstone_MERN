import SolarData from "../models/SolarData.js";

// Get all solar data
export const getSolarData = async (req, res) => {
  try {
    const data = await SolarData.find().sort({ createdAt: -1 });
    res.status(200).json(data);
  } catch (err) {
    console.error("Error fetching solar data:", err);
    res.status(500).json({ error: "Failed to fetch solar data" });
  }
};

// Post new solar data
export const postSolarData = async (req, res) => {
  try {
    const {
      temperature,
      humidity,
      current,
      solarIrradiance,
      powerGeneration,
      powerActual,
      powerPredicted,
      panelEfficiency,
      dailyYield,
      cleaningDays,
      createdAt,
    } = req.body;

    const newData = new SolarData({
      temperature,
      humidity,
      current,
      solarIrradiance,
      powerGeneration,
      powerActual,
      powerPredicted,
      panelEfficiency,
      dailyYield,
      cleaningDays,
      createdAt,
    });

    const savedData = await newData.save();

    res.status(201).json({
      message: "Solar data added successfully",
      data: savedData,
    });
  } catch (err) {
    console.error("Error saving solar data:", err);
    res.status(500).json({ error: "Failed to save solar data" });
  }
};
