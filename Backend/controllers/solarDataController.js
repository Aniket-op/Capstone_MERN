import SolarData from "../models/SolarData.js";
import Notification from "../models/Notification.js";
import { getPrediction, checkMLServerHealth } from "../services/mlService.js";
import { io } from "../server.js";

// Get all solar data with pagination
export const getSolarData = async (req, res) => {
  try {
    const { limit = 100, skip = 0, startDate, endDate } = req.query;
    
    // Build query filter
    const filter = {};
    if (startDate || endDate) {
      filter.createdAt = {};
      if (startDate) filter.createdAt.$gte = new Date(startDate);
      if (endDate) filter.createdAt.$lte = new Date(endDate);
    }
    
    const data = await SolarData.find(filter)
      .sort({ createdAt: -1 })
      .limit(parseInt(limit))
      .skip(parseInt(skip));
    
    const total = await SolarData.countDocuments(filter);
    
    res.status(200).json({
      success: true,
      count: data.length,
      total: total,
      data: data,
    });
  } catch (err) {
    console.error("Error fetching solar data:", err);
    res.status(500).json({ 
      success: false,
      error: "Failed to fetch solar data" 
    });
  }
};

// Get latest solar data
export const getLatestSolarData = async (req, res) => {
  try {
    const latestData = await SolarData.findOne().sort({ createdAt: -1 });
    
    if (!latestData) {
      return res.status(404).json({
        success: false,
        message: "No data available",
      });
    }
    
    res.status(200).json({
      success: true,
      data: latestData,
    });
  } catch (err) {
    console.error("Error fetching latest data:", err);
    res.status(500).json({ 
      success: false,
      error: "Failed to fetch latest data" 
    });
  }
};

// Get solar data statistics
export const getSolarDataStats = async (req, res) => {
  try {
    const { days = 7 } = req.query;
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - parseInt(days));
    
    const data = await SolarData.find({
      createdAt: { $gte: startDate }
    }).sort({ createdAt: 1 });
    
    if (data.length === 0) {
      return res.status(200).json({
        success: true,
        stats: {
          avgPowerGeneration: 0,
          avgEfficiency: 0,
          totalYield: 0,
          dataPoints: 0,
        },
      });
    }
    
    // Calculate statistics
    const stats = {
      avgPowerGeneration: data.reduce((sum, d) => sum + (d.powerGeneration || 0), 0) / data.length,
      avgEfficiency: data.reduce((sum, d) => sum + (d.panelEfficiency || 0), 0) / data.length,
      totalYield: data.reduce((sum, d) => sum + (d.dailyYield || 0), 0),
      maxPower: Math.max(...data.map(d => d.powerGeneration || 0)),
      minPower: Math.min(...data.map(d => d.powerGeneration || 0)),
      avgTemperature: data.reduce((sum, d) => sum + (d.temperature || 0), 0) / data.length,
      avgIrradiance: data.reduce((sum, d) => sum + (d.solarIrradiance || 0), 0) / data.length,
      dataPoints: data.length,
      period: `Last ${days} days`,
    };
    
    res.status(200).json({
      success: true,
      stats: stats,
    });
  } catch (err) {
    console.error("Error calculating stats:", err);
    res.status(500).json({ 
      success: false,
      error: "Failed to calculate statistics" 
    });
  }
};

// Get today's yield and last cleaning status
export const getTodayStats = async (req, res) => {
  try {
    // Get start of today
    const todayStart = new Date();
    todayStart.setHours(0, 0, 0, 0);
    
    // Get all records from today
    const todayData = await SolarData.find({
      createdAt: { $gte: todayStart }
    }).sort({ createdAt: 1 });
    
    // Calculate today's total yield
    let todayYield = 0;
    if (todayData.length > 0) {
      // First, try to use the latest dailyYield if it's set and > 0
      const latestDailyYield = todayData[todayData.length - 1].dailyYield;
      
      // Option 2: Calculate from power generation (sum powerGeneration in kW, convert to kWh)
      // Assuming data comes every few minutes, estimate total energy
      // Power (kW) * time (hours) = Energy (kWh)
      // For simplicity, if we have hourly data: sum(powerGeneration) / 1000 * hours
      // Or use the latest dailyYield which should be cumulative
      
      if (latestDailyYield && latestDailyYield > 0) {
        todayYield = latestDailyYield;
      } else {
        // Fallback: sum all powerGeneration values and estimate
        // This is approximate - assumes data points represent hourly averages
        const totalPower = todayData.reduce((sum, d) => sum + (d.powerGeneration || 0), 0);
        // If data comes every 10 seconds, and we have X data points:
        // Each point represents ~10 seconds of generation
        // Convert: (power in W * time in hours) / 1000 = kWh
        const hoursOfData = todayData.length * (10 / 3600); // Assuming 10 second intervals
        todayYield = (totalPower * hoursOfData) / 1000; // Convert W to kW, then to kWh
      }
    }
    
    // Find last cleaning date from notifications (where userResponse === "yes")
    const lastCleaning = await Notification.findOne({
      userResponse: "yes"
    }).sort({ updatedAt: -1 });
    
    let lastCleaningDays = null;
    let lastCleaningDate = null;
    
    if (lastCleaning) {
      lastCleaningDate = lastCleaning.updatedAt;
      const daysSince = Math.floor((Date.now() - lastCleaningDate.getTime()) / (1000 * 60 * 60 * 24));
      lastCleaningDays = daysSince;
    } else {
      // Fallback: check latest record's cleaningDays
      if (todayData.length > 0) {
        lastCleaningDays = todayData[todayData.length - 1].cleaningDays;
        if (lastCleaningDays !== null && lastCleaningDays !== undefined) {
          console.log(`🧹 Using cleaningDays from latest record: ${lastCleaningDays} days`);
        }
      }
    }
    
    console.log(`📊 Today's yield: ${todayYield.toFixed(2)} kWh, Last cleaning: ${lastCleaningDays} days ago`);
    
    res.status(200).json({
      success: true,
      todayYield: parseFloat(todayYield.toFixed(2)),
      lastCleaningDays: lastCleaningDays,
      lastCleaningDate: lastCleaningDate,
      dataPoints: todayData.length,
    });
  } catch (err) {
    console.error("Error getting today's stats:", err);
    res.status(500).json({ 
      success: false,
      error: "Failed to get today's statistics" 
    });
  }
};

// Post new solar data (manual entry)
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
    });

    const savedData = await newData.save();
    
    // Emit real-time update
    io.emit("solarDataUpdate", savedData);

    res.status(201).json({
      success: true,
      message: "Solar data added successfully",
      data: savedData,
    });
  } catch (err) {
    console.error("Error saving solar data:", err);
    res.status(500).json({ 
      success: false,
      error: "Failed to save solar data" 
    });
  }
};

// Get ML prediction for current conditions
export const getPredictionForData = async (req, res) => {
  try {
    const { 
      ambient_temp, 
      module_temp, 
      irradiation, 
      dc_power, 
      daily_yield 
    } = req.body;

    // Validate required fields
    if (!ambient_temp || !module_temp || !irradiation || !dc_power) {
      return res.status(400).json({
        success: false,
        error: "Missing required fields: ambient_temp, module_temp, irradiation, dc_power",
      });
    }

    const mlPayload = {
      timestamp: new Date().toISOString(),
      ambient_temp: parseFloat(ambient_temp),
      module_temp: parseFloat(module_temp),
      irradiation: parseFloat(irradiation),
      dc_power: parseFloat(dc_power),
      daily_yield: parseFloat(daily_yield) || 0,
    };

    const prediction = await getPrediction(mlPayload);

    res.status(200).json({
      success: true,
      prediction: prediction,
    });
  } catch (err) {
    console.error("Error getting prediction:", err);
    res.status(500).json({ 
      success: false,
      error: err.message || "Failed to get prediction from ML server" 
    });
  }
};

// Delete old solar data (cleanup)
export const deleteOldSolarData = async (req, res) => {
  try {
    const { days = 30 } = req.query;
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - parseInt(days));
    
    const result = await SolarData.deleteMany({
      createdAt: { $lt: cutoffDate }
    });
    
    res.status(200).json({
      success: true,
      message: `Deleted ${result.deletedCount} records older than ${days} days`,
      deletedCount: result.deletedCount,
    });
  } catch (err) {
    console.error("Error deleting old data:", err);
    res.status(500).json({ 
      success: false,
      error: "Failed to delete old data" 
    });
  }
};

// Get ML server health status
export const getMLServerStatus = async (req, res) => {
  try {
    const health = await checkMLServerHealth();
    res.status(200).json({
      success: true,
      mlServer: health,
    });
  } catch (err) {
    console.error("Error checking ML server:", err);
    res.status(500).json({ 
      success: false,
      error: "Failed to check ML server status" 
    });
  }
};

// Get chart data for dashboard
export const getChartData = async (req, res) => {
  try {
    const { hours = 24 } = req.query;
    const startDate = new Date();
    startDate.setHours(startDate.getHours() - parseInt(hours));
    
    const data = await SolarData.find({
      createdAt: { $gte: startDate }
    })
    .sort({ createdAt: 1 })
    .select('createdAt powerGeneration powerPredicted powerActual temperature solarIrradiance panelEfficiency');
    
    // Format for charts
    const chartData = {
      labels: data.map(d => d.createdAt),
      powerGeneration: data.map(d => d.powerGeneration || 0),
      powerPredicted: data.map(d => d.powerPredicted || 0),
      powerActual: data.map(d => d.powerActual || 0),
      temperature: data.map(d => d.temperature || 0),
      irradiance: data.map(d => d.solarIrradiance || 0),
      efficiency: data.map(d => d.panelEfficiency || 0),
    };
    
    res.status(200).json({
      success: true,
      period: `Last ${hours} hours`,
      dataPoints: data.length,
      chartData: chartData,
    });
  } catch (err) {
    console.error("Error fetching chart data:", err);
    res.status(500).json({ 
      success: false,
      error: "Failed to fetch chart data" 
    });
  }
};

// Export data as CSV
export const exportSolarData = async (req, res) => {
  try {
    const { startDate, endDate } = req.query;
    
    const filter = {};
    if (startDate || endDate) {
      filter.createdAt = {};
      if (startDate) filter.createdAt.$gte = new Date(startDate);
      if (endDate) filter.createdAt.$lte = new Date(endDate);
    }
    
    const data = await SolarData.find(filter).sort({ createdAt: 1 });
    
    // Convert to CSV
    const csvHeader = 'Timestamp,Temperature,Humidity,Current,Solar Irradiance,Power Generation,Power Actual,Power Predicted,Panel Efficiency,Daily Yield,Cleaning Days\n';
    
    const csvRows = data.map(d => 
      `${d.createdAt},${d.temperature},${d.humidity},${d.current},${d.solarIrradiance},${d.powerGeneration},${d.powerActual},${d.powerPredicted},${d.panelEfficiency},${d.dailyYield},${d.cleaningDays}`
    ).join('\n');
    
    const csv = csvHeader + csvRows;
    
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', 'attachment; filename=solar_data.csv');
    res.status(200).send(csv);
    
  } catch (err) {
    console.error("Error exporting data:", err);
    res.status(500).json({ 
      success: false,
      error: "Failed to export data" 
    });
  }
};