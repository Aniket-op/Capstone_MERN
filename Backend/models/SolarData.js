import mongoose from "mongoose";

const solarDataSchema = new mongoose.Schema(
  {
    temperature: { type: Number },
    humidity: { type: Number },
    current: { type: Number },
    solarIrradiance: { type: Number },
    powerGeneration: { type: Number },
    powerActual: { type: Number },
    powerPredicted: { type: Number },
    panelEfficiency: { type: Number },
    dailyYield: { type: Number },
    cleaningDays: { type: Number },
    createdAt: { type: Date, default: Date.now },
  },
  { timestamps: true }
);

export default mongoose.models.SolarData ||
  mongoose.model("SolarData", solarDataSchema);
