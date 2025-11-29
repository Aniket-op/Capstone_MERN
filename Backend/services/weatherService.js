import axios from "axios";
import dotenv from "dotenv";

dotenv.config();

const WEATHER_API_KEY = process.env.WEATHER_API_KEY || "";
const WEATHER_API_URL = process.env.WEATHER_API_URL || "https://api.openweathermap.org/data/2.5/weather";
const DEFAULT_LAT = process.env.LATITUDE || "28.6139"; // Default: New Delhi
const DEFAULT_LON = process.env.LONGITUDE || "77.2090";

/**
 * Fetch weather data from OpenWeatherMap API
 * Returns ambient temperature and solar irradiance (estimated)
 */

export const fetchWeatherData = async () => {
  try {
    // If no API key, return default values
    console.log("WEATHER_API_KEY:", WEATHER_API_KEY);
    if (!WEATHER_API_KEY || WEATHER_API_KEY === "") {
      console.log("⚠️ Weather API key not set, using default values");
      return {
        ambientTemp: 28.5,
        solarIrradiance: 850.0,
        source: "default"
      };
    }

    const url = `${WEATHER_API_URL}?lat=${DEFAULT_LAT}&lon=${DEFAULT_LON}&appid=${WEATHER_API_KEY}&units=metric`;
    
    console.log("🌤️ Fetching weather data from API...");
    const response = await axios.get(url, { timeout: 5000 });
    
    const weatherData = response.data;
    const ambientTemp = weatherData.main.temp;
    
    // Estimate solar irradiance based on:
    // - Time of day (hour)
    // - Cloud coverage
    // - Clear sky conditions
    const hour = new Date().getHours();
    const cloudCover = weatherData.clouds?.all || 0;
    const isDaytime = hour >= 6 && hour <= 18;
    
    // Base irradiance calculation
    let solarIrradiance = 0;
    if (isDaytime) {
      // Peak at noon (12:00), lower at morning/evening
      const hourFactor = Math.sin((hour - 6) * Math.PI / 12);
      const baseIrradiance = 1000 * Math.max(0, hourFactor); // Max 1000 W/m²
      const cloudFactor = (100 - cloudCover) / 100; // Less clouds = more irradiance
      solarIrradiance = baseIrradiance * cloudFactor;
    }
    
    // Clamp between 0 and 1200 W/m²
    solarIrradiance = Math.max(0, Math.min(1200, solarIrradiance));
    
    console.log("✅ Weather data fetched:");
    console.log(`   Ambient Temp: ${ambientTemp}°C`);
    console.log(`   Solar Irradiance: ${solarIrradiance.toFixed(1)} W/m²`);
    console.log(`   Cloud Coverage: ${cloudCover}%`);
    
    return {
      ambientTemp: parseFloat(ambientTemp.toFixed(1)),
      solarIrradiance: parseFloat(solarIrradiance.toFixed(1)),
      cloudCover: cloudCover,
      source: "api"
    };
    
  } catch (error) {
    console.error("❌ Error fetching weather data:", error.message);
    console.log("⚠️ Using default weather values");
    
    // Return default values on error
    return {
      ambientTemp: 28.5,
      solarIrradiance: 850.0,
      source: "fallback"
    };
  }
};

