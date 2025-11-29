import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Create logs directory if it doesn't exist
const logsDir = path.join(__dirname, "../logs");
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true });
}

// Log levels
const LOG_LEVELS = {
  ERROR: "ERROR",
  WARN: "WARN",
  INFO: "INFO",
  DEBUG: "DEBUG",
  SUCCESS: "SUCCESS",
};

// Colors for console output
const COLORS = {
  ERROR: "\x1b[31m", // Red
  WARN: "\x1b[33m", // Yellow
  INFO: "\x1b[36m", // Cyan
  DEBUG: "\x1b[35m", // Magenta
  SUCCESS: "\x1b[32m", // Green
  RESET: "\x1b[0m",
};

class Logger {
  constructor() {
    this.logFile = path.join(logsDir, `app-${this.getDateString()}.log`);
    this.errorFile = path.join(logsDir, `error-${this.getDateString()}.log`);
    this.mqttFile = path.join(logsDir, `mqtt-${this.getDateString()}.log`);
    this.mlFile = path.join(logsDir, `ml-${this.getDateString()}.log`);
  }

  // Get current date string for log file names
  getDateString() {
    const now = new Date();
    return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
  }

  // Get formatted timestamp
  getTimestamp() {
    return new Date().toISOString();
  }

  // Format log message
  formatMessage(level, message, meta = {}) {
    const timestamp = this.getTimestamp();
    const metaString = Object.keys(meta).length > 0 ? ` | ${JSON.stringify(meta)}` : "";
    return `[${timestamp}] [${level}] ${message}${metaString}`;
  }

  // Write to file
  writeToFile(filename, message) {
    try {
      fs.appendFileSync(filename, message + "\n", "utf8");
    } catch (error) {
      console.error("Failed to write to log file:", error);
    }
  }

  // Console output with colors
  consoleOutput(level, message) {
    const color = COLORS[level] || COLORS.RESET;
    console.log(`${color}${message}${COLORS.RESET}`);
  }

  // Generic log method
  log(level, message, meta = {}, fileOverride = null) {
    const formattedMessage = this.formatMessage(level, message, meta);
    
    // Console output
    this.consoleOutput(level, formattedMessage);
    
    // File output
    const targetFile = fileOverride || this.logFile;
    this.writeToFile(targetFile, formattedMessage);
    
    // Also write errors to error file
    if (level === LOG_LEVELS.ERROR) {
      this.writeToFile(this.errorFile, formattedMessage);
    }
  }

  // Convenience methods
  error(message, meta = {}) {
    this.log(LOG_LEVELS.ERROR, message, meta);
  }

  warn(message, meta = {}) {
    this.log(LOG_LEVELS.WARN, message, meta);
  }

  info(message, meta = {}) {
    this.log(LOG_LEVELS.INFO, message, meta);
  }

  debug(message, meta = {}) {
    if (process.env.NODE_ENV === "development") {
      this.log(LOG_LEVELS.DEBUG, message, meta);
    }
  }

  success(message, meta = {}) {
    this.log(LOG_LEVELS.SUCCESS, message, meta);
  }

  // MQTT specific logging
  mqtt(message, meta = {}) {
    const formattedMessage = this.formatMessage("MQTT", message, meta);
    this.consoleOutput(LOG_LEVELS.INFO, formattedMessage);
    this.writeToFile(this.mqttFile, formattedMessage);
  }

  // ML server specific logging
  ml(message, meta = {}) {
    const formattedMessage = this.formatMessage("ML", message, meta);
    this.consoleOutput(LOG_LEVELS.INFO, formattedMessage);
    this.writeToFile(this.mlFile, formattedMessage);
  }

  // HTTP request logging
  request(req, res, duration) {
    const message = `${req.method} ${req.originalUrl} - ${res.statusCode} - ${duration}ms`;
    const meta = {
      ip: req.ip,
      userAgent: req.get("user-agent"),
      userId: req.user?._id,
    };
    this.info(message, meta);
  }

  // Database operation logging
  db(operation, collection, meta = {}) {
    const message = `DB ${operation} on ${collection}`;
    this.info(message, meta);
  }

  // Sensor data logging
  sensor(data) {
    const message = "Sensor data received";
    const meta = {
      temperature: data.temperature,
      irradiance: data.solarIrradiance,
      power: data.powerGeneration,
    };
    this.mqtt(message, meta);
  }

  // Prediction logging
  prediction(input, output) {
    const message = "ML prediction completed";
    const meta = {
      input: {
        dc_power: input.dc_power,
        irradiation: input.irradiation,
      },
      output: {
        status: output.status,
        needs_cleaning: output.needs_cleaning,
        loss: output.power_loss_percentage,
      },
    };
    this.ml(message, meta);
  }

  // Alert logging
  alert(level, message, meta = {}) {
    const alertMessage = `🚨 ALERT [${level}]: ${message}`;
    this.warn(alertMessage, meta);
  }

  // Clean old log files (keep last 30 days)
  cleanOldLogs(daysToKeep = 30) {
    try {
      const files = fs.readdirSync(logsDir);
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - daysToKeep);

      files.forEach((file) => {
        const filePath = path.join(logsDir, file);
        const stats = fs.statSync(filePath);
        
        if (stats.mtime < cutoffDate) {
          fs.unlinkSync(filePath);
          console.log(`Deleted old log file: ${file}`);
        }
      });
    } catch (error) {
      console.error("Error cleaning old logs:", error);
    }
  }

  // Get log file contents
  getLogFile(type = "app", date = null) {
    try {
      const dateString = date || this.getDateString();
      let filename;

      switch (type) {
        case "error":
          filename = `error-${dateString}.log`;
          break;
        case "mqtt":
          filename = `mqtt-${dateString}.log`;
          break;
        case "ml":
          filename = `ml-${dateString}.log`;
          break;
        default:
          filename = `app-${dateString}.log`;
      }

      const filePath = path.join(logsDir, filename);
      
      if (fs.existsSync(filePath)) {
        return fs.readFileSync(filePath, "utf8");
      }
      
      return null;
    } catch (error) {
      console.error("Error reading log file:", error);
      return null;
    }
  }

  // Get all log files
  getAllLogFiles() {
    try {
      const files = fs.readdirSync(logsDir);
      return files.map((file) => {
        const filePath = path.join(logsDir, file);
        const stats = fs.statSync(filePath);
        return {
          name: file,
          size: stats.size,
          modified: stats.mtime,
        };
      });
    } catch (error) {
      console.error("Error listing log files:", error);
      return [];
    }
  }

  // Performance monitoring
  startTimer(label) {
    const start = Date.now();
    return {
      end: () => {
        const duration = Date.now() - start;
        this.debug(`⏱️ ${label} took ${duration}ms`);
        return duration;
      },
    };
  }

  // System info logging
  system(message, meta = {}) {
    const systemInfo = {
      ...meta,
      memory: process.memoryUsage(),
      uptime: process.uptime(),
    };
    this.info(`🖥️ SYSTEM: ${message}`, systemInfo);
  }
}

// Create singleton instance
const logger = new Logger();

// Clean old logs on startup (optional)
if (process.env.AUTO_CLEAN_LOGS === "true") {
  logger.cleanOldLogs(30);
}

export default logger;