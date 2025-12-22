/**
 * Mock Hardware Simulator
 * Simulates ESP32 and ESP8266 hardware devices that respond to MQTT commands
 * Run this alongside your backend server to test MQTT communication
 * 
 * Usage: node mockHardware.js
 */

import mqtt from "mqtt";
import dotenv from "dotenv";

dotenv.config();

// MQTT Configuration - Updated to match ESP32 and ESP8266 hardware setup
const host = process.env.MQTT_BROKER_HOST || "broker.emqx.io";
const port = process.env.MQTT_PORT || "1883";
const brokerUrl = `mqtt://${host}:${port}`;

const mqttConfig = {
  brokerUrl: brokerUrl,
  host: host,
  port: port,
  topics: {
    sensorData: process.env.MQTT_TOPIC_SENSOR || "esp32/sensor_data", // ESP32 publishes here
    command: process.env.MQTT_TOPIC_COMMAND || "esp32/command", // Server sends control commands
    response: process.env.MQTT_TOPIC_RESPONSE || "esp32/response",
    solarData: process.env.MQTT_TOPIC_SOLAR || "esp8266/solar_data", // ESP8266 publishes solar panel data here
  },
  options: {
    clientId: `mock_hardware_${Math.random().toString(16).slice(3)}`,
    clean: true,
    connectTimeout: 4000,
    reconnectPeriod: 1000,
  },
};

// State to track previous values for gradual variation (simulates real sensor behavior)
let previousValues = {
  moduleTemp: 40,
  current: 8,
  voltage: 13.5,
  humidity: 50,
  // ESP8266 solar panel values
  solarVoltage: 0.88,
  solarCurrent: -0.4,
  solarPower: 0,
};

// Generate realistic mock ESP32 sensor data (humidity, module_temp)
const generateESP32SensorData = (requestId = null) => {
  const now = new Date();
  const hour = now.getHours();
  
  // Simulate day/night cycle and gradual changes
  // Daytime (6 AM - 6 PM): Higher values, Nighttime: Lower values
  const isDaytime = hour >= 6 && hour < 18;
  
  // Add gradual variation (drift) to previous values (±10% change)
  const driftFactor = 0.9 + Math.random() * 0.2; // 0.9 to 1.1 (10% variation)
  
  // Module temperature: 30-60°C (higher during day, lower at night)
  previousValues.moduleTemp = Math.max(25, Math.min(65, 
    previousValues.moduleTemp * driftFactor + (isDaytime ? 5 : -5) + (Math.random() * 4 - 2)
  ));
  
  // Humidity: 25-85% (varies throughout day)
  previousValues.humidity = Math.max(20, Math.min(90, 
    previousValues.humidity * driftFactor + (Math.random() * 5 - 2.5)
  ));
  
  // Add some daily yield (cumulative, increases during day)
  const dailyYield = isDaytime ? (hour - 6) * 0.5 + Math.random() * 0.3 : 0;
  
  return {
    // Include requestId if provided (for response tracking)
    ...(requestId && { requestId, responseTo: requestId }),
    
    // ESP32 sensor readings (humidity, module_temp)
    module_temp: parseFloat(previousValues.moduleTemp.toFixed(1)),
    humidity: parseFloat(previousValues.humidity.toFixed(1)),
    dailyYield: parseFloat(dailyYield.toFixed(2)),
    cleaningDays: Math.floor(Math.random() * 10) + 1, // 1-10 days
    
    // Metadata
    timestamp: now.toISOString(),
    deviceId: "MOCK_ESP32_001",
    status: "operational"
  };
};

// Generate realistic mock ESP8266 solar panel data (voltage, current, power)
const generateESP8266SolarData = () => {
  const now = new Date();
  const hour = now.getHours();
  
  // Simulate day/night cycle
  const isDaytime = hour >= 6 && hour < 18;
  
  // Add gradual variation (drift) to previous values
  const driftFactor = 0.9 + Math.random() * 0.2; // 0.9 to 1.1 (10% variation)
  
  // Solar voltage: 0.5-1.5 V (varies with sunlight)
  previousValues.solarVoltage = Math.max(0.3, Math.min(1.8, 
    previousValues.solarVoltage * driftFactor + (isDaytime ? 0.2 : -0.3) + (Math.random() * 0.1 - 0.05)
  ));
  
  // Solar current: -0.5 to 2.0 A (can be negative at night)
  previousValues.solarCurrent = Math.max(-0.8, Math.min(2.5, 
    previousValues.solarCurrent * driftFactor + (isDaytime ? 0.3 : -0.5) + (Math.random() * 0.2 - 0.1)
  ));
  
  // Calculate power from voltage and current
  previousValues.solarPower = previousValues.solarVoltage * previousValues.solarCurrent;
  // Ensure power is not negative (or allow it if current is negative)
  if (previousValues.solarPower < 0) {
    previousValues.solarPower = 0;
  }
  
  return {
    voltage: parseFloat(previousValues.solarVoltage.toFixed(2)),
    current: parseFloat(previousValues.solarCurrent.toFixed(2)),
    power: parseFloat(previousValues.solarPower.toFixed(2)),
    timestamp: now.toISOString(),
    deviceId: "MOCK_ESP8266_001",
    status: "operational"
  };
};

// Connect to MQTT broker
console.log("🔌 Mock Hardware: Connecting to MQTT broker...");
console.log(`   Broker: ${mqttConfig.brokerUrl}`);
console.log(`   Client ID: ${mqttConfig.options.clientId}`);

const client = mqtt.connect(mqttConfig.brokerUrl, mqttConfig.options);

// Auto-publish interval (optional - simulates ESP32 sending data periodically)
let autoPublishInterval = null;
const AUTO_PUBLISH_INTERVAL = 30000; // 30 seconds (adjust as needed)

client.on("connect", () => {
  console.log("✅ Mock Hardware: Connected to MQTT broker");
  
  // Subscribe to command topic
  client.subscribe(mqttConfig.topics.command, (err) => {
    if (err) {
      console.error("❌ Mock Hardware: Subscription error:", err);
    } else {
      console.log(`📡 Mock Hardware: Subscribed to command topic: ${mqttConfig.topics.command}`);
      console.log(`\n🎭 Mock Hardware Simulator (ESP32 + ESP8266)`);
      console.log(`   Listening for commands...`);
      console.log(`   Supported commands:`);
      console.log(`   - String: "start", "stop", "spray"`);
      console.log(`   - JSON: { "action": "REQUEST_DATA" }`);
      console.log(`   - JSON: { "action": "START_CLEANING" }`);
      console.log(`   - JSON: { "action": "ALERT" }`);
      console.log(`   `);
      console.log(`   ESP32 publishes to: ${mqttConfig.topics.sensorData}`);
      console.log(`   ESP8266 publishes to: ${mqttConfig.topics.solarData}`);
      console.log(`   Auto-publishing ESP32 sensor data every ${AUTO_PUBLISH_INTERVAL / 1000} seconds\n`);
      
      // Start auto-publishing ESP32 sensor data (simulates ESP32 sending data periodically)
      autoPublishInterval = setInterval(() => {
        const sensorData = generateESP32SensorData();
        const payload = JSON.stringify(sensorData);
        client.publish(mqttConfig.topics.sensorData, payload, { qos: 1 }, (err) => {
          if (err) {
            console.error(`❌ Mock Hardware: Auto-publish failed:`, err);
          } else {
            console.log(`📊 Mock Hardware: Auto-published ESP32 sensor data (${new Date().toLocaleTimeString()})`);
          }
        });
      }, AUTO_PUBLISH_INTERVAL);
    }
  });
});

// Listen for commands
client.on("message", async (topic, payload) => {
  try {
    const message = payload.toString();
    console.log(`\n📨 Mock Hardware: Received command on topic: ${topic}`);
    console.log(`📦 Raw Command: ${message}`);
    
    // Handle string commands (like "start", "stop", "spray")
    if (typeof message === "string" && !message.startsWith("{")) {
      console.log(`\n🎮 Mock Hardware: Received string command: "${message}"`);
      
      if (message === "start") {
        console.log(`   ✅ Starting robot...`);
        await new Promise(resolve => setTimeout(resolve, 500));
        console.log(`   ✅ Robot started successfully\n`);
        
        // Optionally publish status
        const status = {
          command: "start",
          status: "started",
          timestamp: new Date().toISOString(),
        };
        client.publish(mqttConfig.topics.sensorData, JSON.stringify(status), { qos: 1 });
        
      } else if (message === "stop") {
        console.log(`   🛑 Stopping robot...`);
        await new Promise(resolve => setTimeout(resolve, 500));
        console.log(`   ✅ Robot stopped successfully\n`);
        
        const status = {
          command: "stop",
          status: "stopped",
          timestamp: new Date().toISOString(),
        };
        client.publish(mqttConfig.topics.sensorData, JSON.stringify(status), { qos: 1 });
        
      } else if (message === "spray") {
        console.log(`   💧 Activating spray system...`);
        await new Promise(resolve => setTimeout(resolve, 1000));
        console.log(`   ✅ Spray activated for 5 seconds\n`);
        
        const status = {
          command: "spray",
          status: "spraying",
          duration: 5,
          timestamp: new Date().toISOString(),
        };
        client.publish(mqttConfig.topics.sensorData, JSON.stringify(status), { qos: 1 });
        
      } else {
        console.log(`   ⚠️ Unknown string command: "${message}"\n`);
      }
      return;
    }
    
    // Handle JSON commands
    let command;
    try {
      command = JSON.parse(message);
      console.log(`✅ Parsed Command:`, JSON.stringify(command, null, 2));
    } catch (parseError) {
      console.log(`⚠️ Command is not valid JSON or string, ignoring...\n`);
      return;
    }
    
    // Check if it's a REQUEST_DATA command
    if (command.action === "REQUEST_DATA") {
      console.log(`\n🔄 Mock Hardware: Processing REQUEST_DATA command...`);
      console.log(`   Request ID: ${command.requestId || "N/A"}`);
      console.log(`   Source: ${command.source || "unknown"}`);
      
      // Simulate processing delay (like real hardware would have)
      const delayESP32 = 300 + Math.random() * 700; // 300-1000ms
      const delayESP8266 = 200 + Math.random() * 500; // 200-700ms (ESP8266 responds faster)
      
      console.log(`   Simulating hardware processing delays...`);
      console.log(`   ESP32 delay: ${delayESP32.toFixed(0)}ms`);
      console.log(`   ESP8266 delay: ${delayESP8266.toFixed(0)}ms`);
      
      // Generate ESP32 sensor data (humidity, module_temp)
      const esp32Data = generateESP32SensorData(command.requestId);
      console.log(`\n📊 Mock Hardware: Generated ESP32 sensor data:`);
      console.log(JSON.stringify(esp32Data, null, 2));
      
      // Generate ESP8266 solar panel data (voltage, current, power)
      const esp8266Data = generateESP8266SolarData();
      console.log(`\n☀️ Mock Hardware: Generated ESP8266 solar data:`);
      console.log(JSON.stringify(esp8266Data, null, 2));
      
      // Publish ESP32 data (humidity, module_temp) to esp32/sensor_data
      setTimeout(async () => {
        const payloadESP32 = JSON.stringify(esp32Data);
        client.publish(mqttConfig.topics.sensorData, payloadESP32, { qos: 1 }, (err) => {
          if (err) {
            console.error(`❌ Mock Hardware: Failed to publish ESP32 sensor data:`, err);
          } else {
            console.log(`\n✅ Mock Hardware: Published ESP32 sensor data to: ${mqttConfig.topics.sensorData}`);
            console.log(`   Data: humidity=${esp32Data.humidity}, module_temp=${esp32Data.module_temp}`);
            console.log(`   Response time: ${delayESP32.toFixed(0)}ms`);
          }
        });
      }, delayESP32);
      
      // Publish ESP8266 data (voltage, current, power) to esp8266/solar_data
      setTimeout(async () => {
        const payloadESP8266 = JSON.stringify(esp8266Data);
        client.publish(mqttConfig.topics.solarData, payloadESP8266, { qos: 1 }, (err) => {
          if (err) {
            console.error(`❌ Mock Hardware: Failed to publish ESP8266 solar data:`, err);
          } else {
            console.log(`\n✅ Mock Hardware: Published ESP8266 solar data to: ${mqttConfig.topics.solarData}`);
            console.log(`   Data: voltage=${esp8266Data.voltage}, current=${esp8266Data.current}, power=${esp8266Data.power}`);
            console.log(`   Response time: ${delayESP8266.toFixed(0)}ms\n`);
          }
        });
      }, delayESP8266);
      
    } else if (command.action === "START_CLEANING") {
      console.log(`\n🧹 Mock Hardware: Received START_CLEANING command`);
      console.log(`   Notification ID: ${command.notificationId || "N/A"}`);
      
      // Simulate cleaning process
      console.log(`   Starting cleaning process...`);
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Publish cleaning status
      const cleaningStatus = {
        requestId: command.requestId,
        action: "CLEANING_STATUS",
        status: "completed",
        notificationId: command.notificationId,
        timestamp: new Date().toISOString(),
        message: "Cleaning process completed successfully"
      };
      
      client.publish(mqttConfig.topics.sensorData, JSON.stringify(cleaningStatus), { qos: 1 }, (err) => {
        if (err) {
          console.error(`❌ Mock Hardware: Failed to publish cleaning status:`, err);
        } else {
          console.log(`✅ Mock Hardware: Cleaning completed and status published\n`);
        }
      });
      
    } else if (command.action === "ALERT") {
      console.log(`\n🚨 Mock Hardware: Received ALERT command`);
      console.log(`   Level: ${command.level || "UNKNOWN"}`);
      console.log(`   Message: ${command.message || "N/A"}`);
      console.log(`   Mock Hardware acknowledges alert\n`);
      
    } else {
      console.log(`\n⚠️ Mock Hardware: Unknown command action: ${command.action || "N/A"}\n`);
    }
    
  } catch (error) {
    console.error("❌ Mock Hardware: Error processing command:", error);
  }
});

client.on("error", (error) => {
  console.error("❌ Mock Hardware: MQTT Error:", error);
});

client.on("close", () => {
  console.log("🔌 Mock Hardware: MQTT connection closed");
});

client.on("reconnect", () => {
  console.log("🔄 Mock Hardware: Reconnecting to MQTT broker...");
});

// Handle graceful shutdown
process.on("SIGINT", () => {
  console.log("\n🛑 Mock Hardware: Shutting down gracefully...");
  
  // Clear auto-publish interval
  if (autoPublishInterval) {
    clearInterval(autoPublishInterval);
    console.log("   Stopped auto-publish interval");
  }
  
  client.end();
  process.exit(0);
});

console.log("\n🎭 =========================================");
console.log("   MOCK HARDWARE SIMULATOR");
console.log("   ESP32 + ESP8266");
console.log("   =========================================");
console.log("   This script simulates both ESP32 and");
console.log("   ESP8266 devices that respond to MQTT");
console.log("   commands.");
console.log("   ");
console.log("   Broker: broker.emqx.io");
console.log("   Topics:");
console.log("   - esp32/sensor_data (humidity, module_temp)");
console.log("   - esp8266/solar_data (voltage, current, power)");
console.log("   - esp32/command (receives commands)");
console.log("   ");
console.log("   When REQUEST_DATA command is received:");
console.log("   - ESP32 publishes: humidity, module_temp");
console.log("   - ESP8266 publishes: voltage, current, power");
console.log("   ");
console.log("   Keep this running alongside your backend");
console.log("   to test MQTT communication with both devices.");
console.log("   =========================================\n");

