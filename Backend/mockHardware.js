/**
 * Mock Hardware Simulator
 * Simulates a hardware device that responds to MQTT commands
 * Run this alongside your backend server to test MQTT communication
 * 
 * Usage: node mockHardware.js
 */

import mqtt from "mqtt";
import dotenv from "dotenv";

dotenv.config();

const mqttConfig = {
  brokerUrl: process.env.MQTT_BROKER_URL || "mqtt://broker.hivemq.com",
  port: process.env.MQTT_PORT || 1883,
  topics: {
    sensorData: process.env.MQTT_TOPIC_SENSOR || "solar/sensor/data",
    command: process.env.MQTT_TOPIC_COMMAND || "solar/command",
    response: process.env.MQTT_TOPIC_RESPONSE || "solar/response",
  },
  options: {
    clientId: `mock_hardware_${Math.random().toString(16).slice(3)}`,
    clean: true,
    connectTimeout: 4000,
    reconnectPeriod: 1000,
  },
};

// Generate realistic mock sensor data (Hardware format: module_temp, current, voltage, humidity, power)
const generateMockSensorData = (requestId = null) => {
  const now = new Date();
  
  // Simulate realistic hardware sensor readings
  const moduleTemp = 35 + Math.random() * 20; // 35-55°C (module temperature)
  const current = 5 + Math.random() * 10; // 5-15 A
  const voltage = 12 + Math.random() * 3; // 12-15 V
  const humidity = 30 + Math.random() * 50; // 30-80%
  const power = current * voltage; // Power in watts (current * voltage)
  
  return {
    // Include requestId if provided (for response tracking)
    ...(requestId && { requestId, responseTo: requestId }),
    
    // Hardware sensor readings (exact format from hardware)
    module_temp: parseFloat(moduleTemp.toFixed(1)),
    current: parseFloat(current.toFixed(2)),
    voltage: parseFloat(voltage.toFixed(2)),
    humidity: parseFloat(humidity.toFixed(1)),
    power: parseFloat(power.toFixed(2)),
    
    // Metadata
    timestamp: now.toISOString(),
    deviceId: "MOCK_HARDWARE_001",
    status: "operational"
  };
};

// Connect to MQTT broker
console.log("🔌 Mock Hardware: Connecting to MQTT broker...");
console.log(`   Broker: ${mqttConfig.brokerUrl}`);
console.log(`   Client ID: ${mqttConfig.options.clientId}`);

const client = mqtt.connect(mqttConfig.brokerUrl, mqttConfig.options);

client.on("connect", () => {
  console.log("✅ Mock Hardware: Connected to MQTT broker");
  
  // Subscribe to command topic
  client.subscribe(mqttConfig.topics.command, (err) => {
    if (err) {
      console.error("❌ Mock Hardware: Subscription error:", err);
    } else {
      console.log(`📡 Mock Hardware: Subscribed to command topic: ${mqttConfig.topics.command}`);
      console.log(`\n🎭 Mock Hardware is now listening for commands...`);
      console.log(`   Waiting for REQUEST_DATA commands...\n`);
    }
  });
});

// Listen for commands
client.on("message", async (topic, message) => {
  try {
    const messageStr = message.toString();
    console.log(`\n📨 Mock Hardware: Received command on topic: ${topic}`);
    console.log(`📦 Raw Command: ${messageStr}`);
    
    let command;
    try {
      command = JSON.parse(messageStr);
      console.log(`✅ Parsed Command:`, JSON.stringify(command, null, 2));
    } catch (parseError) {
      console.log(`⚠️ Command is not JSON, ignoring...`);
      return;
    }
    
    // Check if it's a REQUEST_DATA command
    if (command.action === "REQUEST_DATA") {
      console.log(`\n🔄 Mock Hardware: Processing REQUEST_DATA command...`);
      console.log(`   Request ID: ${command.requestId || "N/A"}`);
      console.log(`   Source: ${command.source || "unknown"}`);
      
      // Simulate processing delay (like real hardware would have)
      const delay = 500 + Math.random() * 1000; // 500-1500ms
      console.log(`   Simulating hardware processing delay: ${delay.toFixed(0)}ms...`);
      
      await new Promise(resolve => setTimeout(resolve, delay));
      
      // Generate mock sensor data
      const sensorData = generateMockSensorData(command.requestId);
      
      console.log(`\n📊 Mock Hardware: Generated sensor data:`);
      console.log(JSON.stringify(sensorData, null, 2));
      
      // Publish sensor data to sensor data topic
      const payload = JSON.stringify(sensorData);
      client.publish(mqttConfig.topics.sensorData, payload, { qos: 1 }, (err) => {
        if (err) {
          console.error(`❌ Mock Hardware: Failed to publish sensor data:`, err);
        } else {
          console.log(`\n✅ Mock Hardware: Published sensor data to: ${mqttConfig.topics.sensorData}`);
          console.log(`   Request ID included: ${command.requestId ? "Yes" : "No"}`);
          console.log(`   Response time: ${delay.toFixed(0)}ms\n`);
        }
      });
      
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
  client.end();
  process.exit(0);
});

console.log("\n🎭 =========================================");
console.log("   MOCK HARDWARE SIMULATOR");
console.log("   =========================================");
console.log("   This script simulates a hardware device");
console.log("   that responds to MQTT commands.");
console.log("   ");
console.log("   Keep this running alongside your backend");
console.log("   to test MQTT request/response flow.");
console.log("   =========================================\n");

