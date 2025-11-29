import mqtt from "mqtt";
import mqttConfig from "../config/mqtt.js";
import { processSensorData } from "./mlService.js";
import { createNotification } from "./notificationService.js";

class MQTTService {
  constructor() {
    this.client = null;
    this.isConnected = false;
  }

  // Connect to MQTT broker
  connect() {
    console.log("🔌 Connecting to MQTT broker...");
    
    this.client = mqtt.connect(mqttConfig.brokerUrl, mqttConfig.options);

    this.client.on("connect", () => {
      console.log("✅ Connected to MQTT broker");
      this.isConnected = true;

      // Subscribe to sensor data topic
      this.client.subscribe(mqttConfig.topics.sensorData, (err) => {
        if (err) {
          console.error("❌ MQTT subscription error:", err);
        } else {
          console.log(`📡 Subscribed to: ${mqttConfig.topics.sensorData}`);
        }
      });
    });

    this.client.on("message", async (topic, message) => {
      try {
        console.log(`📨 Received message on ${topic}`);
        
        // Parse incoming sensor data
        const sensorData = JSON.parse(message.toString());
        console.log("📊 Sensor Data:", sensorData);

        // Process data (save to DB + ML prediction)
        await processSensorData(sensorData);
        
      } catch (error) {
        console.error("❌ Error processing MQTT message:", error);
      }
    });

    this.client.on("error", (error) => {
      console.error("❌ MQTT Error:", error);
      this.isConnected = false;
    });

    this.client.on("close", () => {
      console.log("🔌 MQTT connection closed");
      this.isConnected = false;
    });

    this.client.on("reconnect", () => {
      console.log("🔄 Reconnecting to MQTT broker...");
    });
  }

  // Publish command to hardware
  publishCommand(command) {
    if (!this.isConnected) {
      console.error("❌ MQTT not connected");
      return false;
    }

    const payload = JSON.stringify(command);
    this.client.publish(mqttConfig.topics.command, payload, { qos: 1 }, (err) => {
      if (err) {
        console.error("❌ Failed to publish command:", err);
      } else {
        console.log("✅ Command published:", command);
      }
    });

    return true;
  }

  // Disconnect
  disconnect() {
    if (this.client) {
      this.client.end();
      console.log("🔌 MQTT disconnected");
    }
  }
}

export default new MQTTService();