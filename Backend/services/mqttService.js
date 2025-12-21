import mqtt from "mqtt";
import mqttConfig from "../config/mqtt.js";
import { processSensorData } from "./mlService.js";
import { createNotification } from "./notificationService.js";

class MQTTService {
  constructor() {
    this.client = null;
    this.isConnected = false;
    this.pendingRequests = new Map(); // Track sent requests
    this.latestMessage = null; // Store latest message for /latest endpoint
  }

  // Connect to MQTT broker
  connect() {
    console.log(`🔌 Connecting to MQTT broker: ${mqttConfig.brokerUrl}`);
    
    this.client = mqtt.connect(mqttConfig.brokerUrl, mqttConfig.options);

    this.client.on("connect", () => {
      console.log(`✅ Connected to MQTT broker: ${mqttConfig.brokerUrl}`);
      this.isConnected = true;

      // Subscribe to sensor data topic (ESP32 publishes here)
      this.client.subscribe(mqttConfig.topics.sensorData, (err) => {
        if (err) {
          console.error("❌ Subscribe error:", err);
        } else {
          console.log(`📡 Subscribed to '${mqttConfig.topics.sensorData}'`);
        }
      });

      // Subscribe to response topic (if hardware publishes responses)
      const responseTopic = mqttConfig.topics.response || "esp32/response";
      this.client.subscribe(responseTopic, (err) => {
        if (err) {
          console.error("❌ MQTT response subscription error:", err);
        } else {
          console.log(`📡 Subscribed to response topic: ${responseTopic}`);
        }
      });
    });

    this.client.on("message", async (topic, payload) => {
      const message = payload.toString();
      console.log(`📩 Message from [${topic}]: ${message}`);

      try {
        // Try to parse as JSON
        let data;
        try {
          data = JSON.parse(message);
        } catch (e) {
          console.warn("⚠️ Non-JSON message received:", message);
          // Store raw message
          this.latestMessage = {
            topic,
            data: { raw: message },
            timestamp: new Date().toLocaleString(),
          };
          return;
        }

        // Store latest message for /latest endpoint
        this.latestMessage = {
          topic,
          data,
          timestamp: new Date().toLocaleString(),
        };

        // Check if this is a response to a request
        if (data.requestId || data.responseTo) {
          const requestId = data.requestId || data.responseTo;
          const pendingRequest = this.pendingRequests.get(requestId);
          
          if (pendingRequest) {
            const responseTime = Date.now() - pendingRequest.sentAt;
            console.log(`🔄 Response to request ${requestId} (${responseTime}ms)`);
            this.pendingRequests.delete(requestId);
          }
        }

        // Handle sensor data topic (ESP32 publishes here)
        if (topic === mqttConfig.topics.sensorData) {
          console.log("📊 Processing sensor data from ESP32...");
          
          // Process data (save to DB + ML prediction)
          try {
            await processSensorData(data);
            console.log("✅ Sensor data processed successfully");
          } catch (processError) {
            console.error("❌ Error in processSensorData:", processError);
          }
        } else if (topic.includes("response")) {
          console.log("💬 Response message received");
        }
        
      } catch (error) {
        console.error("❌ Error processing MQTT message:", error);
        console.error("   Topic:", topic);
        console.error("   Message:", message);
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

  // Publish command to hardware (supports both string and object commands)
  publishCommand(command) {
    if (!this.isConnected) {
      console.error("❌ MQTT not connected");
      return false;
    }

    const topic = mqttConfig.topics.command;
    let payload;
    
    // If command is a string (like "start", "stop", "spray"), send as-is
    if (typeof command === "string") {
      payload = command;
      console.log(`📤 Command sent to ESP32: ${command}`);
    } else {
      // If command is an object, add request ID and timestamp
      const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const commandWithId = {
        ...command,
        requestId: requestId,
        sentAt: new Date().toISOString()
      };

      // Store pending request
      this.pendingRequests.set(requestId, {
        command: command,
        sentAt: Date.now(),
        timestamp: new Date().toISOString()
      });

      payload = JSON.stringify(commandWithId);
      console.log(`📤 Command sent to ESP32:`, JSON.stringify(commandWithId, null, 2));
      
      // Set timeout to remove pending request after 30 seconds if no response
      setTimeout(() => {
        if (this.pendingRequests.has(requestId)) {
          console.log(`⏰ Timeout: No response received for Request ID: ${requestId}`);
          this.pendingRequests.delete(requestId);
        }
      }, 30000);
    }
    
    this.client.publish(topic, payload, { qos: 1 }, (err) => {
      if (err) {
        console.error("❌ Publish error:", err);
        return false;
      }
    });

    return true;
  }

  // Get latest message (for /latest endpoint)
  getLatestMessage() {
    return this.latestMessage;
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