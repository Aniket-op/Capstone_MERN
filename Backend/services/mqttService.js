import mqtt from "mqtt";
import mqttConfig from "../config/mqtt.js";
import { processSensorData } from "./mlService.js";
import { createNotification } from "./notificationService.js";

class MQTTService {
  constructor() {
    this.client = null;
    this.isConnected = false;
    this.pendingRequests = new Map(); // Track sent requests
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

      // Subscribe to response topic (if hardware publishes responses)
      const responseTopic = mqttConfig.topics.response || "solar/response";
      this.client.subscribe(responseTopic, (err) => {
        if (err) {
          console.error("❌ MQTT response subscription error:", err);
        } else {
          console.log(`📡 Subscribed to response topic: ${responseTopic}`);
        }
      });
    });

    this.client.on("message", async (topic, message) => {
      try {
        const messageStr = message.toString();
        const timestamp = new Date().toISOString();
        
        console.log(`\n📨 ===== MQTT MESSAGE RECEIVED =====`);
        console.log(`📅 Timestamp: ${timestamp}`);
        console.log(`📌 Topic: ${topic}`);
        console.log(`📦 Raw Message: ${messageStr}`);
        
        // Try to parse as JSON
        let parsedData;
        try {
          parsedData = JSON.parse(messageStr);
          console.log(`✅ Parsed JSON:`, JSON.stringify(parsedData, null, 2));
        } catch (parseError) {
          console.log(`⚠️ Message is not JSON, raw text: ${messageStr}`);
          parsedData = { raw: messageStr };
        }

        // Check if this is a response to a request
        if (parsedData.requestId || parsedData.responseTo) {
          const requestId = parsedData.requestId || parsedData.responseTo;
          const pendingRequest = this.pendingRequests.get(requestId);
          
          if (pendingRequest) {
            const responseTime = Date.now() - pendingRequest.sentAt;
            console.log(`\n🔄 RESPONSE TO REQUEST:`);
            console.log(`   Request ID: ${requestId}`);
            console.log(`   Original Command: ${JSON.stringify(pendingRequest.command)}`);
            console.log(`   Response Time: ${responseTime}ms`);
            console.log(`   Response Data:`, parsedData);
            
            // Remove from pending
            this.pendingRequests.delete(requestId);
          }
        }

        // Handle sensor data topic
        if (topic === mqttConfig.topics.sensorData) {
          console.log(`\n📊 Processing as Sensor Data...`);
          const sensorData = parsedData;
          console.log("📊 Sensor Data received:", JSON.stringify(sensorData, null, 2));

          // Process data (save to DB + ML prediction)
          try {
            await processSensorData(sensorData);
            console.log("✅ Sensor data processed successfully");
          } catch (processError) {
            console.error("❌ Error in processSensorData:", processError);
          }
        } else if (topic.includes("response")) {
          console.log(`\n💬 This is a response message`);
          // You can add specific response handling here
        }
        
        console.log(`📨 ===== END MESSAGE =====\n`);
        
      } catch (error) {
        console.error("❌ Error processing MQTT message:", error);
        console.error("   Topic:", topic);
        console.error("   Message:", message.toString());
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

    // Add request ID and timestamp to track responses
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

    const payload = JSON.stringify(commandWithId);
    const topic = mqttConfig.topics.command;
    
    console.log(`\n📤 ===== PUBLISHING MQTT COMMAND =====`);
    console.log(`📌 Topic: ${topic}`);
    console.log(`🆔 Request ID: ${requestId}`);
    console.log(`📦 Payload:`, JSON.stringify(commandWithId, null, 2));
    
    this.client.publish(topic, payload, { qos: 1 }, (err) => {
      if (err) {
        console.error(`❌ Failed to publish command:`, err);
        this.pendingRequests.delete(requestId);
      } else {
        console.log(`✅ Command published successfully`);
        console.log(`⏳ Waiting for response (Request ID: ${requestId})...`);
        console.log(`📤 ===== END PUBLISH =====\n`);
        
        // Set timeout to remove pending request after 30 seconds if no response
        setTimeout(() => {
          if (this.pendingRequests.has(requestId)) {
            console.log(`⏰ Timeout: No response received for Request ID: ${requestId}`);
            this.pendingRequests.delete(requestId);
          }
        }, 30000);
      }
    });

    return requestId;
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