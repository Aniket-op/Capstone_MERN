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
    this.latestSolarData = null; // Store latest solar panel data from ESP8266
    this.latestSolarDataTimestamp = null; // Track when ESP8266 data was received
    this.pendingDataCollection = null; // Track data collection when command is sent
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

      // Subscribe to solar data topic (ESP8266 publishes here)
      this.client.subscribe(mqttConfig.topics.solarData, (err) => {
        if (err) {
          console.error("❌ Solar subscribe error:", err);
        } else {
          console.log(`☀️ Subscribed to '${mqttConfig.topics.solarData}'`);
        }
      });
    });

    this.client.on("message", async (topic, payload) => {
      const message = payload.toString();
      console.log(`📩 Message from [${topic}]: ${message}`);
      console.log(`   Topic length: ${topic.length}, Payload length: ${message.length}`);

      try {
        // Try to parse as JSON
        let data;
        try {
          data = JSON.parse(message);
          console.log(`   ✅ Parsed JSON successfully`);
        } catch (e) {
          console.warn("⚠️ Non-JSON message received:", message);
          console.warn(`   Parse error: ${e.message}`);
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
          
          // Check if we're waiting for combined data (command was sent)
          if (this.pendingDataCollection && !this.pendingDataCollection.esp32Data) {
            // Store ESP32 data (humidity, module_temp) for combination
            this.pendingDataCollection.esp32Data = {
              humidity: data.humidity,
              module_temp: data.module_temp || data.moduleTemperature,
              dailyYield: data.dailyYield,
              cleaningDays: data.cleaningDays,
            };
            console.log("📦 ESP32 data stored for combination:", this.pendingDataCollection.esp32Data);
            
            // Check if we have both ESP8266 and ESP32 data
            this.checkAndProcessCombinedData();
          } else if (!this.pendingDataCollection) {
            // Normal processing (no pending command - process ESP32 data as-is)
            try {
              await processSensorData(data);
              console.log("✅ Sensor data processed successfully");
            } catch (processError) {
              console.error("❌ Error in processSensorData:", processError);
            }
          }
        } 
        // Handle solar panel data topic (ESP8266 publishes here)
        else if (topic === mqttConfig.topics.solarData) {
          console.log("☀️ Processing solar panel data from ESP8266...");
          console.log(`   Received data: voltage=${data.voltage}, current=${data.current}, power=${data.power}`);
          
          // Store latest solar data
          this.latestSolarData = {
            voltage: data.voltage,
            current: data.current,
            power: data.power,
            timestamp: new Date().toLocaleString(),
          };
          this.latestSolarDataTimestamp = Date.now(); // Track when data was received
          
          console.log("✅ Solar data stored:", JSON.stringify(this.latestSolarData, null, 2));
          
          // Check if we're waiting for combined data (command was sent)
          if (this.pendingDataCollection && !this.pendingDataCollection.esp8266Data) {
            // Store ESP8266 data (voltage, current, power) for combination
            this.pendingDataCollection.esp8266Data = {
              voltage: data.voltage,
              current: data.current,
              power: data.power,
            };
            console.log("📦 ESP8266 data stored for combination:", this.pendingDataCollection.esp8266Data);
            
            // Check if we have both ESP8266 and ESP32 data
            this.checkAndProcessCombinedData();
          }
        } 
        else if (topic.includes("response")) {
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
    
    // Initialize pending data collection to wait for both ESP8266 and ESP32 responses
    this.pendingDataCollection = {
      command: command,
      esp8266Data: null, // Will store: voltage, current, power
      esp32Data: null,   // Will store: humidity, module_temp
      sentAt: Date.now(),
      timeout: null,
    };
    
    console.log("🔄 Waiting for combined data from ESP8266 and ESP32...");
    
    // If we have recent ESP8266 data (within last 30 seconds), use it immediately
    // This handles the case where ESP8266 publishes periodically but doesn't respond to commands
    if (this.latestSolarData && this.latestSolarDataTimestamp) {
      const dataAge = Date.now() - this.latestSolarDataTimestamp;
      if (dataAge < 30000) { // 30 seconds
        console.log(`📦 Using recent ESP8266 data (${Math.round(dataAge / 1000)}s old)`);
        this.pendingDataCollection.esp8266Data = {
          voltage: this.latestSolarData.voltage,
          current: this.latestSolarData.current,
          power: this.latestSolarData.power,
        };
      } else {
        console.log(`⚠️ ESP8266 data is too old (${Math.round(dataAge / 1000)}s), waiting for new data...`);
      }
    }
    
    // Set timeout to clear pending collection after 10 seconds
    this.pendingDataCollection.timeout = setTimeout(() => {
      if (this.pendingDataCollection) {
        console.log("⏰ Timeout: Did not receive both ESP8266 and ESP32 data within 10 seconds");
        console.log("   ESP8266 data:", this.pendingDataCollection.esp8266Data ? "received" : "missing");
        console.log("   ESP32 data:", this.pendingDataCollection.esp32Data ? "received" : "missing");
        
        // If we have ESP8266 data but not ESP32, we can still process with what we have
        // or just clear and wait for next attempt
        this.pendingDataCollection = null;
      }
    }, 10000);
    
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

  // Check if we have both ESP8266 and ESP32 data, then process combined data
  async checkAndProcessCombinedData() {
    if (!this.pendingDataCollection) {
      return;
    }

    const { esp8266Data, esp32Data } = this.pendingDataCollection;

    // Check if we have both datasets
    if (esp8266Data && esp32Data) {
      console.log("✅ Both ESP8266 and ESP32 data received!");
      console.log("📦 ESP8266 data:", esp8266Data);
      console.log("📦 ESP32 data:", esp32Data);

      // Clear timeout
      if (this.pendingDataCollection.timeout) {
        clearTimeout(this.pendingDataCollection.timeout);
      }

      // Combine data: ESP8266 (voltage, current, power) + ESP32 (humidity, module_temp)
      const combinedData = {
        voltage: esp8266Data.voltage,
        current: esp8266Data.current,
        power: esp8266Data.power,
        humidity: esp32Data.humidity,
        module_temp: esp32Data.module_temp,
        dailyYield: esp32Data.dailyYield || 0,
        cleaningDays: esp32Data.cleaningDays || 0,
      };

      console.log("🔄 Processing combined data:", JSON.stringify(combinedData, null, 2));

      // Process combined data (save to DB + ML prediction)
      try {
        await processSensorData(combinedData);
        console.log("✅ Combined data processed successfully");
      } catch (processError) {
        console.error("❌ Error in processSensorData:", processError);
      }

      // Clear pending collection
      this.pendingDataCollection = null;
    }
  }

  // Get latest message (for /latest endpoint)
  getLatestMessage() {
    return this.latestMessage;
  }

  // Get latest solar data (for /solar/latest endpoint)
  getLatestSolarData() {
    return this.latestSolarData;
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