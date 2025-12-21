import mqtt from "mqtt";
import dotenv from "dotenv";

dotenv.config();

// MQTT Configuration - Updated to match ESP32 hardware
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
  },
  options: {
    clientId: `server_${Math.random().toString(16).slice(3)}`,
    clean: true,
    connectTimeout: 4000,
    reconnectPeriod: 1000,
  },
};

export default mqttConfig;