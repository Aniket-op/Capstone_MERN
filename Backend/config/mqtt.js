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
    clientId: `solar_backend_${Math.random().toString(16).slice(3)}`,
    clean: true,
    connectTimeout: 4000,
    reconnectPeriod: 1000,
  },
};

export default mqttConfig;