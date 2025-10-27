import { useEffect, useState } from "react";
import axios from "axios";
import Navbar from "../components/Navbar";
import ChartCard from "../components/ChartCard";
import NotificationSidebar from "../components/NotificationSidebar";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  AreaChart,
  Area,
  BarChart,
  Bar,
} from "recharts";

const Dashboard = () => {
  const [solarData, setSolarData] = useState([]);
  const [latest, setLatest] = useState(null);
  const [showNotifications, setShowNotifications] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const { data } = await axios.get(
          "http://localhost:5000/api/demo/solar",
          {
            headers: {
              Authorization: `Bearer ${localStorage.getItem("token")}`,
            },
          }
        );
        setSolarData(data);
        if (data.length > 0) setLatest(data[data.length - 1]);
      } catch (err) {
        console.error("Error fetching solar data:", err);
      }
    };
    fetchData();
  }, []);

  console.log(solarData);
  return (
    <div className="bg-gray-100 min-h-screen">
      <Navbar onNotificationClick={() => setShowNotifications(true)} />
      <NotificationSidebar
        isOpen={showNotifications}
        onClose={() => setShowNotifications(false)}
      />
      <div className="p-6">
        {/* -------- TOP KPI CARDS -------- */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white shadow rounded-xl p-4 flex flex-col items-center">
            <p className="text-gray-500">Current Power</p>
            <h2 className="text-2xl text-gray-500 font-bold">
              {latest ? `${latest.powerGeneration} kW` : "0.0 kW"}
            </h2>
            <span className="text-green-500 text-sm">+12.5%</span>
          </div>
          <div className="bg-white shadow rounded-xl p-4 flex flex-col items-center">
            <p className="text-gray-500">Solar Irradiance</p>
            <h2 className="text-2xl text-gray-500 font-bold">
              {latest ? `${latest.solarIrradiance} W/m²` : "0 W/m²"}
            </h2>
            <span className="text-green-500 text-sm">Optimal</span>
          </div>
          <div className="bg-white shadow rounded-xl p-4 flex flex-col items-center">
            <p className="text-gray-500">Temperature</p>
            <h2 className="text-2xl text-gray-500 font-bold">
              {latest ? `${latest.temperature}°C` : "--"}
            </h2>
            <span className="text-orange-500 text-sm">High</span>
          </div>
          <div className="bg-white shadow rounded-xl p-4 flex flex-col items-center">
            <p className="text-gray-500">Humidity</p>
            <h2 className="text-2xl text-gray-500  font-bold">
              {latest ? `${latest.humidity}%` : "--"}
            </h2>
            <span className="text-blue-500 text-sm">Normal</span>
          </div>
        </div>

        {/* -------- MIDDLE CHARTS -------- */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <ChartCard title="Temperature vs Humidity">
            <LineChart
              width={500}
              height={250}
              data={solarData.map((d, i) => ({ ...d, index: i + 1 }))}
            >
              <CartesianGrid stroke="#eee" />
              <XAxis dataKey="index" />
              <YAxis />

              <Line
                type="monotone"
                dataKey="temperature"
                stroke="#f97316"
                name="Temperature (°C)"
              />
              <Line
                type="monotone"
                dataKey="humidity"
                stroke="#2563eb"
                name="Humidity (%)"
              />
              <Tooltip />
            </LineChart>
          </ChartCard>

          <ChartCard title="Current Output">
            <AreaChart
              width={500}
              height={250}
              data={solarData.map((d, i) => ({ ...d, index: i + 1 }))}
            >
              <CartesianGrid stroke="#eee" />
              <XAxis dataKey="index" />
              <YAxis />
              <Tooltip />
              <Area
                type="monotone"
                dataKey="current"
                stroke="#6366f1"
                fill="#c7d2fe"
              />
            </AreaChart>
          </ChartCard>
        </div>

        {/* -------- LOWER CHARTS -------- */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
          <ChartCard title="Solar Irradiance">
            <BarChart
              width={500}
              height={250}
              data={solarData.map((d, i) => ({ ...d, index: i + 1 }))}
            >
              <CartesianGrid stroke="#eee" />
              <XAxis dataKey="index" />
              <YAxis />
              <Tooltip />
              <Bar
                dataKey="solarIrradiance"
                fill="#f59e0b"
                name="Irradiance (W/m²)"
              />
            </BarChart>
          </ChartCard>

          <ChartCard title="Power Generation Prediction">
            <LineChart
              width={500}
              height={250}
              data={solarData.map((d, i) => ({ ...d, index: i + 1 }))}
            >
              <CartesianGrid stroke="#eee" />
              <XAxis dataKey="index" />
              <YAxis />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="powerGeneration"
                stroke="#10b981"
                name="Actual (kW)"
              />
              <Line
                type="monotone"
                // dataKey="powerPredicted"
                stroke="#6b7280"
                strokeDasharray="5 5"
                name="Predicted (kW)"
              />
            </LineChart>
          </ChartCard>
        </div>

        {/* -------- SYSTEM INSIGHTS -------- */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <div className="bg-white shadow rounded-xl p-4 text-center">
            <p className="text-gray-500">Panel Efficiency</p>
            <h2 className="text-2xl font-bold text-green-600">
              {latest ? `${latest.panelEfficiency}%` : "94.2%"}
            </h2>
          </div>
          <div className="bg-white shadow rounded-xl p-4 text-center">
            <p className="text-gray-500">Today's Yield</p>
            <h2 className="text-2xl font-bold text-blue-600">
              {latest ? `${latest.dailyYield} kWh` : "22.1 kWh"}
            </h2>
          </div>
          <div className="bg-white shadow rounded-xl p-4 text-center">
            <p className="text-gray-500">Cleaning Status</p>
            <h2 className="text-2xl font-bold text-orange-600">
              Last: {latest ? `${latest.cleaningDays} days` : "2 days"}
            </h2>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
