import { useEffect, useState } from "react";
import axios from "axios";
import Navbar from "../components/Navbar";
import ChartCard from "../components/ChartCard";
import NotificationSidebar from "../components/NotificationSidebar";
import { Zap, Sun, Thermometer, Droplet } from "lucide-react";
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
  Legend,
  ResponsiveContainer,
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
        {/* Main Title and Subtitle */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Solar Panel Analytics</h1>
          <p className="text-gray-600">Monitor your solar panel performance in real-time</p>
        </div>

        {/* -------- TOP KPI CARDS -------- */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white shadow rounded-xl p-6">
            <div className="flex items-center justify-between mb-3">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                <Zap className="w-6 h-6 text-blue-600" />
              </div>
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-1">
              {latest ? `${latest.powerGeneration} kW` : "0.0 kW"}
            </h2>
            <p className="text-sm text-gray-500 mb-2">Current Power</p>
            <span className="text-green-500 text-sm font-medium">+12.5%</span>
          </div>
          <div className="bg-white shadow rounded-xl p-6">
            <div className="flex items-center justify-between mb-3">
              <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center">
                <Sun className="w-6 h-6 text-orange-600" />
              </div>
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-1">
              {latest ? `${latest.solarIrradiance} W/m²` : "0 W/m²"}
            </h2>
            <p className="text-sm text-gray-500 mb-2">Solar Irradiance</p>
            <span className="text-green-500 text-sm font-medium">Optimal</span>
          </div>
          <div className="bg-white shadow rounded-xl p-6">
            <div className="flex items-center justify-between mb-3">
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                <Thermometer className="w-6 h-6 text-green-600" />
              </div>
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-1">
              {latest ? `${latest.temperature}°C` : "35°C"}
            </h2>
            <p className="text-sm text-gray-500 mb-2">Temperature</p>
            <span className="text-orange-500 text-sm font-medium">High</span>
          </div>
          <div className="bg-white shadow rounded-xl p-6">
            <div className="flex items-center justify-between mb-3">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                <Droplet className="w-6 h-6 text-blue-600" />
              </div>
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-1">
              {latest ? `${latest.humidity}%` : "42%"}
            </h2>
            <p className="text-sm text-gray-500 mb-2">Humidity</p>
            <span className="text-green-500 text-sm font-medium">Normal</span>
          </div>
        </div>

        {/* -------- MIDDLE CHARTS -------- */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <ChartCard title="Temperature vs Humidity" subtitle="24-hour monitoring">
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={solarData.map((d, i) => ({ ...d, index: i + 1 }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="index" stroke="#6b7280" />
                <YAxis stroke="#6b7280" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px'
                  }} 
                />
                <Legend 
                  wrapperStyle={{ paddingTop: '20px' }}
                  iconType="circle"
                />
                <Line
                  type="monotone"
                  dataKey="temperature"
                  stroke="#f97316"
                  strokeWidth={2}
                  name="Temperature (°C)"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="humidity"
                  stroke="#2563eb"
                  strokeWidth={2}
                  name="Humidity (%)"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Current Output" subtitle="Real-time current measurement">
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={solarData.map((d, i) => ({ ...d, index: i + 1 }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="index" stroke="#6b7280" />
                <YAxis stroke="#6b7280" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px'
                  }} 
                />
                <Legend 
                  wrapperStyle={{ paddingTop: '20px' }}
                  iconType="circle"
                />
                <Area
                  type="monotone"
                  dataKey="current"
                  stroke="#2563eb"
                  fill="#2563eb"
                  fillOpacity={0.3}
                  name="Current (A)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* -------- LOWER CHARTS -------- */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
          <ChartCard title="Solar Irradiance" subtitle="Daily irradiance levels">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={solarData.map((d, i) => ({ ...d, index: i + 1 }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="index" stroke="#6b7280" />
                <YAxis stroke="#6b7280" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px'
                  }} 
                />
                <Legend 
                  wrapperStyle={{ paddingTop: '20px' }}
                  iconType="circle"
                />
                <Bar
                  dataKey="solarIrradiance"
                  fill="#f59e0b"
                  name="Irradiance (W/m²)"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Power Generation Prediction" subtitle="AI-powered forecasting">
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={solarData.map((d, i) => ({ 
                ...d, 
                index: i + 1,
                predicted: d.powerPredicted || (d.powerGeneration * 0.98)
              }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="index" stroke="#6b7280" />
                <YAxis stroke="#6b7280" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px'
                  }} 
                />
                <Legend 
                  wrapperStyle={{ paddingTop: '20px' }}
                  iconType="line"
                />
                <Line
                  type="monotone"
                  dataKey="powerGeneration"
                  stroke="#10b981"
                  strokeWidth={2}
                  name="Actual (kW)"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke="#6b7280"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  name="Predicted (kW)"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* -------- SYSTEM INSIGHTS -------- */}
        <div className="mt-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">System Insights</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white shadow rounded-xl p-6 text-center">
              <h2 className="text-3xl font-bold text-green-600 mb-2">
                {latest ? `${latest.panelEfficiency}%` : "94.2%"}
              </h2>
              <p className="text-gray-500 text-sm">Panel Efficiency</p>
            </div>
            <div className="bg-white shadow rounded-xl p-6 text-center">
              <h2 className="text-3xl font-bold text-blue-600 mb-2">
                {latest ? `${latest.dailyYield} kWh` : "23.6 kWh"}
              </h2>
              <p className="text-gray-500 text-sm">Today's Yield</p>
            </div>
            <div className="bg-white shadow rounded-xl p-6 text-center">
              <h2 className="text-3xl font-bold text-orange-600 mb-2">
                Last: {latest ? `${latest.cleaningDays} days` : "2 days"}
              </h2>
              <p className="text-gray-500 text-sm">Cleaning Status</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
