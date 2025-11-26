import { Zap, LogOut } from "lucide-react";
import { useEffect, useState } from "react";

const Navbar = ({ onNotificationClick }) => {
  const [username, setUsername] = useState("");

  useEffect(() => {
    const storedUsername = localStorage.getItem("username");
    if (storedUsername) {
      setUsername(storedUsername);
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("username");
    window.location.href = "/";
  };

  return (
    <nav className="bg-white border-b border-gray-200 px-6 py-4 flex justify-between items-center">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center">
          <Zap className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-lg font-bold text-gray-900">Solar Panel Monitor</h1>
          <p className="text-xs text-gray-500">Real-time Analytics Dashboard</p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {username && (
          <span className="text-gray-700 text-sm font-medium">
            Welcome, {username}
          </span>
        )}
        <button
          onClick={onNotificationClick}
          className="text-gray-600 hover:text-gray-900 text-sm font-medium"
        >
          Notifications
        </button>
        <button
          onClick={handleLogout}
          className="flex items-center gap-2 text-gray-600 hover:text-gray-900 text-sm font-medium"
        >
          Logout
          <LogOut className="w-4 h-4" />
        </button>
      </div>
    </nav>
  );
};

export default Navbar;
