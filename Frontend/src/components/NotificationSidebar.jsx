import { useEffect, useState } from "react";
import axios from "axios";
import { XCircle } from "lucide-react";

const NotificationSidebar = ({ isOpen, onClose }) => {
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    if (!isOpen) return;
    const fetchNotifications = async () => {
      try {
        const { data } = await axios.get("http://localhost:5000/api/notifications", {
          headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
        });
        setNotifications(data);
      } catch (err) {
        console.error("Error fetching notifications:", err);
      }
    };
    fetchNotifications();
  }, [isOpen]);

  const handleResponse = async (id, response) => {
    try {
      await axios.post(
        `http://localhost:5000/api/notifications/respond/${id}`,
        { response },
        {
          headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
        }
      );
      setNotifications((prev) =>
        prev.map((n) =>
          n._id === id ? { ...n, userResponse: response } : n
        )
      );
    } catch (err) {
      console.error("Error updating response:", err);
    }
  };

  return (
    <div
      className={`fixed top-0 right-0 h-full w-96 bg-gradient-to-b from-gray-50 to-white shadow-2xl border-l border-gray-200 transform transition-transform duration-300 ease-in-out z-50 ${
        isOpen ? "translate-x-0" : "translate-x-full"
      }`}
    >
      {/* Header */}
      <div className="sticky top-0 bg-white border-b p-4 flex justify-between items-center shadow-sm z-10">
        <h2 className="text-xl font-semibold text-gray-800">Notifications</h2>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-red-600 transition-colors"
        >
          <XCircle size={22} />
        </button>
      </div>

      {/* Notification List */}
      <div className="p-5 overflow-y-auto h-[calc(100%-64px)] scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-100">
        {notifications.length === 0 ? (
          <div className="flex flex-col items-center justify-center mt-20 text-gray-500">
            <img
              src="https://illustrations.popsy.co/gray/notification.svg"
              alt="No notifications"
              className="w-40 mb-4 opacity-80"
            />
            <p className="text-center">You’re all caught up! 🎉</p>
          </div>
        ) : (
          notifications.map((n) => (
            <div
              key={n._id}
              className="bg-white border border-gray-200 shadow-sm rounded-xl p-4 mb-4 hover:shadow-md transition-shadow"
            >
              <div className="flex justify-between items-start">
                <p className="text-gray-800 font-medium">{n.message}</p>
                {n.userResponse && (
                  <span
                    className={`text-xs font-semibold px-2 py-1 rounded-full ${
                      n.userResponse === "yes"
                        ? "bg-green-100 text-green-700"
                        : "bg-red-100 text-red-700"
                    }`}
                  >
                    {n.userResponse.toUpperCase()}
                  </span>
                )}
              </div>

              <div className="flex justify-end gap-3 mt-3">
                <button
                  onClick={() => handleResponse(n._id, "yes")}
                  disabled={!!n.userResponse}
                  className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
                    n.userResponse === "yes"
                      ? "bg-green-500 text-white shadow-md"
                      : "bg-green-100 hover:bg-green-200 text-green-700"
                  } ${n.userResponse ? "opacity-60 cursor-not-allowed" : ""}`}
                >
                  Yes
                </button>
                <button
                  onClick={() => handleResponse(n._id, "no")}
                  disabled={!!n.userResponse}
                  className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
                    n.userResponse === "no"
                      ? "bg-red-500 text-white shadow-md"
                      : "bg-red-100 hover:bg-red-200 text-red-700"
                  } ${n.userResponse ? "opacity-60 cursor-not-allowed" : ""}`}
                >
                  No
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default NotificationSidebar;
