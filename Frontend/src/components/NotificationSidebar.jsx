import { useEffect, useState } from "react";
import axios from "axios";
import { io } from "socket.io-client";
import { XCircle, ChevronDown, ChevronUp } from "lucide-react";

const NotificationSidebar = ({ isOpen, onClose }) => {
  const [notifications, setNotifications] = useState([]);
  const [expandedNotifications, setExpandedNotifications] = useState(new Set());

  // Fetch notifications (limit to 7 most recent)
  useEffect(() => {
    if (!isOpen) return;
    const fetchNotifications = async () => {
      try {
        const { data } = await axios.get(
          "http://localhost:5000/api/notifications?limit=7",
          {
            headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
          }
        );
        const list = Array.isArray(data) ? data : data?.data || [];
        // Ensure we only show 7 most recent
        const recentNotifications = list.slice(0, 7);
        setNotifications(recentNotifications);
      } catch (err) {
        console.error("Error fetching notifications:", err);
        setNotifications([]);
      }
    };
    fetchNotifications();
  }, [isOpen]);

  // Listen for new notifications via Socket.IO
  useEffect(() => {
    if (!isOpen) return;
    
    const socket = io("http://localhost:5000", {
      transports: ["websocket", "polling"],
    });

    socket.on("newNotification", (newNotification) => {
      setNotifications((prev) => {
        // Add new notification and keep only 7 most recent
        const updated = [newNotification, ...prev].slice(0, 7);
        return updated;
      });
    });

    socket.on("notificationUpdated", (updatedNotification) => {
      setNotifications((prev) =>
        prev.map((n) =>
          n._id === updatedNotification._id ? updatedNotification : n
        )
      );
    });

    return () => {
      socket.disconnect();
    };
  }, [isOpen]);

  // Format ML message for display (same as Toast)
  const formatMLMessage = (notification) => {
    let mlData = null;
    try {
      if (notification.fullMLMessage) {
        mlData = JSON.parse(notification.fullMLMessage);
      }
    } catch (e) {
      console.error("Error parsing ML message:", e);
    }

    if (!mlData) {
      return notification.message;
    }

    // Format exactly like Toast notification
    return `Status: ${(mlData.status || notification.mlStatus || "UNKNOWN").toUpperCase()}\n` +
      `Message: ${mlData.message || notification.message}\n` +
      `Recommendation: ${mlData.recommendation || ""}\n` +
      `Confidence: ${mlData.confidence ? (mlData.confidence * 100).toFixed(1) + "%" : "N/A"}\n` +
      `Power Loss: ${(mlData.power_loss_percentage || notification.powerLoss || 0).toFixed(2)}%\n` +
      `Estimated Energy Loss: ${(mlData.estimated_energy_loss_kwh || 0).toFixed(2)} kWh\n` +
      `Needs Cleaning: ${mlData.needs_cleaning ? "Yes" : "No"}`;
  };

  // Handle YES - Send cleaning command
  const handleYes = async (id) => {
    try {
      await axios.put(
        `http://localhost:5000/api/notifications/${id}/respond`,
        { response: "yes" },
        {
          headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
        }
      );
      
      // Update local state
      setNotifications((prev) =>
        prev.map((n) =>
          n._id === id ? { ...n, userResponse: "yes", status: "responded" } : n
        )
      );
      
      console.log("✅ Cleaning command sent to hardware");
    } catch (err) {
      console.error("Error sending cleaning command:", err);
      alert("Failed to send cleaning command. Please try again.");
    }
  };

  // Handle NO - Delete notification
  const handleNo = async (id) => {
    try {
      await axios.delete(
        `http://localhost:5000/api/notifications/${id}`,
        {
          headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
        }
      );
      
      // Remove from local state
      setNotifications((prev) => prev.filter((n) => n._id !== id));
      
      console.log("✅ Notification deleted");
    } catch (err) {
      console.error("Error deleting notification:", err);
      alert("Failed to delete notification. Please try again.");
    }
  };

  return (
    <div
      className={`fixed top-0 right-0 h-full w-96 bg-white from-gray-50 to-white shadow-2xl border-l border-gray-200 transform transition-transform duration-300 ease-in-out z-50 ${
        isOpen ? "translate-x-0" : "translate-x-full"
      }`}
    >
      {/* Header */}
      <div className="sticky top-0 bg-white border-b p-4 flex justify-between items-center shadow-sm z-10">
        <h2 className="text-xl font-semibold text-gray-800">
          Notifications ({notifications.length})
        </h2>
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
            <p className="text-center">You' re all caught up! 🎉</p>
          </div>
        ) : (
          notifications.map((n) => {
            const isExpanded = expandedNotifications.has(n._id);
            const toggleExpand = () => {
              const newExpanded = new Set(expandedNotifications);
              if (isExpanded) {
                newExpanded.delete(n._id);
              } else {
                newExpanded.add(n._id);
              }
              setExpandedNotifications(newExpanded);
            };

            // Parse full ML message if available
            let mlData = null;
            try {
              if (n.fullMLMessage) {
                mlData = JSON.parse(n.fullMLMessage);
              }
            } catch (e) {
              console.error("Error parsing ML message:", e);
            }

            const formattedMessage = formatMLMessage(n);
            const hasResponded = n.userResponse || n.status === "responded";

            return (
              <div
                key={n._id}
                className="bg-white border border-gray-200 shadow-sm rounded-xl p-4 mb-4 hover:shadow-md transition-shadow"
              >
                {/* Status Badge */}
                {n.mlStatus && (
                  <div className="mb-2">
                    <span
                      className={`text-xs font-semibold px-2 py-1 rounded-full ${
                        n.mlStatus === "red"
                          ? "bg-red-100 text-red-700"
                          : n.mlStatus === "orange"
                          ? "bg-orange-100 text-orange-700"
                          : n.mlStatus === "yellow"
                          ? "bg-yellow-100 text-yellow-700"
                          : "bg-green-100 text-green-700"
                      }`}
                    >
                      {n.mlStatus.toUpperCase()}
                    </span>
                  </div>
                )}

                {/* Main Message (Same format as Toast) */}
                <div className="mb-3">
                  <p className="text-sm text-gray-800 whitespace-pre-wrap font-medium">
                    {formattedMessage}
                  </p>
                </div>

                {/* Expandable Full ML Response */}
                {mlData && (
                  <div className="mt-3">
                    <button
                      onClick={toggleExpand}
                      className="flex items-center gap-2 text-xs text-blue-600 hover:text-blue-800 font-medium"
                    >
                      {isExpanded ? (
                        <>
                          <ChevronUp className="w-3 h-3" />
                          Hide Full ML Response
                        </>
                      ) : (
                        <>
                          <ChevronDown className="w-3 h-3" />
                          Show Full ML Response
                        </>
                      )}
                    </button>
                    {isExpanded && (
                      <div className="mt-2 p-3 bg-gray-50 rounded-lg border border-gray-200">
                        <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono overflow-auto max-h-96">
                          {JSON.stringify(mlData, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                )}

                {/* User Response Badge */}
                {hasResponded && (
                  <div className="mt-2">
                    <span
                      className={`text-xs font-semibold px-2 py-1 rounded-full ${
                        n.userResponse === "yes"
                          ? "bg-green-100 text-green-700"
                          : "bg-red-100 text-red-700"
                      }`}
                    >
                      {n.userResponse ? n.userResponse.toUpperCase() : "RESPONDED"}
                    </span>
                  </div>
                )}

                {/* Action Buttons */}
                {!hasResponded && (
                  <div className="mt-4 pt-3 border-t border-gray-200">
                    <p className="text-sm font-medium text-gray-700 mb-3">
                      Weather to clean solar panel or not?
                    </p>
                    <div className="flex justify-end gap-3">
                      <button
                        onClick={() => handleYes(n._id)}
                        className="px-4 py-2 rounded-lg text-sm font-medium transition-all bg-green-100 hover:bg-green-200 text-green-700 hover:shadow-md"
                      >
                        YES
                      </button>
                      <button
                        onClick={() => handleNo(n._id)}
                        className="px-4 py-2 rounded-lg text-sm font-medium transition-all bg-red-100 hover:bg-red-200 text-red-700 hover:shadow-md"
                      >
                        NO
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default NotificationSidebar;
