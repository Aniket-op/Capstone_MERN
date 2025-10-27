import { useEffect, useState } from "react";
import axios from "axios";
import { X } from "lucide-react";

const NotificationPanel = ({ isOpen, onClose }) => {
  const [notifications, setNotifications] = useState([]);

  const fetchNotifications = async () => {
    try {
      const { data } = await axios.get("http://localhost:5000/api/demo/notifications", {
        headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
      });
      setNotifications(data);
    } catch (err) {
      console.error("Error fetching notifications:", err);
    }
  };

  const handleResponse = async (id, status) => {
    try {
      await axios.put(
        `http://localhost:5000/api/demo/notifications/${id}`,
        { status },
        { headers: { Authorization: `Bearer ${localStorage.getItem("token")}` } }
      );
      fetchNotifications(); // refresh list after update
    } catch (err) {
      console.error("Error updating notification:", err);
    }
  };

  useEffect(() => {
    if (isOpen) fetchNotifications();
  }, [isOpen]);

  return (
    <div
      className={`fixed top-0 right-0 h-full w-80 bg-white shadow-lg transform transition-transform duration-300 z-50 
      ${isOpen ? "translate-x-0" : "translate-x-full"}`}
    >
      <div className="flex justify-between items-center p-4 border-b">
        <h2 className="text-lg font-semibold text-gray-800">Notifications</h2>
        <button onClick={onClose} className="text-gray-600 hover:text-gray-900">
          <X size={20} />
        </button>
      </div>

      <div className="overflow-y-auto h-[calc(100%-64px)] p-4">
        {notifications.length > 0 ? (
          notifications.map((note) => (
            <div
              key={note._id}
              className="border-b py-3 px-2 rounded-lg bg-gray-50 hover:bg-gray-100 mb-2"
            >
              <p className="text-gray-800 font-medium">{note.message}</p>
              <p className="text-xs text-gray-500">{new Date(note.createdAt).toLocaleString()}</p>

              {note.status === "pending" ? (
                <div className="mt-2 flex gap-3">
                  <button
                    onClick={() => handleResponse(note._id, "accepted")}
                    className="bg-green-500 hover:bg-green-600 text-white text-sm px-3 py-1 rounded"
                  >
                    Yes
                  </button>
                  <button
                    onClick={() => handleResponse(note._id, "rejected")}
                    className="bg-red-500 hover:bg-red-600 text-white text-sm px-3 py-1 rounded"
                  >
                    No
                  </button>
                </div>
              ) : (
                <p
                  className={`mt-2 text-sm font-medium ${
                    note.status === "accepted"
                      ? "text-green-600"
                      : "text-red-500"
                  }`}
                >
                  You selected: {note.status === "accepted" ? "Yes" : "No"}
                </p>
              )}
            </div>
          ))
        ) : (
          <p className="text-gray-500 text-center mt-6">No notifications</p>
        )}
      </div>
    </div>
  );
};

export default NotificationPanel;
