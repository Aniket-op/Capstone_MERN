import Notification from "../models/Notification.js";
import { io } from "../server.js";

// Create notification
export const createNotification = async (data) => {
  try {
    const notification = new Notification({
      message: data.message,
      status: "pending",
      userId: data.userId || null, // Optional: link to specific user
      fullMLMessage: data.fullMLMessage || null, // Store full ML server response
      powerLoss: data.powerLoss || null,
      mlStatus: data.status || null, // normal, yellow, orange, red
    });

    const saved = await notification.save();
    
    // Emit to all connected clients
    io.emit("newNotification", saved);
    
    console.log("📬 Notification created:", saved._id);
    return saved;
  } catch (error) {
    console.error("❌ Error creating notification:", error);
    throw error;
  }
};

// Get all notifications
export const getAllNotifications = async (userId) => {
  try {
    const query = userId ? { userId } : {};
    const notifications = await Notification.find(query)
      .sort({ createdAt: -1 })
      .limit(50);
    return notifications;
  } catch (error) {
    console.error("❌ Error fetching notifications:", error);
    throw error;
  }
};

// Update notification response
export const updateNotificationResponse = async (notificationId, response) => {
  try {
    const notification = await Notification.findByIdAndUpdate(
      notificationId,
      {
        userResponse: response,
        status: "responded",
      },
      { new: true }
    );

    if (!notification) {
      throw new Error("Notification not found");
    }

    // Emit update
    io.emit("notificationUpdated", notification);

    return notification;
  } catch (error) {
    console.error("❌ Error updating notification:", error);
    throw error;
  }
};