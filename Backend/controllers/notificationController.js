import Notification from "../models/Notification.js";
import { 
  getAllNotifications, 
  updateNotificationResponse,
  createNotification 
} from "../services/notificationService.js";
import mqttService from "../services/mqttService.js";

// Get all notifications
export const getNotifications = async (req, res) => {
  try {
    const { userId, status, limit = 50 } = req.query;
    
    const filter = {};
    if (userId) filter.userId = userId;
    if (status) filter.status = status;
    
    const notifications = await Notification.find(filter)
      .sort({ createdAt: -1 })
      .limit(parseInt(limit));
    
    res.status(200).json({
      success: true,
      count: notifications.length,
      data: notifications,
    });
  } catch (err) {
    console.error("Error fetching notifications:", err);
    res.status(500).json({ 
      success: false,
      error: "Failed to fetch notifications" 
    });
  }
};

// Get single notification
export const getNotificationById = async (req, res) => {
  try {
    const notification = await Notification.findById(req.params.id);
    
    if (!notification) {
      return res.status(404).json({
        success: false,
        message: "Notification not found",
      });
    }
    
    res.status(200).json({
      success: true,
      data: notification,
    });
  } catch (err) {
    console.error("Error fetching notification:", err);
    res.status(500).json({ 
      success: false,
      error: "Failed to fetch notification" 
    });
  }
};

// Create notification (manual)
export const createNotificationManual = async (req, res) => {
  try {
    const { message, userId } = req.body;
    
    if (!message) {
      return res.status(400).json({
        success: false,
        error: "Message is required",
      });
    }
    
    const notification = await createNotification({
      message,
      userId: userId || req.user?._id,
    });
    
    res.status(201).json({
      success: true,
      data: notification,
    });
  } catch (err) {
    console.error("Error creating notification:", err);
    res.status(500).json({ 
      success: false,
      error: "Failed to create notification" 
    });
  }
};

// Respond to notification (user action)
export const respondToNotification = async (req, res) => {
  try {
    const { id } = req.params;
    const { response } = req.body; // "yes" or "no"
    
    if (!["yes", "no"].includes(response)) {
      return res.status(400).json({
        success: false,
        error: "Response must be 'yes' or 'no'",
      });
    }
    
    const notification = await updateNotificationResponse(id, response);
    
    // If user confirms cleaning, send command to hardware
    if (response === "yes") {
      mqttService.publishCommand({
        action: "START_CLEANING",
        timestamp: new Date().toISOString(),
        notificationId: id,
      });
      
      console.log("🧹 Cleaning command sent to hardware");
    }
    
    res.status(200).json({
      success: true,
      message: `Notification ${response === "yes" ? "confirmed" : "rejected"}`,
      data: notification,
    });
  } catch (err) {
    console.error("Error responding to notification:", err);
    res.status(500).json({ 
      success: false,
      error: "Failed to respond to notification" 
    });
  }
};

// Delete notification
export const deleteNotification = async (req, res) => {
  try {
    const notification = await Notification.findByIdAndDelete(req.params.id);
    
    if (!notification) {
      return res.status(404).json({
        success: false,
        message: "Notification not found",
      });
    }
    
    res.status(200).json({
      success: true,
      message: "Notification deleted successfully",
    });
  } catch (err) {
    console.error("Error deleting notification:", err);
    res.status(500).json({ 
      success: false,
      error: "Failed to delete notification" 
    });
  }
};

// Mark notification as read
export const markAsRead = async (req, res) => {
  try {
    const notification = await Notification.findByIdAndUpdate(
      req.params.id,
      { status: "responded" },
      { new: true }
    );
    
    if (!notification) {
      return res.status(404).json({
        success: false,
        message: "Notification not found",
      });
    }
    
    res.status(200).json({
      success: true,
      data: notification,
    });
  } catch (err) {
    console.error("Error marking notification:", err);
    res.status(500).json({ 
      success: false,
      error: "Failed to mark notification" 
    });
  }
};

// Get unread count
export const getUnreadCount = async (req, res) => {
  try {
    const { userId } = req.query;
    
    const filter = { status: "pending" };
    if (userId) filter.userId = userId;
    
    const count = await Notification.countDocuments(filter);
    
    res.status(200).json({
      success: true,
      unreadCount: count,
    });
  } catch (err) {
    console.error("Error getting unread count:", err);
    res.status(500).json({ 
      success: false,
      error: "Failed to get unread count" 
    });
  }
};