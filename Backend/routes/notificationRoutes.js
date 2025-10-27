import express from "express";
import Notification from "../models/Notification.js";
import { protect } from "../middlewares/authMiddleware.js"; // Assuming JWT auth

const router = express.Router();

// GET all notifications for logged-in user
router.get("/", protect, async (req, res) => {
  try {
    const notifications = await Notification.find({ userId: req.user._id }).sort({ createdAt: -1 });
    res.json(notifications);
  } catch (err) {
    res.status(500).json({ message: "Server error fetching notifications" });
  }
});

// POST - create new notification (e.g., cleaning request)
router.post("/", protect, async (req, res) => {
  try {
    const notification = new Notification({
      message: req.body.message,
      type: req.body.type || "cleaning",
      userId: req.user._id,
    });
    await notification.save();
    res.status(201).json(notification);
  } catch (err) {
    res.status(500).json({ message: "Error creating notification" });
  }
});

// PUT - update status (Yes/No response)
router.put("/:id", protect, async (req, res) => {
  try {
    const notification = await Notification.findById(req.params.id);
    if (!notification) return res.status(404).json({ message: "Not found" });

    notification.status = req.body.status; // 'accepted' or 'rejected'
    await notification.save();

    res.json(notification);
  } catch (err) {
    res.status(500).json({ message: "Error updating notification" });
  }
});

export default router;
