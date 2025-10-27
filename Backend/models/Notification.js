import mongoose from "mongoose";

const notificationSchema = new mongoose.Schema(
  {
    message: { type: String, required: true },
    status: { type: String, enum: ["pending", "responded"], default: "pending" },
    userResponse: { type: String, enum: ["yes", "no", null], default: null },
  },
  { timestamps: true }
);

export default mongoose.model("Notification", notificationSchema);
