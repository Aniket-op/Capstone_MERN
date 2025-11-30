import mongoose from "mongoose";

const notificationSchema = new mongoose.Schema(
  {
    message: { type: String, required: true },
    status: { type: String, enum: ["pending", "responded"], default: "pending" },
    userResponse: { type: String, enum: ["yes", "no", null], default: null },
    userId: {type: mongoose.Schema.Types.ObjectId,ref: "User",required: false}, // Made optional
    fullMLMessage: { type: String }, // Store full ML server response as JSON string
    powerLoss: { type: Number },
    mlStatus: { type: String }, // normal, yellow, orange, red
  },
  { timestamps: true }
);

export default mongoose.model("Notification", notificationSchema);
