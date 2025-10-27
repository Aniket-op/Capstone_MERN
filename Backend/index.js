import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import connectDB from "./config/db.js"; // <-- note the .js
import authRoutes from "./routes/authRoutes.js";
import solarRoutes from "./routes/solarRoutes.js";
import solar from "./routes/solar.js";
import { errorHandler } from "./middlewares/errorMiddleware.js";
import notificationRoutes from "./routes/notificationRoutes.js";
dotenv.config();
connectDB();

const app = express();
app.use(cors());
app.use(express.json());

app.use("/api/auth", authRoutes);
app.use("/api/solar", solarRoutes);
app.use("/api/demo/solar", solar);
app.use("/api/notifications", notificationRoutes);



app.use(errorHandler);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
