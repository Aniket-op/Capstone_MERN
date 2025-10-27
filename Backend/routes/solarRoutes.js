import express from "express";
import { getSolarData , postSolarData } from "../controllers/solarController.js";
import {protect}  from "../middlewares/authMiddleware.js";

const router = express.Router();

router.get("/", protect, getSolarData);
router.post("/", protect, postSolarData);


export default router;
