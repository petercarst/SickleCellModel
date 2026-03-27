import "dotenv/config";
import express from "express";
import multer from "multer";
import fetch from "node-fetch";
import FormData from "form-data";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const app  = express();
const PORT = process.env.PORT || 3000;

// ── FastAPI endpoint ──────────────────────────────────────────────────────────
const FASTAPI_URL = process.env.FASTAPI_URL || "http://localhost:8000/predict";

// ── Multer: memory storage (no disk writes) ───────────────────────────────────
const ALLOWED_MIMES = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
const MAX_SIZE      = 10 * 1024 * 1024; // 10 MB

const upload = multer({
  storage: multer.memoryStorage(),
  limits:  { fileSize: MAX_SIZE },
  fileFilter(_req, file, cb) {
    if (ALLOWED_MIMES.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error("Invalid file type. Please upload a JPEG, JPG, PNG, or WebP image."));
    }
  },
});

// ── Serve static frontend files from /public ─────────────────────────────────
app.use(express.static(path.join(__dirname, "public")));

// ── POST /predict ─────────────────────────────────────────────────────────────
app.post("/predict", upload.single("file"), async (req, res) => {
  // multer fileFilter rejection lands here via error handler below,
  // so if we reach this point the file is valid.
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded." });
  }

  try {
    // Build a multipart form to forward to FastAPI
    const form = new FormData();
    form.append("file", req.file.buffer, {
      filename:    req.file.originalname,
      contentType: req.file.mimetype,
    });

    const fastapiRes = await fetch(FASTAPI_URL, {
      method:  "POST",
      body:    form,
      headers: form.getHeaders(),
      // 30-second timeout via AbortController
      signal:  AbortSignal.timeout(30_000),
    });

    if (!fastapiRes.ok) {
      const text = await fastapiRes.text();
      return res.status(502).json({
        error:  "TF Serving / FastAPI error.",
        detail: text,
      });
    }

    const data = await fastapiRes.json();

    // Relay FastAPI error messages transparently
    if (data.error) {
      return res.status(200).json({ error: data.error });
    }

    return res.json(data);

  } catch (err) {
    // AbortSignal.timeout fires a TimeoutError
    if (err.name === "TimeoutError" || err.name === "AbortError") {
      return res.status(504).json({ error: "Request to prediction service timed out." });
    }
    // FastAPI not running / connection refused
    if (err.code === "ECONNREFUSED" || err.cause?.code === "ECONNREFUSED") {
      return res.status(503).json({
        error:  "Could not reach the prediction service. Please ensure the FastAPI server is running.",
        detail: err.message,
      });
    }
    console.error("Unexpected error:", err);
    return res.status(500).json({ error: "Internal server error.", detail: err.message });
  }
});

// ── Multer / file-validation error handler ────────────────────────────────────
app.use((err, _req, res, _next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === "LIMIT_FILE_SIZE") {
      return res.status(400).json({ error: "File too large. Maximum size is 10 MB." });
    }
    return res.status(400).json({ error: err.message });
  }
  if (err) {
    return res.status(400).json({ error: err.message });
  }
  _next();
});

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`SickleVision backend running at http://localhost:${PORT}`);
  console.log(`Forwarding predictions to: ${FASTAPI_URL}`);
});
