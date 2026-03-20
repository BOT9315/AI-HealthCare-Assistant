import os
import json
import base64
import logging
import threading
import numpy as np
import cv2
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

IMAGES_DIR   = "images"
MODEL_PATH   = "facenet_keras.h5"
DIST_THRESH  = 5.0
IMG_SIZE     = (160, 160)

os.makedirs(IMAGES_DIR, exist_ok=True)

# ── Model (loaded once, thread-safe via lock) ─────────────────────────────────
model      = tf.keras.models.load_model(MODEL_PATH)
model_lock = threading.Lock()
database: dict[str, np.ndarray] = {}

logger.info("FaceNet model loaded successfully.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def img_to_encoding(path: str) -> np.ndarray:
    """Read an image from *path* and return its 128-d FaceNet embedding."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    img_rgb = img[..., ::-1]                          # BGR → RGB

    if img_rgb.shape[:2] != IMG_SIZE:
        img_rgb = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)

    x = np.expand_dims(img_rgb.astype("float32"), axis=0)

    with model_lock:
        embedding = model.predict(x, verbose=0)

    return embedding                                  # shape (1, embedding_dim)


def decode_base64_image(b64_string: str, save_path: str) -> None:
    """Decode a base64 image string (with or without data-URI prefix) to disk."""
    # Strip optional "data:image/...;base64," prefix
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    with open(save_path, "wb") as fh:
        fh.write(base64.b64decode(b64_string))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/register", methods=["POST"])
def register():
    """
    Register a new face in the database.

    Expected JSON body:
        { "username": "alice", "image64": "<base64-encoded JPEG>" }
    """
    path = None
    try:
        payload  = request.get_json(force=True, silent=True) or {}
        username = payload.get("username", "").strip()
        img_data = payload.get("image64", "")

        # ── Input validation ──────────────────────────────────────────────────
        if not username:
            return jsonify({"status": 400, "error": "username is required"}), 400
        if not img_data:
            return jsonify({"status": 400, "error": "image64 is required"}), 400
        # Reject path-traversal attempts
        if any(c in username for c in ("/", "\\", "..")):
            return jsonify({"status": 400, "error": "Invalid username"}), 400

        path = os.path.join(IMAGES_DIR, f"{username}.jpg")
        decode_base64_image(img_data, path)

        database[username] = img_to_encoding(path)
        logger.info("Registered user: %s", username)
        return jsonify({"status": 200})

    except Exception as exc:
        logger.exception("Error during /register")
        # Clean up partial file if it was written
        if path and os.path.exists(path):
            os.remove(path)
        return jsonify({"status": 500, "error": str(exc)}), 500


@app.route("/verify", methods=["POST"])
def verify_identity():
    """
    Identify who is in the supplied image.

    Expected JSON body:
        { "image64": "<base64-encoded JPEG>" }

    Returns:
        { "identity": "<username>" }  on success
        { "identity": 0 }             when no match is found
    """
    path = None
    try:
        payload  = request.get_json(force=True, silent=True) or {}
        img_data = payload.get("image64", "")

        if not img_data:
            return jsonify({"status": 400, "error": "image64 is required"}), 400
        if not database:
            return jsonify({"status": 400, "error": "Database is empty – register users first"}), 400

        # Save to a temporary file named by current timestamp
        timestamp = str(int(datetime.now().timestamp()))
        path      = os.path.join(IMAGES_DIR, f"{timestamp}.jpg")
        decode_base64_image(img_data, path)

        encoding = img_to_encoding(path)

        # ── Find nearest neighbour ────────────────────────────────────────────
        min_dist, identity = float("inf"), None
        for name, db_enc in database.items():
            dist = l2_distance(encoding, db_enc)
            logger.debug("Distance to %s: %.4f", name, dist)
            if dist < min_dist:
                min_dist, identity = dist, name

        if min_dist > DIST_THRESH:
            logger.info("No match found (min dist=%.4f)", min_dist)
            return jsonify({"identity": 0})

        logger.info("Matched %s (dist=%.4f)", identity, min_dist)
        return jsonify({"identity": identity})

    except Exception as exc:
        logger.exception("Error during /verify")
        return jsonify({"status": 500, "error": str(exc)}), 500

    finally:
        # Always remove the temporary file
        if path and os.path.exists(path):
            os.remove(path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Use threaded=True so Flask handles concurrent requests safely
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5000)
