from flask import Flask, render_template, request, redirect, url_for, flash
import os
import json
import joblib
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "change-me-please"  # thay đổi khi triển khai thật

# Tải model và metadata
MODEL_PATH = os.path.join("models", "model.pkl")
META_PATH = os.path.join("models", "meta.json")

if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
    raise RuntimeError("Chưa có model. Hãy chạy: python train_model.py")

model = joblib.load(MODEL_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

FEATURE_NAMES = meta["feature_names"]
CLASS_MAP = meta.get("class_map", {"0": "malignant", "1": "benign"})  # fallback

ALLOWED_EXTENSIONS = {"csv"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    # Form đơn lẻ + Batch upload
    return render_template("index.html", feature_names=FEATURE_NAMES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Lấy dữ liệu từ form: tên input theo feature
        inputs = []
        for feat in FEATURE_NAMES:
            val = request.form.get(feat, "", type=str).strip()
            if val == "":
                raise ValueError(f"Thiếu giá trị cho '{feat}'")
            try:
                inputs.append(float(val))
            except ValueError:
                raise ValueError(f"Giá trị không hợp lệ cho '{feat}': {val}")
        X = np.array(inputs, dtype=float).reshape(1, -1)
        y_pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        # Giả sử lớp 0 = malignant, 1 = benign theo sklearn dataset
        label = CLASS_MAP.get(str(int(y_pred)), str(y_pred))

        res = {
            "label": label,
            "proba_malignant": float(proba[0]),
            "proba_benign": float(proba[1]),
        }
        return render_template("result.html", result=res, single=True)
    except Exception as e:
        flash(str(e), "danger")
        return redirect(url_for("index"))

@app.route("/batch", methods=["POST"])
def batch():
    file = request.files.get("file", None)
    if file is None or file.filename == "":
        flash("Vui lòng chọn tệp CSV.", "warning")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Chỉ chấp nhận tệp CSV.", "warning")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    temp_path = os.path.join("uploads")
    os.makedirs(temp_path, exist_ok=True)
    full_path = os.path.join(temp_path, filename)
    file.save(full_path)

    try:
        df = pd.read_csv(full_path)
        # Kiểm tra cột: cần đủ và đúng thứ tự
        missing = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing:
            raise ValueError("Thiếu cột trong CSV: " + ", ".join(missing))

        X = df[FEATURE_NAMES].values
        preds = model.predict(X)
        probas = model.predict_proba(X)

        out = df.copy()
        out["prediction"] = [CLASS_MAP.get(str(int(p)), str(p)) for p in preds]
        out["proba_malignant"] = probas[:, 0]
        out["proba_benign"] = probas[:, 1]

        # Lưu kết quả
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", f"predictions_{filename}")
        out.to_csv(out_path, index=False, encoding="utf-8-sig")

        return render_template("result.html", batch=True, table=out.head(20).to_html(classes="table table-striped", index=False), download_path=out_path)
    except Exception as e:
        flash(str(e), "danger")
        return redirect(url_for("index"))
    finally:
        try:
            os.remove(full_path)
        except Exception:
            pass

if __name__ == "__main__":
    # Chạy dev server
    app.run(host="0.0.0.0", port=5001, debug=True)
