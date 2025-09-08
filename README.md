# Ứng dụng web dự đoán ung thư vú (Random Forest + Flask)

Ứng dụng Flask này sử dụng **Random Forest** để dự đoán ung thư vú (ác tính/ lành tính) trên bộ dữ liệu **Wisconsin Breast Cancer** có sẵn trong `scikit-learn`.

## Tính năng
- Trang web có 2 chế độ:
  1. **Dự đoán đơn lẻ**: nhập giá trị cho 30 đặc trưng và nhận kết quả tức thì.
  2. **Dự đoán theo lô (CSV)**: tải lên tệp `.csv` với đúng tên cột, hệ thống trả về file kết quả kèm xác suất.

- Pipeline gồm **StandardScaler** và **RandomForestClassifier** (300 cây).

## Cài đặt & chạy
> Yêu cầu Python 3.9+

1. Tạo và kích hoạt virtualenv (khuyến nghị):
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

2. Cài thư viện:
   ```bash
   pip install -r requirements.txt
   ```

3. Huấn luyện mô hình:
   ```bash
   python train_model.py
   ```

4. Chạy server Flask (dev):
   ```bash
   python app.py
   # Mặc định http://127.0.0.1:5000
   ```

## CSV mẫu cho batch
- File CSV cần **đủ cột** và **đúng tên** như hiển thị trên trang chủ (theo `feature_names` của sklearn).  
- Thứ tự cột có thể bất kỳ, ứng dụng sẽ đọc theo tên cột.

Ví dụ tên cột (rút gọn, thực tế có 30 cột):
```
mean radius, mean texture, mean perimeter, mean area, mean smoothness, ..., worst fractal dimension
```

## Ghi chú quan trọng
- Ứng dụng này dùng dữ liệu mẫu từ `scikit-learn` để minh họa. Nếu bạn có **code Random Forest riêng** hoặc **tập dữ liệu khác**,
  bạn có thể:
  - Chỉnh sửa `train_model.py` để đọc dữ liệu của bạn (CSV/Parquet…) và huấn luyện lại.
  - Đảm bảo **thứ tự và tên cột** được ghi vào `meta.json` (biến `feature_names`) để Flask biết cách nhận dữ liệu đầu vào.
- Với dữ liệu y tế thật, cần xem xét chuẩn hoá, xử lý mất cân bằng, đánh giá mô hình nghiêm ngặt (ROC-AUC, calibration,…).
