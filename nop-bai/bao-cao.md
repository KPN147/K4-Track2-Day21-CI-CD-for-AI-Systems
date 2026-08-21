# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Phạm Nam Khánh |
| MSSV | 2A202601718 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/KPN147/K4-Track2-Day21-CI-CD-for-AI-Systems.git |
| Ngày nộp | 21/08/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 200 | 0.1 | 5 | 0.714932126... | 0.874 |
| 2 | 100 | 0.1 | 3 | 0.710900473... | 0.878 |
| 3 | 50 | 0.05 | 2 | 0.605128205... | 0.846 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ 200/0.1/5 có F1 cao nhất trong ba lần chạy, đạt khoảng 0.7149. Bộ 100/0.1/3 có accuracy cao nhất là 0.878 nhưng F1 thấp hơn, khoảng 0.7109. Điều này cho thấy accuracy không nhất thiết chọn ra mô hình tốt nhất cho lớp thu nhập cao. Trong hai cấu hình có learning rate 0.1, tăng số estimator và độ sâu cây giúp F1 tăng nhẹ; cấu hình learning rate 0.05 với ít estimator có F1 thấp nhất.

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu có khoảng 24,8% mẫu thuộc lớp thu nhập cao và khoảng 75,2% thuộc lớp thu nhập thấp. Vì vậy, một mô hình luôn dự đoán “thu nhập thấp” vẫn có thể đạt accuracy khoảng 0.752, dù không phát hiện được mẫu thu nhập cao nào. F1 của lớp dương kết hợp precision và recall, nên phản ánh tốt hơn khả năng nhận diện nhóm thu nhập cao. Quality gate dùng `f1_score(y_eval, preds)` mặc định cho lớp dương, không dùng `average="weighted"` hoặc `average="macro"`, vì các cách đó có thể bị lớp đa số kéo cao và làm sai ý nghĩa ngưỡng 0.65.

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| MLflow không chạy được lúc đầu | Môi trường thiếu `pkg_resources` khi dùng MLflow 2.13.0 | Tạo lại môi trường bằng `uv`, dùng Python 3.11 và cài `setuptools<81`. |
| API không khởi động trên VM | Sai indentation trong `src/serve.py` và sau đó model không tương thích phiên bản scikit-learn | Sửa indentation, kiểm tra chạy thủ công, rồi đồng bộ `scikit-learn==1.4.2`, `joblib==1.4.2` và `numpy==1.26.4`. |
| Triển khai CI/CD cần nhiều thông tin xác thực | DVC/GCS và SSH cần credentials riêng | Cấu hình Service Account cho GCS và lưu credentials, bucket, VM host, user, SSH key trong GitHub Secrets. |

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149321267 | 0.874 |
| Bước 3 (thêm `train_batch2`) | 0.7354260090 | 0.882 |

**Nhận xét:** Sau khi bổ sung `train_batch2`, F1 tăng từ 0.7149 lên 0.7354 và accuracy tăng từ 0.874 lên 0.882. Bước 3 được kích hoạt bởi commit dữ liệu và hoàn thành toàn bộ pipeline, cho thấy quy trình huấn luyện và triển khai liên tục hoạt động đúng.

## 5. Phần Bonus Đã Thực Hiện

- [x] Bonus 1 - Tracking MLflow từ xa với DagsHub: Các experiment được ghi lên MLflow server của DagsHub; ảnh minh chứng: `06-dagshub-mlflow.png`.
- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: Quét threshold từ 0.10 đến 0.90, chọn threshold có F1 cao nhất và sử dụng threshold đó khi serving.
- [x] Bonus 3 - Báo cáo precision / recall tự động: Tạo `outputs/detail.txt` với confusion matrix, precision và recall; file được upload cùng report artifact.
- [x] Bonus 4 - Hoàn trả về phiên bản trước: So sánh F1 model mới với model hiện tại, lưu phiên bản trước và hủy triển khai nếu F1 mới thấp hơn.
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: Tính `positive_rate`, cảnh báo khi lệch quá 5 điểm phần trăm so với tỷ lệ tham chiếu và ghi kết quả vào report.
