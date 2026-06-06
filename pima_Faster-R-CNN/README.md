# PIMA Faster R-CNN: Hướng Tiếp Cận End-to-End

Thư mục `pima_Faster-R-CNN` chứa phiên bản mô hình xử lý hình ảnh dựa trên **Faster R-CNN**. Khác với cách tiếp cận Vision Transformer (ViT) đòi hỏi phải cắt viên thuốc ra khỏi nền, phiên bản này thực hiện trích xuất đặc trưng trực tiếp từ ảnh đơn thuốc nguyên bản (End-to-End).

---

## 🌟 Kiến Trúc Hệ Thống

### 1. Nhánh Thị Giác (Vision Branch) - Faster R-CNN
- **Mô hình cốt lõi:** `FasterRCNN_ResNet50_FPN`.
- **Nguyên lý hoạt động:** Nhận đầu vào là ảnh gốc cùng các tọa độ Bounding Box của viên thuốc. Mạng sử dụng xương sống (backbone) ResNet50 kết hợp với mạng kim tự tháp đặc trưng (FPN). Sau đó, module **RoI Align** được sử dụng để trích xuất trực tiếp vector đặc trưng cho từng viên thuốc từ các hộp giới hạn (Bounding Box) đó.
- **Ưu điểm:** Hoàn toàn End-to-End, dễ dàng tích hợp vào hệ thống thực tế vì không cần công cụ cắt nền phụ trợ.
- **Thử thách:** Đặc trưng trích xuất chứa yếu tố nhiễu của nền, đòi hỏi mô hình phải học cách lọc nhiễu tốt hơn.

### 2. Nhánh Ngôn Ngữ, OCR & Đồ Thị (Language, OCR & Graph Branch)
- Vẫn kế thừa những nâng cấp tốt nhất từ phiên bản PIMA_NEW:
  - **OCR:** Tích hợp End-to-End với `PaddleOCR` (PP-OCRv3) để tự động trích xuất chữ và Bounding Box từ ảnh đơn thuốc, hỗ trợ đọc văn bản tự nhiên hiệu quả.
  - **Ngôn ngữ:** Sử dụng `all-MiniLM-L6-v2` để nhúng văn bản nhánh và chính xác.
  - **Đồ thị:** Sử dụng Relational Graph Attention (`RGATConv`) để hiểu không gian có hướng của các hộp văn bản trên đơn thuốc, kết hợp với bộ phân loại Pseudo-Classifier dự đoán xác suất chứa tên thuốc.

---

## 📊 Kết Quả Huấn Luyện & Đánh Giá

Việc sử dụng phương pháp End-to-End đồng nghĩa với việc giữ lại lượng mẫu lớn hơn (không bị loại bỏ ở bước cắt ảnh). 

- **Tổng số cặp thuốc được đánh giá (Evaluation Pairs):** `7,058`
- **Khớp chính xác (Top-1 Matches):** Lên đến `5,635` cặp
- **Độ chính xác (Accuracy):** Dao động trong khoảng **79.84%** đến **83.47%** (tùy epoch tốt nhất).

> **Nhận xét:** So với phiên bản Baseline cũ, hệ thống Faster R-CNN kết hợp R-GAT và InfoNCE Loss đã cải thiện hiệu năng vượt bậc. Dù xử lý ảnh thô có nhiễu, mô hình vẫn đạt độ chính xác xấp xỉ ~80-83%, biến đây thành phiên bản cực kỳ phù hợp để triển khai thực tế.

---

## ⚙️ Cấu Trúc Mã Nguồn

- `train.py`: Hỗ trợ huấn luyện đa GPU với `torchrun` (DDP). Có sử dụng Gradient Accumulation (`--accumulate-steps`) và Early Stopping.
- `eval.py`: Chạy đánh giá độ chính xác của mô hình trên tập validation.
- `plot_from_json.py`: Hỗ trợ vẽ biểu đồ Loss và Accuracy từ file `train_results.json` (`train_0306.json`).
- `preprocess_pills.py`, `data_split.py`: Chuẩn bị và chia dữ liệu (Train/Val).
- `models/`: Chứa định nghĩa toàn bộ kiến trúc mạng (Faster R-CNN, R-GAT, MiniLM, Cross-Attention).
