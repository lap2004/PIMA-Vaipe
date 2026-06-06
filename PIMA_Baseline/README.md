# Mô hình 1: PIMA Baseline

Thư mục `PIMA_Baseline` (thuộc hệ thống PIMA-VAIPE) chứa mã nguồn gốc của mô hình **PIMA Baseline**. Đây là phiên bản cơ sở chuẩn để so sánh và làm nền tảng cho các biến thể khác như PIMA_OCR hay PIMA_ablation.

Mô hình này nhằm giải quyết bài toán **ghép cặp (matching)** giữa hình ảnh viên thuốc (Pill Image) và tên thuốc (Drug Name) trên đơn thuốc. Khác với phiên bản OCR tự động, ở mô hình Baseline này, hệ thống sử dụng **Ground Truth**, giả định rằng tọa độ (Bounding Box) và văn bản (Text) trên đơn thuốc đã được trích xuất chuẩn xác 100%.

---

## 🔬 Kiến trúc Kỹ thuật

Baseline sử dụng hai nhánh mạng nơ-ron chính để trích xuất đặc trưng đa phương thức (Multimodal):
1. **Nhánh Hình ảnh (Vision):** Sử dụng mạng `ResNet50` để trích xuất đặc trưng ảnh từ các viên thuốc đã được cắt (cropped pill images).
2. **Nhánh Văn bản & Không gian (Text & Spatial Graph):** Sử dụng Mạng Neural Đồ thị (`GraphSAGE`). Đồ thị được xây dựng từ các node văn bản, các cạnh liên kết dựa trên khoảng cách không gian (spatial coordinates) trên đơn thuốc. 

**Cơ chế Ghép cặp (Matching):**
- Hai luồng đặc trưng (ảnh và chữ/đồ thị) được chiếu (projected) vào cùng một không gian nhúng (embedding space).
- Hàm mất mát **Contrastive Loss** được sử dụng để tối ưu khoảng cách: kéo gần các cặp (ảnh thuốc - tên thuốc) đúng và đẩy xa các cặp sai. 
- Hàm **NLLLoss** được áp dụng bổ trợ để phân loại đồ thị.

---

## ⚙️ Cấu Trúc Mã Nguồn

- `train.py`: Script huấn luyện chính của mô hình Baseline. Nó nạp dữ liệu từ thư mục cấu hình sẵn, tiến hành huấn luyện qua các epoch và lưu lại trọng số tốt nhất `model_best.pth`. Nó tự động đánh giá Validation trong quá trình huấn luyện.
- `inference.py`: Script chạy dự đoán thực tế. Khi đưa vào một cấu trúc dữ liệu đơn thuốc, mô hình sẽ tính toán độ tương đồng Cosine (Cosine Similarity) giữa ảnh và đồ thị chữ, sử dụng ngưỡng mềm (threshold > 0.5) để dự đoán thuốc tương ứng hoặc trả về "No match".
- `preprocess.py`: Các hàm tiền xử lý dữ liệu và tạo đồ thị ban đầu cho mô hình chuẩn.
- `config.py`: Tệp chứa các hằng số, cấu hình, siêu tham số và đường dẫn hệ thống.
