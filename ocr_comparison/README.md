# Mô hình 3: Mở rộng Baseline tích hợp OCR (PIMA_OCR)

Thư mục `ocr_comparison` (thuộc hệ thống PIMA-VAIPE) chứa mã nguồn cho **Phương pháp PIMA có tích hợp công nghệ Trích xuất Quang học (OCR)**.

Trong các phiên bản trước (như Baseline và Ablation), hệ thống giả định rằng các Bounding Box chứa tên thuốc trên đơn thuốc đã được con người đánh dấu sẵn (Ground Truth). Tuy nhiên, để đưa mô hình vào ứng dụng thực tế, mô hình phải tự đọc được chữ từ ảnh đơn thuốc thô. Thư mục này giải quyết bài toán đó.

---

## 🔬 Sự Khác Biệt & Kiến trúc Kỹ thuật

Phiên bản tích hợp OCR vẫn sử dụng lõi thuật toán của Baseline (`ResNet50` + `GraphSAGE`), nhưng quy trình xử lý dữ liệu đầu vào (Data Pipeline) được thiết kế lại hoàn toàn:

### 1. Trích xuất Text Tự động (PaddleOCR / PP-OCRv3)
- Thay vì nạp nhãn văn bản có sẵn, hệ thống nhúng trực tiếp thư viện **PaddleOCR**.
- Quá trình tiền xử lý ảnh và chạy OCR được hỗ trợ bởi các script như `preprocess_paddleocr.py` và có thể mô phỏng giả lập OCR thông qua `simulate_ocr.py`.
- Khi nạp một ảnh đơn thuốc, hệ thống tự động chạy mô hình OCR để dò tìm các vùng văn bản, sinh ra các Bounding Box và dự đoán chuỗi ký tự bên trong đó.
- Các chuỗi ký tự này (có thể có nhiễu hoặc sai chính tả do lỗi OCR) sau đó mới được đưa vào Text Encoder.

### 2. Các nhánh còn lại (Vision & Graph)
- **Đồ thị (Graph):** Đồ thị lúc này được dựng lên từ tọa độ không gian do AI OCR dự đoán, thay vì tọa độ do con người gán nhãn.
- **Hình ảnh (Vision):** Vẫn sử dụng `ResNet50` nhận đầu vào là ảnh viên thuốc đã cắt.
- **Ghép cặp (Matching):** Vẫn sử dụng Contrastive Learning, nhưng thử thách khó hơn rất nhiều vì văn bản đầu vào chứa nhiễu từ bộ OCR.

---

## ⚙️ Cấu Trúc Mã Nguồn

Các tệp mã nguồn chính phục vụ cho việc huấn luyện và đánh giá mô hình:
- `train_ocr.py`: Script dùng để huấn luyện mô hình. Hàm `main` hỗ trợ cờ đánh giá qua `--use_ocr`. Loss chính bao gồm `Contrastive Loss` (khớp văn bản-ảnh) và `Graph Loss`. Mô hình tốt nhất được lưu lại tự động (`model_best_ocr.pth` / `model_best_no_ocr.pth`).
- `eval.py`: Script tải trọng số đã train tương ứng và đánh giá độ chính xác ghép cặp (Top-1 Matching Accuracy) trên tập kiểm thử (Test/Val).
- `preprocess_paddleocr.py`: Script hỗ trợ việc trích xuất văn bản và bounding box từ ảnh gốc dựa trên PaddleOCR.
- `simulate_ocr.py`: Giả lập dữ liệu OCR phục vụ cho các trường hợp thử nghiệm.
- `plot_curves.py`: Công cụ trực quan hóa biểu đồ learning curves (Accuracy / Loss) sau quá trình train.

---

## 📊 Đánh giá Phương pháp (Pros & Cons)

- **Ưu điểm:** Khả năng ứng dụng thực tế (End-to-End) cao hơn so với Baseline. Mô hình tự động hoàn thiện quy trình phân tích văn bản mà không cần sự can thiệp của con người.
- **Nhược điểm:** 
  - Gánh chịu **Sai số tích lũy (Cascading Errors)**: Bất kỳ một lỗi nhận diện chữ nào của OCR đều sẽ bị khuếch đại khi đưa vào GraphSAGE và làm độ chính xác matching giảm sút nghiêm trọng.
  - Tốc độ xử lý (Inference Time) chậm hơn do phải gánh thêm tiến trình quét ảnh bằng OCR trước khi chạy GNN.

---

## 📈 Kết Quả Đánh Giá & Thực Nghiệm

Nhằm mục đích so sánh khách quan mức độ ảnh hưởng của sai số OCR đến hệ thống, chúng tôi tiến hành đánh giá trên cả 2 kịch bản (có dùng OCR và không dùng OCR nhưng chạy chung một kiến trúc mô hình). 

Dưới đây là bảng tổng hợp các thông số và kết quả chi tiết sau quá trình huấn luyện:

| Kịch Bản | Phương Pháp | Loss Huấn Luyện (Cuối) | Độ Chính Xác (Val Accuracy) | Top-1 Matching (Đánh giá) |
| :--- | :--- | :---: | :---: | :---: |
| **Kịch bản 1 (Không dùng OCR)** | *Ground Truth Text & BBox* | 0.239 (Epoch 10) | 44.08% | **44.26%** |
| **Kịch bản 2 (Có dùng OCR)** | *PaddleOCR End-to-End* | 0.648 (Epoch 10) | 50.95% | **44.31%** |
| **Kịch bản Ablation** | *Thử nghiệm cắt bỏ* | 0.816 (Epoch 6) | 43.73% | 0.00% |

### Chi Tiết Quá Trình Huấn Luyện

**1. Kịch bản 1: Không dùng OCR (Ground Truth Text & Bounding Box)**
Sử dụng nhãn và tọa độ Bounding Box được gán chuẩn xác bởi con người (100% chính xác), không bị nhiễu văn bản.
- **File trọng số:** `model_best_no_ocr.pth`
- **File log ghi nhận:** `train_no_ocr.log`, `log_eval.log`
- **Train Loss (Total):** 0.239 | **Top-1 Matching Accuracy:** 44.26%
- *Đánh giá:* Mô hình đạt hiệu năng tốt khi dữ liệu sạch (không bị nhiễu).

**2. Kịch bản 2: Có dùng OCR (End-to-End với PaddleOCR)**
Hệ thống tự động đọc văn bản và trích xuất Bounding Box. Dữ liệu văn bản chứa các sai số tự nhiên.
- **File trọng số:** `model_best_ocr.pth`
- **File log ghi nhận:** `train_ocr.log`, `log_eval.log`
- **Train Loss (Total):** 0.648 | **Top-1 Matching Accuracy:** 44.31%
- *Đánh giá:* Ở kịch bản dùng OCR, Train Loss khó hội tụ hơn (`0.648` so với `0.239`) do mô hình phải đối mặt với sai số tích lũy (Cascading Errors). Tuy nhiên, độ chính xác Top-1 (44.31%) vẫn gần tương đương với Kịch bản 1. Điều này cho thấy khả năng chống chịu nhiễu cực tốt của nhánh Vision và Graph.

**3. Kịch bản Ablation**
Thử nghiệm phân rã (Ablation study) trên mô hình.
- **Train Loss (Total):** 0.816 | **Validation Accuracy:** 43.73%

> 💡 **Lưu ý:** Cơ chế Early Stopping đã được kích hoạt trong hầu hết các kịch bản sau khoảng 5 đến 10 epochs. 

