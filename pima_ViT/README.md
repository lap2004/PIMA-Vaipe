# PIMA_NEW: Nâng Cấp Toàn Diện (Kiến trúc Đa Phương Thức - Transformer)

Thư mục `pima_ViT` (hay `PIMA_NEW`) là phiên bản **nâng cấp đột phá** của hệ thống PIMA-VAIPE. Phiên bản này thay thế gần như toàn bộ các module cũ bằng các công nghệ Deep Learning tiên tiến nhất (SOTA), chuyển dịch từ mô hình đối chiếu đồ thị đơn giản sang một **Khung Cross-Attention Đa Phương Thức dựa trên Transformer**.

Đây là phiên bản được tối ưu hóa chuyên sâu nhất cho bộ dữ liệu VAIPE.

---

Phiên bản này sử dụng **Vision Transformer (ViT-B/16)** cho luồng xử lý thị giác:

### Vision Transformer (ViT-B/16) - *Cách tiếp cận "Cắt Gọn Hoàn Hảo"*
- **Nguyên lý hoạt động:** Giả định viên thuốc đã được cắt khỏi nền (sử dụng công cụ như `rembg`). Ảnh viên thuốc sạch được đưa qua mô hình `ViT-B/16`.
- **Ưu điểm:** ViT không bị nhiễu nền nên có thể tập trung hoàn toàn vào việc bắt giữ các đặc trưng hình thái, đem lại độ chính xác cực kỳ cao.
- **Nhược điểm:** Đòi hỏi bước tiền xử lý ảnh riêng biệt. Không thể tự động dò tìm viên thuốc trên ảnh đơn thuốc thô.
- **Độ chính xác (Accuracy):** `82.46%`

---

## 🧠 Những Đổi Mới Cốt Lõi 

Các module khác được xây dựng xoay quanh ViT bao gồm:

### 1. Nhánh Ngôn Ngữ & OCR: MiniLM + PP-OCRv3
- **Thay thế BERT bằng `all-MiniLM-L6-v2`:** Cung cấp vector nhúng câu văn nhanh và chính xác hơn rất nhiều so với phiên bản trước.
- **Tích hợp End-to-End OCR:** Hỗ trợ `PaddleOCR` (PP-OCRv3) cho phép mạng lưới tự động trích xuất chữ và Bounding Box trực tiếp từ ảnh chụp đơn thuốc.

### 2. Nhánh Đồ Thị: Relational Graph Attention (R-GAT)
- **Thay thế `GraphSAGE` bằng `RGATConv`:** 
- Khác với GCN thông thường, R-GAT có thể hiểu được *mối liên kết không gian có hướng* (trái, phải, trên, dưới) giữa các hộp văn bản.
- **Pseudo-Classifier:** Nhánh đồ thị có thêm một bộ phân loại nhị phân dự đoán xác suất một hộp văn bản có chứa tên thuốc hay không, đóng vai trò như một cổng chú ý (attention gate).

### 3. Ghép cặp (Fusion): Multi-Modal Cross-Attention
- **Thay thế Cosine Similarity cơ bản bằng module `MultiModalCrossAttention` mạnh mẽ.**
- Đặc trưng hình ảnh đóng vai trò làm **Queries**, trong khi đặc trưng văn bản từ GNN là **Keys/Values**. Điều này giúp mạng lưới tự động "tìm kiếm" phần text phù hợp nhất trên đơn thuốc dựa vào diện mạo của viên thuốc.

### 4. Hàm Mục Tiêu: InfoNCE Loss
- **Thay thế Contrastive Loss tiêu chuẩn bằng `InfoNCE Loss`.**
- Tối ưu hóa Cross-entropy trên toàn bộ Batch (với hệ số nhiệt độ - temperature), kéo gần các cặp đa phương thức đúng lại với nhau và đồng thời đẩy xa tất cả các cặp sai lệch.

---

## ⚙️ Cấu Trúc Mã Nguồn

- `train.py`: Hỗ trợ huấn luyện đa GPU với `torchrun` (Distributed Data Parallel - DDP). 
- `eval.py`: Kịch bản đánh giá mô hình, đo lường độ chính xác Top-1 Matching trên tập dữ liệu.
- `preprocess_pills.py`, `data_split.py`: Các script hỗ trợ chuẩn bị và chia tập dữ liệu huấn luyện.

---

## 📈 Kết Quả Đánh Giá & Thực Nghiệm (Chi Tiết)

Quá trình đánh giá được thực hiện trên tập dữ liệu kiểm thử (test set) nhằm đo lường khả năng khớp hình ảnh viên thuốc thực tế với tên thuốc.

| Mô Hình | Phương Pháp | Tổng Số Lượt Đánh Giá | Khớp Chính Xác (Top-1) | Độ Chính Xác (Accuracy) |
| :--- | :--- | :---: | :---: | :---: |
| **Vision Transformer (ViT-B/16)** | *Cắt Gọn Hoàn Hảo* | 3,627 | 2,991 | **82.46%** |

### Mô Hình Vision Transformer (ViT-B/16)
- **Tổng số viên thuốc được đánh giá (có nhãn khớp):** 3,627
- **Số viên khớp chính xác (Top-1):** 2,991
- **Average Loss (Dừng sớm tại Epoch 45):** 0.2075
> **Nhận xét:** Việc loại bỏ được nhiễu từ nền xung quanh giúp mạng lưới Transformer tập trung hoàn toàn vào việc bắt giữ các đặc trưng hình thái của viên thuốc, đem lại độ chính xác rất cao.
