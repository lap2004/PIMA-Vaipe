# Mô hình 2: Thử nghiệm Cắt tỉa (PIMA Ablation Study)

Thư mục `pima_ablation_study` (thuộc hệ thống PIMA-VAIPE) chứa mã nguồn phục vụ cho phương pháp **Ablation Study (Nghiên cứu Cắt tỉa/Thử nghiệm thành phần)**. 

Trong nghiên cứu Deep Learning, Ablation Study là một bước cực kỳ quan trọng để chứng minh tính hiệu quả của hệ thống. Chúng tôi tinh chỉnh, thay đổi hoặc loại bỏ một số module lõi của phương pháp PIMA gốc (ví dụ: đổi hàm Loss, thay đổi cách kết hợp đồ thị) để quan sát xem hiệu năng thay đổi như thế nào trên bộ dữ liệu **VAIPE**.

---

## 🔬 Các Kiến trúc được Thử nghiệm (Ablations)

Trong phiên bản này, các công nghệ nền tảng phần lớn vẫn giống với PIMA Baseline (`ResNet50` cho hình ảnh, `GraphSAGE` cho đồ thị văn bản). Tuy nhiên, có một số cấu hình đã được thay đổi để đo lường:

### 1. Thử nghiệm thay đổi Hàm mất mát (Loss Function Variations)
- Hệ thống thử nghiệm tinh chỉnh các trọng số của **Contrastive Loss** và **BCE/NLL Loss** để tìm ra điểm cân bằng tốt nhất giữa việc phân loại chữ (Text Node) và việc ghép cặp ảnh-chữ (Image-Text Matching).
- Các mẫu (samples) khi thực hiện inference được lọc với ngưỡng softmax nghiêm ngặt (ngưỡng 0.8) để đánh giá khả năng mô hình tự tin trên các đặc trưng đã học.

### 2. Thử nghiệm cấu trúc Đồ thị (Graph Variations)
- Chặn/bỏ qua (bypass) một phần thuật toán GraphSAGE để so sánh trực tiếp: Liệu việc ghép cặp thuần túy (Text thuần - Image thuần) có tốt hơn là dùng Đồ thị hay không? Qua đó chứng minh vai trò thiết yếu của GNN.

---

## ⚙️ Cấu Trúc Mã Nguồn

- `train_ablation.py`: Kịch bản huấn luyện chính trong thư mục Ablation, tích hợp việc kết hợp đặc trưng đồ thị và ảnh, sử dụng `ContrastiveLoss` trong việc đo lường độ tương đồng của ảnh viên thuốc và chữ viết. Mô hình tốt nhất được lưu tại `./logs/weights/model_best.pth`.
- `eval.py`: Tải trọng số tốt nhất sau huấn luyện và tính toán độ chính xác Top-1 Matching trên tập dữ liệu Test.
- `config.py`: File cấu hình chung lưu tham số.
- Thư mục `models` và `utils`: Chứa mô hình `PrescriptionPill` đã được điều chỉnh riêng biệt cho nghiên cứu cắt tỉa, cùng các hàm tiện ích như Data Loaders và Metrics.

---

## 📊 Phân tích Thực nghiệm (Pros & Cons)

- **Vai trò trong nghiên cứu:** Bước đệm hoàn hảo để rút ra những điểm yếu cốt lõi của kiến trúc cũ. Nhờ Ablation Study, chúng tôi nhận ra rằng chỉ tinh chỉnh nhỏ (đổi Loss, đổi Data Augmentation) là **không đủ** để vượt qua các rào cản đặc thù của bộ dữ liệu VAIPE.
- **Tiền đề phát triển:** Kết quả hạn chế từ thư mục này chính là lý do và động lực bắt buộc phải đập đi xây lại toàn bộ kiến trúc (dẫn đến sự ra đời của phiên bản `PIMA_NEW` hoàn thiện nhất).

---

## 📈 Kết Quả Đánh Giá & Thực Nghiệm

Dưới đây là các thông số chi tiết được ghi nhận lại trong quá trình huấn luyện mô hình Ablation nhằm đo lường sự thay đổi hiệu năng:

| Mô Hình | Epoch Dừng Sớm | Loss Huấn Luyện | Độ Chính Xác (Val Accuracy) | Khớp Chính Xác (Top-1 Matching) |
| :--- | :---: | :---: | :---: | :---: |
| **Mô hình Ablation** | Epoch 9 | 0.862 | 50.52% | **26.23%** |

### Chi Tiết Quá Trình Huấn Luyện
- **File log ghi nhận huấn luyện:** `ablation.log`
- **File log ghi nhận đánh giá (Evaluation):** `log_eval.log`
- **Số Epoch huấn luyện đạt được:** 9 (Mô hình hội tụ khá nhanh và kích hoạt Early Stopping ở epoch 9)
- **Train Loss (Total Loss):** 0.862
- **Validation Accuracy (trong lúc train):** 50.52%
- **Top-1 Matching Accuracy (trên tập Test):** 26.23%

### Nhận Xét Tổng Quan

- **Về Train Loss:** Mức loss đạt được là `0.862`, cho thấy quá trình huấn luyện của kiến trúc mạng bị cắt tỉa gặp khó khăn nhất định so với phiên bản PIMA gốc. 
- **Về Độ Chính Xác:** Mặc dù **Validation Accuracy** trên tập xác thực trong quá trình train đạt mức khá tốt (trên 50%), nhưng khi thực hiện ghép cặp thực tế trên tập kiểm thử (Test set), **Top-1 Matching Accuracy** lại bị giảm sâu, chỉ dừng lại ở **26.23%**. Sự chênh lệch này cho thấy mô hình dễ bị quá khớp (overfitting) hoặc gặp khó khăn khi tổng quát hóa lên dữ liệu ghép cặp thực tế, một phần có thể do việc lọc ngưỡng softmax quá cứng nhắc hoặc do bị loại bỏ các đặc trưng đồ thị thiết yếu.
- **Kết luận từ Ablation Study:** Việc thay đổi cấu trúc đồ thị hoặc hàm loss trong thử nghiệm Ablation không đem lại hiệu năng lý tưởng như kỳ vọng. Kết quả này là bằng chứng thực nghiệm quan trọng, giúp phát hiện điểm yếu cốt lõi trong quy trình trích xuất và kết hợp đặc trưng. Nhờ đó, chúng ta có đủ căn cứ để nâng cấp toàn diện sang một kiến trúc ưu việt hơn (phiên bản `PIMA_NEW`).