# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyễn Vũ Quang Minh

Công cụ gán nhãn đã dùng: AnyLabeling / CVAT Docker

---

## 1. Dữ liệu và cách chia tập

### Lý do chia tập theo trục thời gian kèm vùng đệm (buffer)
Tập dữ liệu trong lab được trích xuất từ video YouTube quay đường cao tốc ban đêm cố định từ trên cầu vượt với tần suất 2.5 khung hình/giây (mỗi khung hình cách nhau 0.4 giây). Vì camera đặt cố định và các phương tiện di chuyển với tốc độ hữu hạn, một chiếc xe thường xuất hiện liên tục trong khung hình từ 5 đến 15 giây (tương đương 12 đến 37 frame kế tiếp nhau).

Nếu chia ngẫu nhiên (random splitting), các khung hình liền kề chứa cùng một chiếc xe, cùng góc quay, điều kiện ánh sáng và bối cảnh mặt đường sẽ bị phân tán vào cả tập chưa gán nhãn (pool/train) và tập kiểm thử (test). Khi đó xảy ra hiện tượng **rò rỉ dữ liệu (data leakage)** nghiêm trọng: mô hình không học được đặc trưng tổng quát của lớp `car` trong bóng đêm, mà chỉ đơn thuần "ghi nhớ" (memorize) chính các đối tượng cụ thể mà nó đã được huấn luyện ở khoảnh khắc trước hoặc sau đó vài phần mười giây.

### Hướng lệch của số đo nếu chia ngẫu nhiên
Nếu chia ngẫu nhiên, các số đo hiệu năng trên tập kiểm thử (như AP50, Precision, Recall) sẽ bị **thổi phồng quá mức (optimistically biased - lệch cao giả tạo)**. Mô hình có vẻ đạt độ chính xác rất cao trên giấy tờ, nhưng khi mang sang áp dụng cho một đoạn video mới, một khung giờ khác hay một góc máy khác thì hiệu năng sẽ sụt giảm thảm hại.

### Cách thiết kế tập kiểm thử trong bài lab
Để ngăn chặn hoàn toàn hiện tượng rò rỉ dữ liệu, bài lab đã thiết kế:
- **Tập kiểm thử (test set)** gồm 20 ảnh, trích xuất từ 4 đoạn cách xa nhau có tâm ở các giây 20, 60, 100 và 140 (mỗi đoạn lấy 5 ảnh cách nhau 1.2 giây).
- **Vùng đệm (buffer)** gồm 112 ảnh trong khoảng 4 giây trước và sau mỗi đoạn kiểm thử, cùng các ảnh xen kẽ, bị loại bỏ hoàn toàn khỏi tập pool.
- Nhờ vậy, ảnh trong tập pool gần ảnh kiểm thử nhất vẫn có khoảng cách thời gian ít nhất là **4.4 giây**, đảm bảo toàn bộ các phương tiện trong đoạn test đã hoàn toàn đi ra khỏi góc nhìn của camera trước khi frame pool bắt đầu, bảo đảm tính độc lập khách quan của tập kiểm thử.

---

## 2. Mô hình khởi đầu lạnh (cold start)

### Dòng số đo vòng 0 từ `reports/rounds_table.md`
Trích xuất từ `reports/rounds_table.md` và `outputs/metrics_round0.json`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

### Phân tích không khớp nhãn tham chiếu qua `outputs/compare_round0.jpg`
Mô hình khởi đầu lạnh sử dụng trọng số pretrained COCO (`yolov8n.pt`, gộp các lớp `car`, `bus`, `truck` thành lớp `car`). Trên tập kiểm thử 20 ảnh với 403 box tham chiếu (đã loại bỏ 14 box nhỏ dưới 16px), mô hình đạt Precision rất cao (**0.925**, chỉ 16 FP), nhưng Recall chỉ đạt **0.489** (TP = 197, FN = 206, tức bỏ sót hơn 51% số lượng xe).

Quan sát trực quan từ `outputs/compare_round0.jpg` cho thấy mô hình cold start không khớp với nhãn tham chiếu ở các nhóm xe sau:
1. **Xe ở xa / kích thước nhỏ**: Các xe ở sát đường chân trời hoặc làn xa chỉ hiển thị dưới dạng hai đốm sáng mờ ảo, mô hình COCO hoàn toàn không phát hiện được.
2. **Xe bị che khuất (occluded)**: Xe nối đuôi nhau trên làn bên trái bị xe phía trước hoặc thanh chắn đường che mất một phần thân.
3. **Xe màu tối / xe bị nhoè chuyển động**: Xe có thân xe màu đen tiệp vào nền đường tối, chỉ thấy cụm đèn nhưng không rõ viền thân xe.

### Ý nghĩa của độ phủ (Recall) theo kích thước xe
- **R small = 0.182 (12 / 66 box)**: Cực kỳ thấp. Mô hình bỏ sót tới hơn 81% các xe nhỏ ở xa. Điều này phản ánh rõ hạn chế của mô hình COCO thông thường khi áp dụng vào bài toán camera giám sát ban đêm: các xe nhỏ ban đêm không có đủ đặc trưng hình học rõ ràng (bánh xe, kính, lưới tản nhiệt) mà chủ yếu là cụm đèn pha và viền mờ.
- **R medium = 0.547 (162 / 296 box)**: Độ phủ trung bình, mô hình nhận diện được hơn một nửa các xe ở cự ly vừa khi đèn xe chiếu sáng phần nào thân xe.
- **R large = 0.561 (23 / 41 box)**: Độ phủ cao nhất đối với các xe lớn ở tiền cảnh, nơi thân xe và cấu trúc xe hiện rõ nhất.

### Trường hợp cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai
Theo quy định tại `data/DATA.md`, nhãn tham chiếu trong `data/test/labels/` được tạo tự động bởi một mô hình AI khác và **chưa được con người rà soát thủ công từng box**. Do đó, nhãn tham chiếu không phải là chân lý tuyệt đối.

Một trường hợp điển hình quan sát được trên `outputs/compare_round0.jpg` (ví dụ tại `frame_0250` hoặc `frame_0350`):
- Tại góc phải của `frame_0250`, có những vùng vệt sáng đèn xe phản chiếu loang lổ trên mặt đường ướt/rào chắn bị nhãn tham chiếu khoanh nhầm thành xe (false positive của nhãn tham chiếu). Khi mô hình cold start không dự đoán box tại đó, hệ thống lại ghi nhận đó là một False Negative (FN) của mô hình.
- Ngược lại, ở một số vị trí xe cự ly xa sát mép trên khung hình, mô hình cold start nhận diện đúng hai đốm đèn xe đang di chuyển, nhưng nhãn tham chiếu lại bỏ sót, khiến dự đoán đúng của mô hình bị tính oan thành False Positive (FP).
- Vì vậy, người đánh giá cần rà soát lại trực tiếp trên ảnh gốc: chỉ khi box tương ứng với một chiếc xe 4 bánh thật sự (nhìn thấy thân xe hoặc viền xe quanh cụm đèn theo `GUIDELINE_LABEL.md`) thì mới được kết luận mô hình dự đoán sai.

---

## 3. Chiến lược chọn mẫu

### Giải thích công thức tính điểm và vai trò của các tham số
Công thức chọn mẫu học chủ động:
$$\text{score} = W_U \cdot U + W_A \cdot A + W_D \cdot D$$
với trọng số mặc định $W_U = 0.5$, $W_A = 0.3$, $W_D = 0.2$ (cộng thêm $\text{EMPTY\_BONUS} = 0.5$ nếu frame không có box nào).

- **Hàm bất định từng box $u(c) = 1 - |2c - 1|$**: Khi độ tin cậy $c = 0.5$, $u(c) = 1.0$ đạt cực đại (mô hình hoàn toàn phân vân không biết là xe hay không phải xe). Khi $c$ tiến dần về 0 hoặc tiến dần về 1, $u(c) \to 0$ (mô hình rất tự tin về dự đoán của mình).
- **Thành phần $U$ (Uncertainty)**: Là trung bình cộng của 5 giá trị $u(c)$ lớn nhất trong khung hình (`TOP_N = 5`). Đại diện cho mức độ phân vân của mô hình trên nhóm các đối tượng khó nhất trong frame.
- **Thành phần $A$ (Ambiguity)**: Là tỉ lệ số box có độ tin cậy mập mờ trong khoảng $[0.15, 0.50)$, chuẩn hoá bằng cách chia cho giá trị lớn nhất trong pool. $A$ đo lường mật độ diện rộng của các đối tượng gây nghi ngờ trong ảnh.
- **Thành phần $D$ (Temporal Diversity)**: Là khoảng cách thời gian từ frame đang xét tới frame đã được gán nhãn gần nhất, chia cho trần `DIVERSITY_CAP_S = 10.0` giây (ở vòng 1 chưa có frame nào được gán nên $D = 1.0$ cho tất cả). $D$ thúc đẩy việc chọn các mẫu trải đều theo trục thời gian thay vì dồn vào một thời điểm.
- **Vai trò của `MIN_GAP_S` (2.0 giây)**: Trong thuật toán chọn tham lam (Greedy Selection), `MIN_GAP_S = 2.0s` đóng vai trò là khoảng cách an toàn tối thiểu giữa các frame được chọn trong cùng một lô. Vì camera cố định trên cầu vượt quay ở 2.5 fps, hai khung hình cách nhau dưới 2 giây ghi lại cùng một đoàn xe với vị trí hầu như không thay đổi. Nếu chọn các frame sát nhau, người gán nhãn sẽ phải tốn công sức lặp lại vô ích mà mô hình không tiếp nhận thêm thông tin mới. `MIN_GAP_S` giúp tối ưu hoá ngân sách gán nhãn, loại trừ các mẫu trùng lặp ngữ cảnh.

### Dẫn chứng 3 frame model chọn và 1 frame khác từ `reports/SELECTION.md`
1. **`frame_0182.jpg`** (Rank 1, t = 72.8s, Score = 0.9591): Có $U = 0.9182$, $A = 1.0$ (18 box mập mờ, cao nhất pool). Đây là khung hình giữa video với lưu lượng xe dày đặc, nhiều xe ở cự ly gần và trung bình đang chạy tới tạo ra nhiều vùng bất định cao.
2. **`frame_0312.jpg`** (Rank 7, t = 124.8s, Score = 0.9100): Có $A = 1.0$ (18 box mập mờ). Điểm đặc biệt của frame này là sự xuất hiện của một chiếc **xe container/đầu kéo rơ-moóc lớn màu trắng** ở làn giữa mà model cold start hoàn toàn không nhận diện được, cùng một box dán lệch quá khổ bao trùm làn đường.
3. **`frame_0392.jpg`** (Rank 15, t = 156.8s, Score = 0.8874): Có điểm bất định đỉnh cao nhất ($U = 0.9747$). Khung hình ghi lại cảnh cuối đường với xe buýt lớn và cặp xe bị che khuất (occluded) ở làn trái, đúng với dự đoán trong `BLIND_SCAN.md`.
4. **Frame có điểm cao nhưng bị loại - `frame_0372.jpg`** (Rank 6, t = 148.8s, Score = 0.9101): Frame này có score đứng thứ 6 trong toàn bộ tập pool (cao hơn cả `frame_0312.jpg`), nhưng **bị thuật toán loại bỏ (`selected = False`)** vì cách `frame_0369.jpg` (Rank 2, t = 147.6s) chỉ có **1.2 giây** ($< \text{MIN\_GAP\_S} = 2.0s$). Cả hai frame đều ghi lại cùng một đoàn xe đang lưu thông; việc gạt bỏ `frame_0372.jpg` giúp tiết kiệm chi phí rà nhãn mà vẫn đảm bảo độ bao phủ dữ liệu.

### Điểm bất định có chứng minh ảnh đó sẽ cải thiện mô hình không?
**Hoàn toàn không.** Điểm bất định cao chỉ phản ánh trạng thái thiếu tự tin của mô hình hiện tại, không đồng nghĩa với việc thông tin đó mang lại giá trị gia tăng tích cực cho quá trình huấn luyện:
- **Nhiễu dữ liệu (Aleatoric Uncertainty)**: Bất định có thể xuất phát từ các yếu tố nhiễu không thể khắc phục, ví dụ như vệt sáng đèn pha phản chiếu trên mặt đường ướt, ánh đèn đường nhấp nháy, hoặc các chấm sáng li ti ở tận đường chân trời (< 16px). Việc chọn những ảnh có nhiều nhiễu như vậy có thể khiến mô hình học phải các mẫu giả (spurious correlations).
- **Độ lệch phân phối**: Tập chọn ưu tiên các frame có nhiều box mập mờ (thường là xe nhỏ ở xa) có thể làm chệch phân phối huấn luyện, khiến mô hình bị mất cân bằng và giảm sút độ chính xác trên các nhóm xe thông thường khác.
- **Rủi ro quá thận trọng / quên tri thức (Catastrophic Forgetting)**: Khi fine-tune trên một lô nhỏ (chỉ 12 ảnh) toàn những ca khó và mập mờ, mô hình có xu hướng đẩy ngưỡng tự tin lên cao hoặc trở nên quá thận trọng, dẫn đến việc sụt giảm Recall trên diện rộng như thực tế đã ghi nhận ở Vòng 1.

---

## 4. Các vòng học chủ động (active learning)

### Bảng số đo các vòng từ `reports/rounds_table.md`
Trích xuất từ `reports/rounds_table.md`, `outputs/metrics_round0.json` và `outputs/metrics_round1.json`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 338 | 0.501 | -0.270 | 1.000 | 0.057 | 0.108 | 0.000 | 0.037 | 0.293 |

### Trình bày chi tiết về quá trình sửa nhãn và kết quả
1. **Mức độ sửa nhãn gợi ý (từ `outputs/round1_diff.md` và `reports/REVIEW_LOG.csv`)**:
   - Trong lô 12 ảnh vòng 1, model ban đầu đề xuất **169 box**. Sau khi được rà soát và hiệu chỉnh toàn diện theo `GUIDELINE_LABEL.md`, số lượng box thực tế tăng lên **338 box** (tăng gấp đôi).
   - Các hành động chỉnh sửa chính:
     - **Thêm mới (`added`)**: Bổ sung hàng loạt xe mà AI bỏ sót hoàn toàn: xe container lớn ở `frame_0312`, xe tải thùng lớn ở `frame_0270`, xe buýt lớn và cặp xe occluded ở `frame_0392`, các xe chạy tới ở cự ly gần bật đèn pha sáng rực rọi xuống mặt đường tại `frame_0099`, `frame_0187`, `frame_0270`.
     - **Xoá bỏ (`deleted`)**: Xoá bỏ các box dán sai như vệt sáng phản chiếu đèn đỏ trên dải phân cách ở `frame_0187` (Box 13); xoá các box trùng lặp (duplicate) dán đè lên cùng một xe ở `frame_0107` (Box 12 & 4), `frame_0182` (Box 12 & 8), `frame_0380` (Box 11 & 3), `frame_0392` (Box 9 & 6).
     - **Chỉnh sửa (`edited`)**: Khắc phục hiện tượng AI cắt đôi 1 chiếc xe sedan có 2 đèn hậu đỏ thành 2 box nửa thân ở `frame_0331` (Box 4 & 19); chỉnh lại box dán lệch quá khổ ở `frame_0312` (Box 9); tách và chỉnh lại box gộp xe tải thùng với xe con ở `frame_0227` (Box 10).
2. **Biến động AP50**:
   - AP50 của mô hình sau fine-tune Vòng 1 đạt **0.501**, giảm **-0.270** so với cold start (0.771).
3. **Nhóm xe tốt lên / xấu đi theo số đo trên tập test**:
   - **Tốt lên vượt bậc về độ chuẩn xác (Precision)**: Precision tại ngưỡng conf 0.25 tăng từ 0.925 lên mức **tuyệt đối 1.000** (FP = 0). Mô hình sau khi học nhãn sạch đã hoàn toàn không còn đưa ra bất kỳ dự đoán nhầm lẫn nào đối với các vệt sáng hay vật thể không phải xe.
   - **Xấu đi về độ phủ (Recall)**: Recall tại ngưỡng conf 0.25 giảm sâu từ 0.489 xuống **0.057** (TP giảm từ 197 xuống 23, FN tăng từ 206 lên 380).
     - **Xe nhỏ (small)**: Recall giảm từ 0.182 về **0.000** (hoàn toàn không bắt được xe nhỏ ở ngưỡng conf 0.25).
     - **Xe vừa (medium)**: Recall giảm từ 0.547 về **0.037** (chỉ phát hiện được 11 / 296 box).
     - **Xe lớn (large)**: Duy trì tốt nhất, Recall đạt **0.293** (12 / 41 box).

### Phân tích ca thay đổi qua `outputs/compare_round1.jpg`
Quan sát trên cả 4 frame so sánh (`frame_0050`, `frame_0150`, `frame_0250`, `frame_0350`):
- Ở mô hình cold start (cột giữa), mô hình phát hiện rải rác các xe từ tiền cảnh tới trung cảnh (TP màu xanh lá, FP màu đỏ).
- Sang mô hình Round 1 (cột phải), các box FP màu đỏ hoàn toàn biến mất (FP = 0), nhưng đồng thời hầu hết các box màu xanh lá ở cự ly vừa và xa cũng biến mất, thay bằng viền vàng biểu thị False Negative (FN). Mô hình chỉ còn giữ lại dự đoán với độ tự tin cao ở 1 đến 3 xe lớn nằm sát ống kính camera ở tiền cảnh dưới cùng.
- **Lý do có thể kiểm chứng**:
  1. *Quy mô tập huấn luyện quá nhỏ*: Chỉ với 12 ảnh (338 box) nhưng huấn luyện tới 50 epochs trên nền YOLOv8n với kích thước ảnh lớn (imgsz = 960). Hiện tượng quá khớp (overfitting) trên tập mẫu nhỏ kết hợp với sự thay đổi phân bổ nhãn khiến mô hình dịch chuyển phân phối confidence sang hướng cực kỳ dè dặt / quá thận trọng.
  2. *Tác động của ngưỡng lọc cố định conf = 0.25*: Sau fine-tune, mô hình có thể vẫn định vị được các xe ở xa nhưng gán confidence thấp (trong khoảng 0.10 - 0.20). Bộ lọc cố định tại ngưỡng 0.25 đã gạt bỏ toàn bộ các dự đoán này, làm Recall sụt giảm nghiêm trọng.

### Đối chiếu 3 nguồn bằng chứng: `BLIND_SCAN.md`, `REVIEW_LOG.csv` và kết quả sau train
- **Quan sát độc lập (`reports/BLIND_SCAN.md`)**: Khi quét mù `frame_0392.jpg` trước khi mở nhãn AI, người thực hiện đã ghi nhận: *"Cuối đường, có 1 cặp xe đang đè lên nhau, điểm mờ, không rõ ràng nên dễ bị bỏ sót/vẽ sai. 2 xe occluded, hở 1 nửa phần xe có nhìn rõ đèn pha và kính chắn gió."*
- **Đối chiếu lỗi pre-label (`reports/REVIEW_LOG.csv`)**: Mở pre-label của AI tại `frame_0392.jpg` cho thấy AI đã bỏ sót hoàn toàn cặp xe occluded này (đúng như dự đoán), đồng thời bỏ sót cả chiếc xe buýt lớn màu trắng đỏ và tạo ra box 9 trùng đè lên box 6. Người rà soát đã thêm box cho cặp xe occluded và xoá box trùng.
- **Kết quả mô hình sau train**: Mô hình sau khi fine-tune học được việc triệt tiêu hoàn toàn False Positive (không vẽ nhầm vệt đèn), nhưng do chỉ mới được học trên 12 ảnh, mô hình chưa đủ độ tự tin để khôi phục việc phát hiện các ca khó bị che khuất ở cự ly xa khi đánh giá trên tập test.

### Mô tả một ca khó theo guideline
- **Ca xe sedan bị cắt đôi tại `frame_0331.jpg` (Box 4 và Box 19)**:
  - Tình huống: Model AI ban đầu tạo ra 2 box riêng biệt: Box 19 bao quanh cụm đèn hậu bên trái, Box 4 bao quanh nửa thân xe bên phải.
  - Xử lý theo `GUIDELINE_LABEL.md`: Quy tắc nêu rõ *"Mỗi chiếc xe được đánh dấu bằng một hộp giới hạn... hình chữ nhật, cạnh song song với cạnh ảnh. Không gán nhãn riêng cho từng đèn"*. Người rà nhãn đã xoá bỏ Box 19 và điều chỉnh Box 4 kéo dài ngang để ôm khít trọn vẹn cả hai cụm đèn và thân xe, không tính phần vệt sáng phản chiếu xuống mặt đường.

---

## 5. Kết luận và giới hạn

### Đánh giá kết quả Vòng 1 và quyết định dừng hay tiếp tục
- So với khởi đầu lạnh, mô hình sau Vòng 1 đã đạt được sự cải thiện tuyệt đối về chất lượng nhận diện: **Precision đạt 1.000 (không còn bất kỳ False Positive nào)**, chứng minh nhãn được rà soát đã loại bỏ thành công nhiễu và các dự đoán giả mạo. Tuy nhiên, AP50 giảm từ 0.771 xuống 0.501 do Recall bị suy giảm khi mô hình trở nên quá thận trọng trên tập mẫu nhỏ 12 ảnh.
- **Quyết định: TIẾP TỤC VÒNG 2.** Đúng như phân tích trong `GUIDE.md` và `RUBRIC.md`, việc AP50 giảm ở vòng đầu là hiện tượng bình thường khi mô hình chuyển từ trọng số tổng quát COCO sang thích nghi miền dữ liệu hẹp. Việc dừng lại ở Vòng 1 sẽ để lại một mô hình có độ phủ quá thấp. Cần tiếp tục Vòng 2 với lô ảnh mới trong `to_label/round2/` (đã được trích xuất từ `day8_round1_out`) nhằm bổ sung thêm dữ liệu huấn luyện, giúp mô hình khôi phục độ phủ cho các xe ở cự ly vừa và xa.

### Đề xuất hai ca còn yếu hoặc bất định cho vòng sau (từ `outputs/selection_round2.csv`)
1. **Ca xe kích thước nhỏ ở cự ly xa trên nền đường tối**:
   - Đề xuất các frame như **`frame_0073.jpg`** (Rank 1, t = 29.2s, Score = 0.7206, D = 1.0) hoặc **`frame_0123.jpg`** (Rank 2, t = 49.2s, U = 0.7716).
   - *Lý do*: Bù đắp trực tiếp cho điểm yếu lớn nhất hiện tại là Recall small = 0.000.
   - *Chi phí rà nhãn và nguy cơ*: Chi phí rà nhãn cao do mỗi frame có rất nhiều chấm đèn nhỏ ở xa, đòi hỏi người gán phải zoom lớn để vẽ box khít thân xe. Nguy cơ ảnh gần trùng được kiểm soát nhờ thành phần $D$ và `MIN_GAP_S` khi `frame_0073` cách frame đã gán gần nhất tới 10.4s ($D = 1.0$).
2. **Ca các phương tiện lớn / xe buýt / xe tải bị che khuất hoặc lưu thông ở làn ngoài**:
   - Đề xuất các frame như **`frame_0387.jpg`** (Rank 3, t = 154.8s, Score = 0.7074, A = 1.0) hoặc **`frame_0169.jpg`** (Rank 4, t = 67.6s, Score = 0.6574).
   - *Lý do*: Giúp mô hình củng cố khả năng nhận diện các phương tiện thương mại cỡ lớn nhiều bánh theo đúng phạm vi lớp `car`.
   - *Chi phí rà nhãn và nguy cơ*: Chi phí rà nhãn ở mức trung bình vì số lượng xe lớn ít hơn, nhưng cần lưu ý ranh giới giữa đầu xe và rơ-moóc để tránh vẽ nhầm thành 2 box hoặc dán lệch. Cần kiểm tra kỹ `MIN_GAP_S` vì `frame_0387` khá gần cụm frame cuối video ($D = 0.20$).

### Ảnh hưởng của các giới hạn thực nghiệm đến kết luận
1. **Quy mô tập kiểm thử nhỏ (20 ảnh)**: Với chỉ 20 ảnh kiểm thử, mỗi ảnh đóng góp tới 5% trọng số. Sự thay đổi chỉ 1-2 box trên một ảnh có thể gây biến động đáng kể đến AP50. Do đó, biến động số đo giữa các vòng không thể coi là bằng chứng tuyệt đối duy nhất mà phải kết hợp chặt chẽ với quan sát trực quan trên ảnh kiểm thử.
2. **Luật bỏ qua xe cao dưới 16 pixel**: Có 14 box tham chiếu nhỏ dưới 16px được bỏ qua khi chấm điểm. Quy tắc này giúp loại bớt sự tranh cãi ở vùng chân trời mờ mịt, nhưng cũng có nghĩa là số đo AP50 không đánh giá khả năng phát hiện xe ở cự ly cực xa.
3. **Nhãn tham chiếu do mô hình AI tạo ra chưa qua rà tay**: Vì nhãn test được sinh bởi một mô hình khác, nó chứa sẵn các sai số hệ thống (bỏ sót xe tối, bắt nhầm vệt đèn). Do đó, điểm AP50 thực chất là độ tương đồng với mô hình tạo nhãn tham chiếu chứ không hoàn toàn tương đương với độ chính xác theo chuẩn con người.

### Các yếu tố cần kiểm tra trước khi huấn luyện thêm nếu AP50 giảm
Nếu AP50 tiếp tục giảm ở vòng tiếp theo, trước khi vội vã gán thêm dữ liệu hoặc thay đổi cấu trúc, cần kiểm tra 3 điểm trọng yếu:
1. **Kiểm tra phân phối Confidence Score và đường cong Precision-Recall**: Vẽ biểu đồ F1-Confidence curve để xác định xem ngưỡng conf tối ưu có bị dịch chuyển xuống thấp (ví dụ conf ~ 0.10 - 0.15) hay không, thay vì chỉ đánh giá cứng tại mốc conf = 0.25.
2. **Kiểm tra tính nhất quán và chất lượng nhãn đã gán**: Mở lại toàn bộ các file nhãn trong `labels/round1/` và `labels/round2/`, rà soát xem có xuất hiện box nào vi phạm quy tắc (như ôm vệt phản chiếu đèn, box cắt cụt thân xe, hoặc box trùng lặp) khiến mô hình học phải tín hiệu nhiễu hay không.
3. **Kiểm tra chiến lược siêu tham số (Hyperparameters)**: Giảm số lượng epochs huấn luyện (ví dụ từ 50 xuống 20-30 epochs) hoặc tinh chỉnh learning rate / weight decay để chống hiện tượng overfitting trên các lô dữ liệu có số lượng ảnh nhỏ.
