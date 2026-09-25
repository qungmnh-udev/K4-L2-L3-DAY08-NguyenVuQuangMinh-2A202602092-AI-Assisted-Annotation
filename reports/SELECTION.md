# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, nếu chỉ có ngân sách rà 5 ảnh, tôi ưu tiên đề xuất 5 frame sau nhằm tối đa hoá độ bất định đồng thời đảm bảo độ phân tán thời gian (tránh lãng phí công rà nhãn vào các cảnh gần trùng):

1. **`frame_0182.jpg`** (Rank 1, t = 72.8s, Score = 0.9591, U = 0.9182, A = 1.0, D = 1.0): Điểm cao nhất toàn tập pool, có tới 18 box mập mờ (A = 1.0) và 28 dự đoán; đại diện cho đoạn giữa video với mật độ xe đông đúc và nhiều xe bị che khuất một phần.
2. **`frame_0369.jpg`** (Rank 2, t = 147.6s, Score = 0.9324, U = 0.9315, A = 0.8889, D = 1.0): Điểm bất định U rất cao (0.9315), 43 dự đoán với 16 box mập mờ; đặc trưng cho đoạn cuối video khi xe nối đuôi dồn ứ ở làn ngược chiều.
3. **`frame_0380.jpg`** (Rank 3, t = 152.0s, Score = 0.9170, U = 0.9340, A = 0.8333, D = 1.0): Cách `frame_0369.jpg` 4.4 giây (lớn hơn `MIN_GAP_S = 2.0s`), có U cao nhất trong top 5 (0.9340), có xe tải lớn màu trắng ở làn phải mà model chưa phân định rõ.
4. **`frame_0326.jpg`** (Rank 4, t = 130.4s, Score = 0.9155, U = 0.9310, A = 0.8333, D = 1.0): Đại diện cho cụm thời gian ~130s với 39 box, độ bất định cao ở các xe chạy làn giữa.
5. **`frame_0099.jpg`** (Rank 8, t = 39.6s, Score = 0.9063, U = 0.9460, A = 0.7778, D = 1.0): Quyết định không chọn rank 5 (`frame_0331.jpg`, t = 132.4s) và rank 7 (`frame_0312.jpg`, t = 124.8s) vì chúng quá gần `frame_0326.jpg` (chỉ cách 2.0s và 5.6s, thuộc cùng một đợt lưu thông). Thay vào đó, chọn `frame_0099.jpg` ở t = 39.6s giúp phân bổ đều dữ liệu về đoạn đầu video, mang lại độ đa dạng ngữ cảnh cao với U vượt trội (0.9460).

---

### Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet:

1. **`frame_0182.jpg`** (Rank 1, Score = 0.9591): Trên CSV có A = 1.0 (18 box mập mờ, cao nhất pool). Trên contact sheet `selection_round1.jpg`, frame này có mật độ xe dày đặc ở cả 2 hướng, model phân vân giữa việc nhận diện đèn xe hay thân xe bị nhoè ở mép dưới bên phải.
2. **`frame_0312.jpg`** (Rank 7, Score = 0.9100, t = 124.8s): CSV ghi nhận A = 1.0 (18 box mập mờ). Contact sheet cho thấy có một xe container kéo rơ-moóc lớn màu trắng ở làn giữa mà model khởi đầu lạnh hoàn toàn bỏ sót hoặc cho confidence rất thấp, cùng box 9 dán lệch quá khổ bao trùm làn đường.
3. **`frame_0392.jpg`** (Rank 15, Score = 0.8874, t = 156.8s): CSV ghi nhận điểm bất định U cao kỷ lục (U = 0.9747, trung bình 5 box khó nhất có conf cực sát 0.5). Contact sheet thể hiện rõ cảnh cuối đường với xe buýt lớn màu trắng đỏ và cặp xe bị che khuất (occluded) ở làn trái khớp chính xác với dự đoán tại `BLIND_SCAN.md`.

---

### Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:

- **Frame điểm cao bị loại - `frame_0372.jpg`** (Rank 6, Score = 0.9101, t = 148.8s): Mặc dù nằm trong top 6 với score cao hơn cả `frame_0312.jpg` và `frame_0099.jpg`, frame này **bị thuật toán loại bỏ (selected = False)** vì cách `frame_0369.jpg` (t = 147.6s, Rank 2) chỉ đúng 1.2 giây, vi phạm ngưỡng `MIN_GAP_S = 2.0s`. Hai frame cách nhau 1.2s có cùng một đoàn xe di chuyển gần như giống hệt nhau, việc gán nhãn cả hai sẽ gây lãng phí ngân sách và trùng lặp thông tin huấn luyện.
- **Frame điểm thấp vẫn nên xem - `frame_0026.jpg`** (Rank 48, Score = 0.8215, t = 10.4s) hoặc các frame đầu video: Mặc dù score thấp hơn do ít box mập mờ (A = 0.7778), nhưng đây là đoạn đường thông thoáng đầu video với góc chiếu đèn pha phản xạ mạnh trên mặt đường ướt/bóng. Model có thể tự tin sai (overconfident) với các vệt sáng phản chiếu, cần người rà soát để tránh false positive.

---

### Điều phép chọn này chưa chứng minh về chất lượng mô hình:

Điểm bất định (Uncertainty Score) chỉ đo lường sự phân vân của mô hình hiện tại (confidence gần ngưỡng 0.5 hoặc nhiều box mập mờ), **hoàn toàn không chứng minh rằng việc gán nhãn ảnh đó chắc chắn sẽ cải thiện chất lượng mô hình sau fine-tune**. Lý do:
1. **Nhiễu và nhãn khó (Aleatoric Uncertainty)**: Một frame có điểm bất định cao có thể chứa các đối tượng cực kỳ mờ, vệt sáng phản chiếu gây nhiễu, hoặc các xe quá nhỏ ở đường chân trời (< 16px). Việc nạp các mẫu này vào tập train nhỏ có thể làm mô hình học phải nhiễu (noise) thay vì học đặc trưng tổng quát.
2. **Mất cân bằng kích thước đối tượng**: Tập chọn có xu hướng ưu tiên frame đông xe nhỏ ở xa (nhiều box conf thấp), dẫn đến việc fine-tune không tối ưu cho xe cự ly gần hoặc xe tải lớn.
3. **Hiện tượng catastrophic forgetting / over-cautiousness**: Với tập train nhỏ (12 ảnh), fine-tune có thể khiến model dịch chuyển phân phối dự đoán về phía quá thận trọng (đẩy threshold thực tế lên cao, giảm Recall nghiêm trọng như đã thấy ở Round 1).
