# GIÁO ÁN CHI TIẾT KHÓA HỌC: PHÂN TÍCH DỮ LIỆU & MACHINE LEARNING VỚI PYTHON

* **Thời lượng:** 3 buổi × 3 giờ (Tổng: 9 giờ)
* **Hình thức học tập:** 100% thực hành trực tiếp trên Jupyter Notebook / Google Colab xuyên suốt một bộ dữ liệu duy nhất từ đầu đến cuối.
* **Bộ thư viện chuẩn:** `numpy`, `pandas`, `matplotlib`, `scikit-learn` (kèm `seaborn` bổ trợ trực quan hóa).
* **Đối tượng:** Sinh viên năm nhất ngành Khoa học Dữ liệu & Trí tuệ Nhân tạo (đã có kiến thức lập trình Python cơ bản).
* **Bộ dữ liệu thực hành:** *Student Performance and Attendance Dataset* (Marvy Ayman Halim - Kaggle), gồm 5 tệp CSV:
  1. `students.csv`: Thông tin nhân khẩu học sinh viên.
  2. `attendance.csv`: Nhật ký điểm danh chuyên cần theo ngày.
  3. `homework.csv`: Tỷ lệ hoàn thành bài tập về nhà theo từng buổi/môn.
  4. `performance.csv`: Kết quả điểm thi và đánh giá học tập.
  5. `teacher_parent_communication.csv`: Ghi chép tương tác giữa nhà trường và phụ huynh.

---

## 1. BỐI CẢNH DỰ ÁN & VAI TRÒ MÔ PHỎNG

* **Vai trò đóng giả (Role-play):** Sinh viên đóng vai là **Junior Data Analyst** thuộc Ban Đảm bảo Chất lượng Đào tạo của Nhà trường.
* **Nhiệm vụ xuyên suốt 3 buổi học:**
  * **Buổi 1:** Tiếp nhận 5 tệp dữ liệu học vụ thô rời rạc, kiểm tra chất lượng, làm sạch chuỗi bằng Regex, xử lý trùng lặp/dữ liệu thiếu và hợp nhất thành một bảng dữ liệu tổng hợp chuẩn (`historical_students.csv` và `new_students.csv`).
  * **Buổi 2:** Thực hiện thống kê mô tả, tư duy đặt câu hỏi phân tích và trực quan hóa dữ liệu bằng `Matplotlib` (và `Seaborn`) để tìm ra các yếu tố ảnh hưởng đến kết quả học tập.
  * **Buổi 3:** Xây dựng, đánh giá và diễn giải 2 mô hình học máy cốt lõi: **Linear Regression** (dự báo điểm số cụ thể) và **K-Nearest Neighbors (KNN)** (cảnh báo sớm sinh viên có nguy cơ trượt môn).

---

## 2. TIMELINE VÀ NỘI DUNG CHI TIẾT THEO TỪNG BUỔI

### BUỔI 1: CHUẨN BỊ DỮ LIỆU, REGEX, LÀM SẠCH & HỢP NHẤT 5 FILE (180 PHÚT)

**Mục tiêu cốt lõi:**
* Nắm vững 6 giai đoạn của một quy trình phân tích dữ liệu chuẩn (từ xác định bài toán đến báo cáo kết quả).
* Thành thạo thao tác mảng `NumPy` và các hàm kiểm tra, làm sạch đặc thù của `Pandas`.
* Ứng dụng thành thạo **Biểu thức chính quy (Regex)** để chuẩn hóa dữ liệu chuỗi bị nhiễu.
* Phân biệt rõ ràng giữa *dữ liệu lỗi/thiếu cần xử lý* với *dữ liệu đối tượng mới chưa có nhãn mục tiêu* để tách tập dự báo riêng.

**Timeline chi tiết:**

* **00:00 – 00:30 (30 phút) | Quy trình phân tích dữ liệu, Tải dữ liệu & Ôn tập NumPy**
  * Tổng quan 6 bước trong một dự án Data Analysis thực tế.
  * Hướng dẫn tải tự động bộ dữ liệu từ Kaggle về thư mục `./data` bằng `kagglehub` (không cần API key):
    ```python
    import os, shutil, kagglehub
    cache_path = kagglehub.dataset_download("marvyaymanhalim/student-performance-and-attendance-dataset")
    shutil.copytree(cache_path, "./data", dirs_exist_ok=True)
    ```
  * Ôn nhanh `NumPy`: Khởi tạo mảng 1D/2D (`np.array`, `.reshape()`), tính toán vector hóa, lọc điều kiện (boolean indexing), nhận diện và xử lý `np.nan`.

* **00:30 – 01:10 (40 phút) | Đọc dữ liệu & Kiểm tra chất lượng (Data Auditing) trên 5 file**
  * Sử dụng `pd.read_csv(..., na_values=['NA', '?', 'null', ''])` để nhận diện đa dạng ký hiệu khuyết thiếu ngay từ bước nạp file.
  * Thực hành bộ lệnh kiểm tra tổng quan cấu trúc trên cả 5 bảng: `.head()`, `.shape`, `.dtypes`, `.info()`, `.isna().sum()`.
  * Phát hiện bản ghi trùng lặp với `.duplicated()` và `.duplicated(subset=[...])`.
  * Rà soát các "bẫy" dữ liệu thực tế: trùng mã sinh viên/ngày điểm danh, định dạng chuỗi lộn xộn, ký tự đơn vị dính liền số (`85%`), điểm số vượt ngưỡng `[0, 100]`, và các sinh viên khóa mới chưa có điểm tổng kết (`NaN` / `?`).

* **01:10 – 01:55 (45 phút) | Làm sạch dữ liệu chuỗi nâng cao với Regex trong Pandas**
  * Hệ thống các ký tự đại diện Regex phổ biến: `\d` (chữ số), `\w` (ký tự chữ/số), `\s` (khoảng trắng), `+`, `*`, `^`, `$`, `[]`.
  * Thực hành trực tiếp trên các cột dữ liệu lỗi bằng `.str.replace(..., regex=True)` và `.str.extract()`:
    * **File `homework.csv`:** Tách phần giá trị số ra khỏi chuỗi dính ký tự `%`, khoảng trắng thừa hoặc tiền tố chữ (ví dụ: `" 85% "`, `"score: 90"` $\rightarrow$ ép kiểu `float`).
    * **File `students.csv`:** Chuẩn hóa định dạng `Student_ID`, loại bỏ ký tự đặc biệt và gộp nhiều khoảng trắng liên tiếp (`\s+`) thành một dấu cách chuẩn cho họ tên.
    * **File `attendance.csv`:** Chuẩn hóa nhãn trạng thái chuyên cần về 3 nhóm chuẩn (`Present`, `Absent`, `Late`).

* **01:55 – 02:35 (40 phút) | Xử lý trùng lặp, giá trị thiếu, ngoại lai & Hợp nhất bảng**
  * Loại bỏ bản ghi trùng lặp giữ lại dòng đầu tiên: `.drop_duplicates(subset=[...], keep='first')`.
  * Xử lý ngoại lai (dùng `.clip(0, 100)` cho điểm số/tỷ lệ) và xử lý missing values ở biến đầu vào (`.dropna()`, `.fillna()`).
  * **Nguyên tắc tư duy quan trọng:** Không điền giá trị trung bình (mean) vào các bản ghi chưa có nhãn mục tiêu ở tương lai/đối tượng mới; tách riêng nhóm này thành bảng `new_students` để phục vụ dự báo ở Buổi 3.
  * Dùng `.groupby('Student_ID').agg(...)` tổng hợp các chỉ số từ bảng nhật ký: tỷ lệ chuyên cần (`attendance_rate`), điểm bài tập trung bình (`avg_hw`), số lần tương tác phụ huynh (`comm_count`).
  * Dùng `pd.merge()` ghép các bảng theo `Student_ID` và sắp xếp dữ liệu tăng dần theo mã sinh viên/thời gian bằng `.sort_values()`.

* **02:35 – 03:00 (25 phút) | Tạo biến mục tiêu, lưu dữ liệu sạch & Q&A Buổi 1**
  * Chuẩn bị 2 biến mục tiêu cho Buổi 3:
    * Biến liên tục (Hồi quy): `final_score` (lấy từ điểm thi tổng kết trong `performance.csv`).
    * Biến phân loại (KNN): Dùng cột xếp loại sẵn có hoặc quy đổi nhị phân `is_at_risk` (`1` = nguy cơ trượt môn, `0` = an toàn).
  * Xuất 2 tệp sạch ra ổ đĩa: `historical_students.csv` (dữ liệu lịch sử đầy đủ) và `new_students.csv` (nhóm sinh viên mới cần dự báo).

---

### BUỔI 2: THỐNG KÊ MÔ TẢ, TƯ DUY ĐẶT CÂU HỎI & TRỰC QUAN HÓA (180 PHÚT)

**Mục tiêu cốt lõi:**
* Biết cách chuyển hóa vấn đề nghiệp vụ thành các câu hỏi phân tích dữ liệu cụ thể.
* Thành thạo các hàm thống kê mô tả và phân tích gom nhóm bằng `Pandas` và `NumPy`.
* Viết thành thạo cú pháp vẽ biểu đồ bằng **`Matplotlib` thuần** (đầy đủ tiêu đề, nhãn trục, chú giải, ký hiệu/màu sắc phân lớp) và kết hợp `Seaborn` để mở rộng góc nhìn đa biến.

**Timeline chi tiết:**

* **00:00 – 00:35 (35 phút) | Thống kê mô tả & Phân tích gom nhóm (Descriptive Statistics)**
  * Nạp tệp `historical_students.csv` đã làm sạch ở Buổi 1.
  * Tính toán và đối chiếu các đại lượng thống kê cơ bản bằng cả `Pandas` và `NumPy`:
    * Trung bình (`mean`), Trung vị (`median`), Giá trị nhỏ nhất (`min`), Giá trị lớn nhất (`max`), Độ lệch chuẩn (`std`).
  * Thống kê tần suất các lớp phân loại bằng `.value_counts()` và so sánh các chỉ số hành vi giữa các nhóm sinh viên bằng `.groupby('is_at_risk')[['attendance_rate', 'avg_hw', 'final_score']].mean()`.

* **00:35 – 01:05 (30 phút) | Tư duy đặt câu hỏi phân tích dữ liệu (Analytical Questioning)**
  * Hướng dẫn sinh viên xây dựng 4 nhóm câu hỏi khám phá dữ liệu:
    1. *Nhóm Xu hướng (Trend):* Tỷ lệ chuyên cần hoặc điểm trung bình qua từng tuần/giai đoạn học tập biến động tăng hay giảm?
    2. *Nhóm Tương quan (Relationship):* Tỷ lệ đi học và hoàn thành bài tập về nhà tác động như thế nào đến điểm thi cuối kỳ?
    3. *Nhóm Phân phối (Distribution):* Phổ điểm của toàn khóa tập trung ở khoảng nào, có xuất hiện điểm ngoại lai không?
    4. *Nhóm Phân tách lớp (Class Separation):* Nhóm sinh viên an toàn (`0`) và nhóm nguy cơ (`1`) phân bố và tách biệt ra sao trên mặt phẳng đặc trưng?

* **01:05 – 02:15 (70 phút) | Thực hành trực quan hóa trọng tâm với Matplotlib**
  * Rèn luyện cú pháp chuẩn của `matplotlib.pyplot` cho 4 dạng biểu đồ nền tảng (bắt buộc đầy đủ `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.legend()`, `plt.grid()`):
    1. **Biểu đồ đường (`plt.plot`):** Theo dõi diễn biến tỷ lệ chuyên cần / điểm trung bình qua các tuần học (tùy chỉnh `marker='o'`, `linestyle='-'`, `color`).
    2. **Biểu đồ phân tán (`plt.scatter`):** Vẽ mối quan hệ giữa `attendance_rate` (trục X) và `avg_hw` (trục Y), tách riêng 2 lệnh vẽ cho 2 nhóm `is_at_risk = 0` và `1` với màu sắc (`color`) và ký hiệu (`marker`) khác nhau để quan sát vùng ranh giới.
    3. **Biểu đồ cột (`plt.bar`):** So sánh điểm trung bình và tỷ lệ nộp bài tập giữa các khối lớp / nhóm học lực.
    4. **Biểu đồ tần suất & hộp (`plt.hist`, `plt.boxplot`):** Quan sát hình dạng phân phối điểm thi và phát hiện các trường hợp cá biệt.

* **02:15 – 02:45 (30 phút) | Bổ trợ trực quan hóa thống kê với Seaborn**
  * Sử dụng `Seaborn` để trực quan hóa nhanh mối quan hệ đa biến:
    * `sns.regplot()`: Vẽ biểu đồ phân tán kèm đường xu hướng hồi quy tuyến tính giữa chuyên cần và điểm thi.
    * `sns.heatmap(df.corr(), annot=True)`: Vẽ ma trận tương quan để lựa chọn các biến đầu vào có ý nghĩa nhất cho mô hình ở Buổi 3.

* **02:45 – 03:00 (15 phút) | Rút trích Insight & Tổng kết Buổi 2**
  * Sinh viên viết đoạn nhận xét ngắn gọn (2–3 ý chính) ngay dưới mỗi biểu đồ trên Notebook để rèn kỹ năng đọc biểu đồ và diễn giải xu hướng.

---

### BUỔI 3: XÂY DỰNG & ĐÁNH GIÁ MÔ HÌNH LINEAR REGRESSION & KNN (180 PHÚT)

**Mục tiêu cốt lõi:**
* Nắm vững quy trình chuẩn bị dữ liệu cho `scikit-learn`: định dạng mảng 2D cho biến đầu vào `X`, chiến lược chia tập dữ liệu (ngẫu nhiên vs tuần tự theo thời gian), chuẩn hóa đặc trưng với `StandardScaler` và nguyên tắc chống **Data Leakage**.
* Hiểu rõ bản chất toán học trực quan, tham số và cú pháp code của **Linear Regression** và **K-Nearest Neighbors (KNN)**.
* Biết cách viết công thức, tính toán, đọc hiểu và diễn giải đầy đủ các chỉ số đánh giá cho cả hai bài toán Hồi quy và Phân loại.

**Timeline chi tiết:**

* **00:00 – 00:35 (35 phút) | Chuẩn bị dữ liệu cho Scikit-learn & Chống Data Leakage**
  * Tách tập biến độc lập `X` (bắt buộc ở dạng DataFrame hoặc mảng 2 chiều `[[]]`) và biến mục tiêu `y` (mảng 1 chiều).
  * **Chiến lược chia tập Train / Test:**
    * Thực hành `train_test_split(X, y, test_size=0.2, random_state=42)` cho bài toán phân loại dữ liệu độc lập.
    * Phân tích & thực hành chia tuần tự theo thời gian (lấy giai đoạn đầu làm `train`, giai đoạn sau làm `test`) đối với dữ liệu chuỗi thời gian để tránh dùng tương lai dự đoán quá khứ.
  * **Chuẩn hóa dữ liệu (`StandardScaler`) & Chống Data Leakage:** Giải thích hiện tượng rò rỉ dữ liệu; thực hành quy tắc chuẩn: chỉ gọi `scaler.fit_transform(X_train)` trên tập huấn luyện và `scaler.transform(X_test)` trên tập kiểm tra.

* **00:35 – 01:25 (50 phút) | Thuật toán 1: Hồi quy tuyến tính (`LinearRegression`)**
  * **Bản chất & Toán học:** Phương trình đường thẳng $\hat{y} = ax + b$ (và mở rộng đa biến $\hat{y} = w_1x_1 + w_2x_2 + b$), nguyên lý Bình phương tối thiểu (OLS). Giải thích ý nghĩa thực tiễn của hệ số góc (`model.coef_`) và hệ số chặn (`model.intercept_`).
  * **Thực hành Scikit-learn:**
    * Khởi tạo `LinearRegression()`, huấn luyện `.fit(X_train, y_train)`, in và diễn giải `coef_` và `intercept_`.
    * Dự đoán trên tập kiểm tra `.predict(X_test)` và dự báo điểm cuối kỳ cho nhóm `new_students` (đảm bảo truyền đầu vào dạng 2D).
  * **Đánh giá mô hình Hồi quy (`sklearn.metrics`):**
    * Viết công thức toán và tính bằng hàm `mean_absolute_error(y_test, y_pred)`, `mean_squared_error(y_test, y_pred)`, `r2_score(y_test, y_pred)`.
    * Vẽ biểu đồ `plt.scatter(y_test, y_pred)` đối chiếu giá trị thực tế và giá trị dự đoán.

* **01:25 – 02:20 (55 phút) | Thuật toán 2: K-Nearest Neighbors (`KNeighborsClassifier`)**
  * **Bản chất & Toán học:**
    * Công thức khoảng cách Euclid giữa hai điểm dữ liệu: $d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$.
    * Cơ chế bầu chọn theo đa số (Majority Voting) và vai trò của tham số $k$ (chọn $k$ lẻ để tránh hòa phiếu; tác động gây underfitting/overfitting khi thay đổi $k$).
    * **Thí nghiệm kiểm chứng trên Notebook:** So sánh kết quả dự đoán của KNN trước và sau khi dùng `StandardScaler` để thấy rõ tính nhạy cảm với thang đo (scale sensitivity) của thuật toán dựa trên khoảng cách.
  * **Thực hành Scikit-learn:**
    * Khởi tạo `KNeighborsClassifier(n_neighbors=3)` (thử nghiệm thêm với $k = 3, 5, 7$).
    * Huấn luyện `.fit()`, dự đoán `.predict(X_test)` và phân loại cảnh báo nguy cơ cho nhóm `new_students`.
  * **Đánh giá mô hình Phân loại (`sklearn.metrics`):**
    * Định nghĩa 4 thành phần `TP, TN, FP, FN` (với lớp `1` là Positive) và xuất `confusion_matrix(y_test, y_pred)`.
    * Viết công thức và tính `accuracy_score`, `precision_score`, `recall_score`, `f1_score`.
    * Phân tích các trường hợp mô hình phân loại nhầm (False Positive / False Negative) dựa trên phân bố dữ liệu và ý nghĩa thực tế trong quản lý đào tạo.

* **02:20 – 03:00 (40 phút) | Diễn giải kết quả, Hạn chế mô hình & Tổng kết khóa học**
  * Hướng dẫn sinh viên viết đoạn kết luận tổng hợp (5–7 dòng) tóm tắt: xu hướng chính của dữ liệu, kết quả dự báo cho nhóm sinh viên mới, và mức độ tin cậy của dự báo dựa trên các chỉ số đánh giá.
  * Thảo luận các hạn chế của hai thuật toán (quy mô mẫu nhỏ, giả định tuyến tính cứng nhắc của Linear Regression, độ nhạy với thang đo và nhiễu của KNN) và đề xuất thêm các đặc trưng mới giúp nâng cao chất lượng dự báo.
  * Rà soát checklist hoàn thiện sản phẩm Notebook cuối khóa.

---

## 3. CHECKLIST SẢN PHẨM NOTEBOOK CUỐI KHÓA DÀNH CHO SINH VIÊN

Mỗi sinh viên nộp lại 01 tệp Jupyter Notebook (`.ipynb`) hoàn chỉnh đáp ứng đầy đủ các tiêu chí sau:

1. **Giai đoạn Tiền xử lý (Buổi 1):**
   * Đọc đủ 5 tệp CSV với tham số `na_values`, kiểm tra `.shape`, `.dtypes`, `.isna().sum()`.
   * Áp dụng đúng Regex để làm sạch dữ liệu chuỗi, loại bỏ bản ghi trùng lặp bằng `.drop_duplicates()`.
   * Tách riêng nhóm `new_students` chưa có nhãn và xuất tệp `historical_students.csv` sạch.
2. **Giai đoạn Phân tích & Trực quan hóa (Buổi 2):**
   * Tính đầy đủ `mean, median, min, max` và thống kê gom nhóm `.groupby()`.
   * Vẽ tối thiểu 3 dạng biểu đồ bằng `Matplotlib` thuần (`plot`, `scatter`, `bar`/`hist`) có đầy đủ tiêu đề, nhãn trục, chú giải và kèm nhận xét insight rõ ràng bên dưới.
3. **Giai đoạn Mô hình hóa & Đánh giá (Buổi 3):**
   * Chia tập `train`/`test` hợp lý, thực hiện chuẩn hóa dữ liệu đúng quy tắc chống Data Leakage.
   * Huấn luyện và đánh giá thành công `LinearRegression` (trích xuất `coef_`, `intercept_`, tính `MAE, MSE, R2`) và `KNeighborsClassifier` (xuất `Confusion Matrix`, tính `Accuracy, Precision, Recall, F1-score`).
   * Đưa ra dự đoán cụ thể cho nhóm `new_students` và viết đoạn kết luận 5–7 dòng đánh giá độ tin cậy, hạn chế cũng như hướng cải tiến mô hình.