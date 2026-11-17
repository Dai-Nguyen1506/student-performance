# BUỔI 3: KIẾN THỨC ÁP DỤNG MÔ HÌNH HỌC MÁY (LINEAR REGRESSION & KNN)

> **Mục tiêu tài liệu:** Tổng hợp đầy đủ kiến thức về quy trình chuẩn bị dữ liệu cho `Scikit-learn`, mã hóa biến, chiến lược chia tập, chống **Data Leakage**, nguyên lý toán học và toàn bộ bộ chỉ số đánh giá cho **Linear Regression** và **K-Nearest Neighbors (KNN)**.

---

## 1. Tổng quan Học máy có giám sát (Supervised Learning)

Trong học máy có giám sát, mô hình học từ dữ liệu lịch sử đã biết đáp án $(X, y)$ để dự đoán cho dữ liệu mới:

| Loại bài toán | Đặc điểm biến mục tiêu `y` | Thuật toán học trong khóa | Ví dụ bài toán |
| :--- | :--- | :--- | :--- |
| **Regression (Hồi quy)** | Giá trị số liên tục | `LinearRegression` | Dự báo điểm thi cụ thể (`final_score`) |
| **Classification (Phân loại)** | Nhãn rời rạc (`0` / `1`, hoặc nhiều lớp) | `KNeighborsClassifier` | Cảnh báo sớm nguy cơ trượt môn (`is_at_risk`) |

---

## 2. Chuẩn bị dữ liệu cho Scikit-learn & Chống Data Leakage

### 2.1. Định dạng đầu vào `X` (2D) và `y` (1D)

* **`X` (Ma trận đặc trưng đầu vào):** **Bắt buộc ở dạng 2 chiều (2D)** — shape `(n_samples, n_features)`.  
  *(Lưu ý: Nếu chỉ dùng 1 biến đầu vào, phải viết `df[['col']]` hoặc `.to_numpy().reshape(-1, 1)`, không được truyền mảng 1D `df['col']`)*.
* **`y` (Biến mục tiêu):** Dạng **1 chiều (1D)** — shape `(n_samples,)`.
* **Lưu ý giá trị thiếu:** `LinearRegression` và `KNeighborsClassifier` trong Scikit-learn **không chấp nhận `NaN`** trong `X` hoặc `y` $\rightarrow$ phải đảm bảo đã làm sạch hoàn toàn trước khi `.fit()`.

### 2.2. Mã hóa biến phân loại (Categorical Encoding) — Nếu muốn đưa cột chữ vào mô hình

Mô hình toán học chỉ tính toán trên số, nên nếu muốn dùng thêm các cột dạng chữ (như `Grade_Level` hay `Subject`), ta cần mã hóa:

| Loại biến chữ | Phương pháp mã hóa | Cú pháp |
| :--- | :--- | :--- |
| **Có thứ tự** (`Grade 1`..`Grade 5`, `A+`..`F`) | Ánh xạ sang số tăng dần (`.map` / Regex) | `df['grade_num'] = df['Grade_Level'].str.extract(r'(\d+)').astype(int)` |
| **Không có thứ tự** (`Math`, `Arabic`, `Science`) | **One-Hot Encoding** (tạo các cột nhị phân `0/1`) | `pd.get_dummies(df, columns=['Subject'], drop_first=True)` |

### 2.3. Chiến lược chia tập Train / Test

| Chiến lược | Khi nào dùng? | Cú pháp chuẩn |
| :--- | :--- | :--- |
| **Chia ngẫu nhiên** | Các dòng dữ liệu độc lập nhau | `train_test_split(X, y, test_size=0.2, random_state=42)` |
| **Chia phân tầng (`stratify`)** | Bài toán **Phân loại** có tỷ lệ lớp `0`/`1` mất cân bằng | `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` |
| **Chia tuần tự theo thời gian** | Dữ liệu **chuỗi thời gian** (dự báo tương lai từ quá khứ) | Sắp xếp theo thời gian rồi cắt `train = df[:split_idx]`, `test = df[split_idx:]` |

### 2.4. Chuẩn hóa dữ liệu (`StandardScaler`) & Chống Data Leakage

* **Data Leakage (Rò rỉ dữ liệu):** Xảy ra khi thông tin của tập `test` bị lọt vào quá trình huấn luyện — lỗi phổ biến nhất là gọi `scaler.fit_transform(X)` trên **toàn bộ dữ liệu** trước khi chia `train/test`.
* **Công thức Z-score của `StandardScaler`:** $z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$ (đưa các biến về trung bình $0$, độ lệch chuẩn $1$).

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Bước 1: Chia Train / Test TRƯỚC
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 2: Khởi tạo Scaler -> CHỈ fit trên X_train!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Học mean, std của Train và biến đổi Train
X_test_scaled  = scaler.transform(X_test)        # Dùng mean, std của Train để biến đổi Test
X_new_scaled   = scaler.transform(X_new)         # Dùng đúng scaler đó cho tập dự báo mới
```

---

## 3. Thuật toán 1: Hồi quy tuyến tính (`LinearRegression`)

### 3.1. Bản chất toán học & Tham số

* **Mô hình 1 biến:** $\hat{y} = ax + b$
* **Mô hình đa biến:** $\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_p x_p + b$
* **Nguyên lý học (OLS - Bình phương tối thiểu):** Tìm bộ trọng số $(w, b)$ sao cho tổng bình phương sai số $\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ đạt giá trị nhỏ nhất.
* **Diễn giải tham số sau huấn luyện:**
  * `model.coef_` ($w_j$): Khi biến $x_j$ tăng $1$ đơn vị (các biến khác không đổi), giá trị dự đoán $\hat{y}$ thay đổi $w_j$ đơn vị.
  * `model.intercept_` ($b$): Giá trị dự đoán gốc khi mọi biến đầu vào bằng $0$.

### 3.2. Cú pháp Scikit-learn & Bộ chỉ số đánh giá Hồi quy

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Xem hệ số hồi quy
print("Intercept:", lr.intercept_)
print("Coefficients:", lr.coef_)
```

| Chỉ số | Công thức | Cách đọc & Lưu ý | Hàm `sklearn.metrics` |
| :--- | :--- | :--- | :--- |
| **MAE** | $\frac{1}{n}\sum \|y_i - \hat{y}_i\|$ | Sai số tuyệt đối trung bình (cùng đơn vị với $y$, dễ hiểu nhất) | `mean_absolute_error(y_test, y_pred)` |
| **MSE** | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ | Bình phương sai số trung bình (phạt nặng các điểm đoán lệch xa) | `mean_squared_error(y_test, y_pred)` |
| **RMSE** | $\sqrt{\text{MSE}}$ | Căn bậc hai MSE (đưa về cùng đơn vị với $y$) | `np.sqrt(mean_squared_error(y_test, y_pred))` |
| **$R^2$ Score** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Tỷ lệ phương sai được giải thích ($R^2 \le 1$; gần $1$ là tốt, gần $0$ hoặc âm nghĩa là biến đầu vào gần như không giải thích được $y$) | `r2_score(y_test, y_pred)` |

### 3.3. Biểu đồ kiểm tra mô hình Hồi quy

1. **Biểu đồ Thực tế vs Dự đoán:** `plt.scatter(y_test, y_pred)` + đường chéo $y = x$. Điểm nằm càng sát đường chéo thì dự đoán càng chuẩn.
2. **Biểu đồ phần dư (Residual Plot):** `residuals = y_test - y_pred`, vẽ `plt.scatter(y_pred, residuals)` + `plt.axhline(0)`. Phần dư phân bố đều ngẫu nhiên quanh $0$ là dấu hiệu tốt.

---

## 4. Thuật toán 2: K-Nearest Neighbors (`KNeighborsClassifier`)

### 4.1. Bản chất toán học, Khoảng cách & Tham số $k$

KNN là thuật toán học lười (lazy learning) — không xây phương trình mà dự đoán dựa trên sự tương đồng với **$k$ láng giềng gần nhất** trong tập huấn luyện:
1. Tính **khoảng cách Euclid** giữa điểm mới $x$ và từng điểm $z$ trong `X_train`:
   $$d(x, z) = \sqrt{\sum_{j=1}^{p} (x_j - z_j)^2}$$
2. Chọn $k$ điểm có khoảng cách nhỏ nhất.
3. **Bầu chọn theo đa số (Majority Voting):** Nhãn chiếm đa số trong $k$ láng giềng là nhãn dự đoán.

**Các quy tắc sống còn khi dùng KNN:**
* **Bắt buộc chuẩn hóa (`StandardScaler`):** Nếu một biến có thang `[0, 100]` và biến kia có thang `[0, 5]`, biến có thang lớn sẽ áp đảo hoàn toàn phép tính khoảng cách Euclid. *(Hãy thí nghiệm so sánh trước và sau `StandardScaler` để kiểm chứng!)*.
* **Chọn $k$ lẻ** ($3, 5, 7\dots$) cho bài toán 2 lớp để tránh hòa phiếu.
* **Đánh đổi Bias–Variance khi đổi $k$:**
  * $k$ quá nhỏ (vd $k=1$) $\rightarrow$ ranh giới gồ ghề, dễ **Overfitting** (nhạy cảm với nhiễu).
  * $k$ quá lớn $\rightarrow$ ranh giới quá phẳng, dễ **Underfitting** (bị lớp chiếm đa số lấn át).

### 4.2. Cú pháp Scikit-learn & Tìm giá trị $k$ tốt nhất

```python
from sklearn.neighbors import KNeighborsClassifier

# Huấn luyện và dự đoán (cả nhãn 0/1 lẫn xác suất dự báo)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
y_prob = knn.predict_proba(X_test_scaled)[:, 1]   # Xác suất rơi vào lớp 1

# Vòng lặp tìm k tối ưu (ví dụ thử các số lẻ từ 3 đến 15)
for k in range(3, 16, 2):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    # Ghi nhận F1-score / Recall / Accuracy trên tập test để so sánh
```

### 4.3. Bộ chỉ số đánh giá Phân loại (`sklearn.metrics`)

Với quy ước lớp `1` (Nguy cơ) là **Positive** và lớp `0` (An toàn) là **Negative**:

| | Dự đoán `0` | Dự đoán `1` |
| :--- | :---: | :---: |
| **Thực tế `0`** | **TN** (Đoán đúng An toàn) | **FP** (Báo động giả) |
| **Thực tế `1`** | **FN** (**Bỏ sót** ca nguy cơ) | **TP** (Bắt đúng ca nguy cơ) |

| Chỉ số | Công thức | Ý nghĩa thực tiễn & Khi nào ưu tiên? | Hàm `sklearn.metrics` |
| :--- | :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Tỷ lệ đoán đúng chung *(dễ gây ảo giác nếu dữ liệu lệch lớp vd: 90% lớp 0)* | `accuracy_score(y_test, y_pred)` |
| **Precision** | $\frac{TP}{TP + FP}$ | Độ tinh khiết của cảnh báo: trong các em bị báo nguy cơ, bao nhiêu % đúng? *(Ưu tiên khi chi phí báo động giả cao)* | `precision_score(y_test, y_pred, zero_division=0)` |
| **Recall** | $\frac{TP}{TP + FN}$ | Độ phủ: trong các em **thực sự** có nguy cơ, bắt được bao nhiêu %? *(Cực kỳ quan trọng trong cảnh báo sớm / y tế để không bỏ sót)* | `recall_score(y_test, y_pred, zero_division=0)` |
| **F1-score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Trung bình điều hòa giữa Precision & Recall *(chỉ số cân bằng tốt nhất khi lệch lớp)* | `f1_score(y_test, y_pred, zero_division=0)` |

```python
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Xuất ma trận nhầm lẫn và báo cáo tổng hợp cả 4 chỉ số
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred, zero_division=0))

# Vẽ trực quan Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["An toàn (0)", "Nguy cơ (1)"]).plot(cmap="Blues")
```

---

## 5. Diễn giải kết quả, Hạn chế mô hình & Hướng cải tiến

Khi hoàn thành mô hình, người làm Data cần trả lời trung thực 3 câu hỏi sau trong phần kết luận:

1. **Mức độ tin cậy thực tế:** Dựa trên $R^2$, MAE (với Hồi quy) và Recall, F1-score (với Phân loại), mô hình dự báo hiện tại có đủ tin cậy để ra quyết định hay chưa? Nếu $R^2 \approx 0$ hoặc F1 thấp, nguyên nhân đến từ **dữ liệu đầu vào thiếu tín hiệu dự báo** (tương quan với nhãn gần bằng $0$) chứ không hẳn do lỗi code.
2. **Hạn chế thuật toán:**
   * `LinearRegression`: Giả định tuyến tính cứng nhắc, nhạy cảm với ngoại lai, không bắt được quan hệ phi tuyến.
   * `KNN`: Chậm khi dự đoán trên tập dữ liệu lớn, nhạy cảm với nhiễu và các biến dư thừa (curse of dimensionality).
3. **Hướng cải tiến:**
   * **Feature Engineering:** Khai thác thêm biến mới (vd: điểm trung bình từng môn, tỷ lệ đi muộn, quy đổi `Grade_Feedback` thành GPA, tỷ lệ chữ ký phụ huynh...).
   * **Điều chỉnh bài toán / Thuật toán:** Cân bằng lại ngưỡng phân loại, thử các mô hình phi tuyến như *Decision Tree*, *Random Forest*, *Logistic Regression*.
