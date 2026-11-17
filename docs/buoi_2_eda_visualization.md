# BUỔI 2: KIẾN THỨC VỀ THỐNG KÊ MÔ TẢ, ĐẶT CÂU HỎI & TRỰC QUAN HÓA

> **Mục tiêu tài liệu:** Cung cấp đầy đủ kiến thức về thống kê mô tả, phương pháp tư duy đặt câu hỏi phân tích, bộ hàm vẽ & tinh chỉnh biểu đồ bằng `Matplotlib` thuần, các hàm bổ trợ thống kê của `Seaborn` và kỹ năng đọc – viết insight chuẩn xác.

---

## 1. Thống kê mô tả (Descriptive Statistics) với Pandas & NumPy

Thống kê mô tả giúp tóm tắt tập dữ liệu lớn thành các đại lượng đặc trưng để hiểu hình dạng phân phối, xu hướng trung tâm và sự khác biệt giữa các nhóm.

### 1.1. Các đại lượng đo lường cốt lõi

| Nhóm | Đại lượng | Ý nghĩa & Lưu ý sử dụng | Cú pháp `Pandas` / `NumPy` |
| :--- | :--- | :--- | :--- |
| **Vị trí trung tâm** | **Mean** (Trung bình) | Cộng tất cả chia cho số phần tử; dễ bị kéo lệch bởi ngoại lai | `df['x'].mean()` / `np.mean(x)` |
| **Vị trí trung tâm** | **Median** (Trung vị) | Giá trị ở chính giữa (50%); **bền vững** khi có ngoại lai hoặc phân phối lệch | `df['x'].median()` / `np.median(x)` |
| **Vị trí trung tâm** | **Mode** (Yếu vị) | Giá trị xuất hiện nhiều nhất (thường dùng cho biến phân loại) | `df['x'].mode()[0]` |
| **Độ phân tán** | **Min / Max / Range** | Giá trị nhỏ nhất / lớn nhất / biên độ (`max - min`) | `df['x'].min()`, `.max()` |
| **Độ phân tán** | **Std / Var** | Độ lệch chuẩn / Phương sai; đo độ phân tán quanh trung bình | `df['x'].std()` / `np.std(x)` |
| **Tứ phân vị** | **Q1, Q3 & IQR** | Mốc 25%, 75% và khoảng giữa `IQR = Q3 - Q1` (nền tảng của Boxplot) | `df['x'].quantile([0.25, 0.75])` |

### 1.2. Thống kê tần suất, Bảng chéo, Gom nhóm & Tương quan

```python
# 1. Thống kê mô tả tổng quát cho toàn bộ cột số
df.describe().round(2)

# 2. Tần suất và tỷ lệ (%) của biến phân loại
df["Grade_Level"].value_counts()
df["is_at_risk"].value_counts(normalize=True) * 100

# 3. Bảng tần suất chéo (Crosstab) giữa 2 biến phân loại
# Ví dụ: Tỷ lệ nhóm nguy cơ phân bổ theo từng khối lớp
pd.crosstab(df["Grade_Level"], df["is_at_risk"], normalize="index") * 100

# 4. Phân tích gom nhóm (GroupBy) với nhiều phép thống kê cùng lúc
df.groupby("is_at_risk")[["att_rate", "hw_done_rate", "final_score"]].agg(
    ["count", "mean", "median", "std"]
).round(2)

# 5. Ma trận hệ số tương quan Pearson r trong [-1, 1]
corr_matrix = df.corr(numeric_only=True)
```

> **Cách đọc hệ số tương quan Pearson $r$:**
> * $r > 0$: Tương quan thuận (biến này tăng $\rightarrow$ biến kia có xu hướng tăng).
> * $r < 0$: Tương quan nghịch (biến này tăng $\rightarrow$ biến kia có xu hướng giảm).
> * $|r| \approx 0$ (vd: $-0.02$ đến $+0.02$): **Không có mối liên hệ tuyến tính** giữa hai biến (dấu hiệu quan trọng cần báo cáo trung thực khi đánh giá chất lượng dữ liệu!).

---

## 2. Tư duy đặt câu hỏi phân tích dữ liệu (Analytical Questioning)

Không vẽ biểu đồ theo cảm tính! Mỗi biểu đồ sinh ra phải trả lời cho một **câu hỏi nghiệp vụ cụ thể**:

| Nhóm câu hỏi | Câu hỏi mẫu cần đặt ra | Biểu đồ phù hợp nhất |
| :--- | :--- | :--- |
| **1. Xu hướng (Trend)** | Chỉ số chuyên cần hoặc điểm số thay đổi tăng/giảm thế nào qua các tuần/tháng? | Biểu đồ đường (`plt.plot`) |
| **2. So sánh (Comparison)** | Điểm trung bình hay tỷ lệ hoàn thành bài tập giữa các khối lớp / môn học khác nhau ra sao? | Biểu đồ cột (`plt.bar`, `plt.barh`) |
| **3. Phân phối (Distribution)** | Phổ điểm tập trung ở khoảng nào? Có lệch hay có điểm ngoại lai (outlier) bất thường không? | `plt.hist` & `plt.boxplot` |
| **4. Tương quan & Phân tách lớp** | Hai biến số có liên hệ tuyến tính không? Nhóm `0` và nhóm `1` có tách biệt nhau trên mặt phẳng đặc trưng không? | `plt.scatter`, `sns.regplot`, `sns.heatmap` |

---

## 3. Trực quan hóa dữ liệu toàn diện với Matplotlib thuần

### 3.1. Cấu trúc chuẩn & Các hàm tinh chỉnh biểu đồ

Mọi biểu đồ báo cáo bắt buộc phải rõ ràng, tự giải thích được với đầy đủ thành phần:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 5))                      # Kích thước (rộng, cao) tính bằng inch
# ... Lệnh vẽ biểu đồ ...
plt.title("Tiêu đề mô tả rõ nội dung biểu đồ", fontsize=13, fontweight="bold")
plt.xlabel("Nhãn trục X (kèm đơn vị)")
plt.ylabel("Nhãn trục Y (kèm đơn vị)")
plt.xticks(rotation=45)                         # Xoay nhãn trục X nếu tên dài bị chồng chéo
plt.ylim(0, 105)                                # Giới hạn thang đo trục Y (nếu cần)
plt.axvline(x_val, color="red", linestyle="--", label="Đường tham chiếu")  # Đường dọc/ngang (.axhline)
plt.legend()                                    # Hiển thị chú giải cho các label đã khai báo
plt.grid(True, linestyle="--", alpha=0.5)       # Lưới mờ giúp gióng giá trị chính xác
plt.tight_layout()                              # Tự động căn lề để không bị mất chữ
# plt.savefig("chart.png", dpi=150)             # Lưu biểu đồ thành ảnh (nếu cần xuất báo cáo)
plt.show()
```

### 3.2. Vẽ nhiều biểu đồ cạnh nhau với `plt.subplots()`

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

# Biểu đồ bên trái: axes[0]
axes[0].hist(data, bins=20, color="skyblue", edgecolor="black")
axes[0].set_title("Tiêu đề biểu đồ 1")
axes[0].set_xlabel("Trục X")
axes[0].set_ylabel("Tần suất")

# Biểu đồ bên phải: axes[1]
axes[1].boxplot(data, vert=False)
axes[1].set_title("Tiêu đề biểu đồ 2")
axes[1].set_xlabel("Giá trị")

plt.tight_layout()
plt.show()
```

### 3.3. Cú pháp chi tiết 5 dạng biểu đồ nền tảng

#### A. Biểu đồ đường (`plt.plot`) — Theo dõi diễn biến theo thời gian

```python
plt.plot(
    x_dates, y_values,
    marker="o",         # Điểm đánh dấu ('o', 's', '^', 'd')
    linestyle="-",      # Kiểu nét ('-', '--', '-.', ':')
    color="tab:blue",
    linewidth=2,
    label="Chuỗi A"
)
```

#### B. Biểu đồ phân tán (`plt.scatter`) — Quan sát tương quan & phân tách 2 nhóm

Để kiểm tra xem 2 nhóm nhãn (`0` và `1`) có tách biệt được hay không, tách riêng 2 lệnh `plt.scatter()` với `color` và `marker` khác nhau:

```python
plt.scatter(x_0, y_0, color="tab:green", marker="o", s=25, alpha=0.5, label="Nhóm An toàn (0)")
plt.scatter(x_1, y_1, color="tab:red",   marker="^", s=30, alpha=0.6, label="Nhóm Nguy cơ (1)")
```

#### C. Biểu đồ cột (`plt.bar`, `plt.barh`) — So sánh giữa các nhóm

```python
# Cột đứng đơn
plt.bar(categories, values, color="steelblue", width=0.5)

# Cột ngang (rất tiện khi tên nhóm dài, ví dụ tên môn học)
plt.barh(categories, values, color="teal")

# Cột ghép (so sánh 2 chỉ số trên cùng nhóm)
x = np.arange(len(categories))
w = 0.35
plt.bar(x - w/2, values_1, width=w, label="Chỉ số 1")
plt.bar(x + w/2, values_2, width=w, label="Chỉ số 2")
plt.xticks(x, categories)
```

#### D. Biểu đồ tần suất (`plt.hist`) — Quan sát phân phối dữ liệu liên tục

```python
plt.hist(data, bins=20, color="skyblue", edgecolor="black", alpha=0.8)
# Thêm đường trung bình và trung vị để kiểm tra độ lệch phân phối:
plt.axvline(data.mean(),   color="red",   linestyle="--", label=f"Mean: {data.mean():.1f}")
plt.axvline(data.median(), color="green", linestyle="-",  label=f"Median: {data.median():.1f}")
```

#### E. Biểu đồ hộp (`plt.boxplot`) — Phát hiện ngoại lai & So sánh phân bố nhiều nhóm

* **Cách đọc Boxplot:** Đường ở giữa hộp là **Median (Q2)**; hai cạnh hộp là **Q1 (25%)** và **Q3 (75%)**; hai đầu râu là phạm vi bình thường (`Q1 - 1.5*IQR` đến `Q3 + 1.5*IQR`); các chấm tròn nằm ngoài râu là **Outliers**.

```python
# Boxplot đơn:
plt.boxplot(data, vert=False, patch_artist=True)

# Boxplot so sánh nhiều nhóm bằng Matplotlib thuần:
plt.boxplot([group_0_scores, group_1_scores], tick_labels=["An toàn (0)", "Nguy cơ (1)"])
```

---

## 4. Bổ trợ trực quan hóa thống kê với Seaborn

`Seaborn` xây dựng trên `Matplotlib`, rất mạnh khi cần trực quan hóa mối quan hệ thống kê đa biến chỉ trong 1 câu lệnh:

```python
import seaborn as sns

# 1. Scatter Plot kèm đường hồi quy tuyến tính & vùng tin cậy 95%
sns.regplot(
    data=df, x="bien_x", y="bien_y",
    scatter_kws={"alpha": 0.3, "s": 15},
    line_kws={"color": "red", "linewidth": 2}
)

# 2. Heatmap ma trận tương quan (công cụ chọn lọc biến đầu vào cho Buổi 3)
plt.figure(figsize=(8, 6))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5
)

# 3. Một số biểu đồ bổ trợ nhanh khác:
# sns.countplot(data=df, x="Grade_Level", hue="is_at_risk")  # Đếm tần suất chia theo nhóm màu
# sns.histplot(data=df, x="final_score", kde=True)           # Histogram kèm đường mật độ mượt
# sns.boxplot(data=df, x="is_at_risk", y="att_rate")         # Boxplot theo nhóm nhanh gọn
```

---

## 5. Kỹ năng đọc biểu đồ & Viết Insight chuẩn (*What – Pattern – So What*)

Dưới mỗi biểu đồ trên Notebook, viết nhận xét 2–3 gạch đầu dòng theo đúng 3 tầng tư duy:

1. **What (Quan sát sự thật):** Nêu con số, khoảng tập trung hoặc hình dạng phân phối nổi bật nhất *(tránh nhận xét chung chung kiểu "biểu đồ rất đẹp")*.
2. **Pattern (Xu hướng / So sánh / Bất thường):** Nhóm này cao hay thấp hơn nhóm kia bao nhiêu? Hai biến có tương quan mạnh hay gần như độc lập ($r \approx 0$)? Có hiện tượng chồng lấn (overlap) giữa 2 nhóm lớp hay không?
3. **So What (Hàm ý hành động / Mô hình hóa):** Phát hiện này nói lên điều gì cho bước xây dựng mô hình ở Buổi 3 *(ví dụ: nếu hai nhóm chồng lấn mạnh hoặc tương quan tuyến tính yếu, ta dự báo trước rằng mô hình tuyến tính đơn giản có thể gặp khó khăn và cần thêm đặc trưng mới)*.
