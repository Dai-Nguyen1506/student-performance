# BUỔI 1: KIẾN THỨC NỀN TẢNG VỀ XỬ LÝ DỮ LIỆU (NUMPY, PANDAS & REGEX)

> **Mục tiêu tài liệu:** Hệ thống hóa đầy đủ kiến thức, tư duy làm sạch dữ liệu và danh sách đầy đủ các hàm/câu lệnh của `NumPy`, `Pandas` và `Regex` để học viên tra cứu và áp dụng trực tiếp trên Jupyter Notebook.

---

## 1. Quy trình 6 bước trong một dự án Phân tích Dữ liệu

Mỗi dự án Data Analysis thực tế đều đi theo 6 giai đoạn chuẩn:

1. **Xác định vấn đề (Problem Definition):** Hiểu rõ câu hỏi nghiệp vụ cần trả lời (dự báo điểm số, phân loại rủi ro, hay tìm nguyên nhân?).
2. **Thu thập dữ liệu (Data Collection):** Tiếp nhận dữ liệu từ nhiều nguồn (các tệp CSV rời rạc, database, API...).
3. **Kiểm tra & Làm sạch dữ liệu (Data Auditing & Cleaning):** Phát hiện dữ liệu thiếu, trùng lặp, sai định dạng chuỗi/ngày tháng, ngoại lai, chuẩn hóa và hợp nhất thành bảng sạch. *(Chiếm 60–80% thời gian thực tế).*
4. **Phân tích khám phá (EDA - Exploratory Data Analysis):** Tính toán thống kê mô tả, đặt câu hỏi phân tích và trực quan hóa dữ liệu để tìm quy luật.
5. **Mô hình hóa & Đánh giá (Modeling & Evaluation):** Chuẩn bị dữ liệu đầu vào, huấn luyện thuật toán Machine Learning và đánh giá độ tin cậy.
6. **Diễn giải & Báo cáo (Interpretation & Reporting):** Tổng hợp insight, nêu rõ giới hạn của dữ liệu/mô hình và đề xuất giải pháp cải tiến.

---

## 2. Thư viện NumPy — Toàn bộ hàm & thao tác cốt lõi

`NumPy` là thư viện tính toán mảng số học tốc độ cao nhờ cơ chế **vector hóa (vectorization)** — thực hiện phép toán trên toàn bộ mảng cùng lúc thay vì viết vòng lặp `for`.

### 2.1. Khởi tạo mảng & Kiểm tra thuộc tính

| Hàm / Thuộc tính | Công dụng | Ví dụ cú pháp |
| :--- | :--- | :--- |
| `np.array()` | Tạo mảng 1D hoặc 2D từ Python list | `a = np.array([10, 20, 30])` |
| `np.zeros()` / `np.ones()` | Tạo mảng toàn số `0` hoặc `1` | `np.zeros((2, 3))` |
| `np.arange(start, stop, step)` | Tạo dãy số cách đều theo bước nhảy `step` | `np.arange(0, 10, 2)` $\rightarrow$ `[0, 2, 4, 6, 8]` |
| `np.linspace(start, stop, n)` | Tạo `n` điểm chia đều trong đoạn `[start, stop]` | `np.linspace(0, 100, 5)` |
| `.shape` | Kích thước mảng `(số_dòng, số_cột)` | `a.shape` |
| `.ndim` | Số chiều của mảng (`1` = 1D, `2` = 2D) | `a.ndim` |
| `.size` | Tổng số phần tử trong mảng | `a.size` |
| `.dtype` | Kiểu dữ liệu phần tử (`int64`, `float64`...) | `a.dtype` |
| `.reshape(r, c)` | Đổi hình dạng mảng (hay dùng `(-1, 1)` để đổi 1D $\rightarrow$ 2D) | `a.reshape(-1, 1)` |

### 2.2. Tính toán vector hóa, Thống kê, Lọc điều kiện & Xử lý `NaN`

```python
import numpy as np

arr = np.array([45.0, 80.0, np.nan, 92.0, 60.0])

# 1. Tính toán vector hóa (áp dụng cho cả mảng)
arr_scaled = (arr + 5) * 2

# 2. Thống kê nhanh trên mảng thường (lưu ý: gặp np.nan sẽ trả về nan!)
# np.sum(arr), np.mean(arr), np.median(arr), np.min(arr), np.max(arr), np.std(arr)

# 3. Thống kê bỏ qua NaN (rất hay dùng khi dữ liệu có giá trị thiếu)
print("Sum bỏ NaN:",    np.nansum(arr))
print("Mean bỏ NaN:",   np.nanmean(arr))
print("Median bỏ NaN:", np.nanmedian(arr))
print("Std bỏ NaN:",    np.nanstd(arr))

# 4. Kiểm tra và lọc điều kiện (Boolean Indexing)
is_missing = np.isnan(arr)                   # Mảng True/False tại vị trí NaN
valid_scores = arr[~np.isnan(arr)]           # Dấu ~ dùng để phủ định điều kiện
passed = arr[arr >= 50]                      # Lấy các phần tử >= 50

# 5. Gán nhãn theo điều kiện
# np.where(điều_kiện, nếu_đúng, nếu_sai) -> dùng cho 2 nhánh
status_2 = np.where(arr >= 50, "Đạt", "Trượt")

# np.select(list_điều_kiện, list_kết_quả, default=...) -> dùng khi có từ 3 nhánh trở lên
conditions = [arr >= 80, arr >= 50]
choices = ["Giỏi", "Đạt"]
status_3 = np.select(conditions, choices, default="Chưa đạt")
```

---

## 3. Thư viện Pandas — Đọc, Lưu & Kiểm tra chất lượng (Data Auditing)

### 3.1. Đọc & Xuất dữ liệu CSV

```python
import pandas as pd

# Khai báo các ký hiệu đại diện cho giá trị thiếu ngay khi đọc file
NA_TOKENS = ["NA", "N/A", "?", "null", "NULL", "None", ""]
df = pd.read_csv("path/to/file.csv", na_values=NA_TOKENS)

# Xuất DataFrame sạch ra file CSV (luôn đặt index=False để tránh sinh cột Unnamed: 0)
df.to_csv("path/to/clean_file.csv", index=False)
```

### 3.2. Bộ lệnh kiểm tra tổng quan, giá trị thiếu & trùng lặp

| Câu lệnh | Mục đích | Lưu ý thực tế |
| :--- | :--- | :--- |
| `df.head(n)` / `df.tail(n)` / `df.sample(n)` | Xem `n` dòng đầu / cuối / ngẫu nhiên | Dùng `.sample(5)` giúp phát hiện lỗi nằm ở giữa file |
| `df.shape` | Xem `(số_dòng, số_cột)` | Kiểm tra số dòng trước và sau khi lọc/xóa/merge |
| `df.columns` | Liệt kê danh sách tên tất cả các cột | Kiểm tra xem tên cột có khoảng trắng thừa hay không |
| `df.dtypes` | Xem kiểu dữ liệu từng cột | Cột lẽ ra là số/ngày nhưng hiện `object`/`str` $\rightarrow$ đang bị dính ký tự lạ |
| `df.info()` | Tổng quan số dòng, non-null count, kiểu dữ liệu & bộ nhớ | Nhận biết nhanh cột nào bị khuyết thiếu |
| `df.describe(include='all')` | Thống kê nhanh cả cột số lẫn cột chữ | Thấy ngay `min`/`max` vô lý (vd: `-5`, `110`) |
| `df.isna().sum()` | Đếm số lượng `NaN` của từng cột | Kết hợp `(df.isna().mean() * 100).round(2)` để xem % thiếu |
| `df['col'].unique()` / `.nunique()` | Liệt kê giá trị duy nhất / Đếm số giá trị duy nhất | Phát hiện chữ hoa/thường lộn xộn, emoji, khoảng trắng |
| `df['col'].value_counts(dropna=False)` | Đếm tần suất từng giá trị (kể cả `NaN`) | Dùng để kiểm tra các nhóm bất thường như `' '` |
| `df.duplicated().sum()` | Đếm số dòng **trùng lặp hoàn toàn** ở tất cả các cột | Phải kiểm tra trước khi tổng hợp dữ liệu |
| `df.duplicated(subset=['c1', 'c2']).sum()` | Đếm số dòng **trùng theo khóa nghiệp vụ** | Vd: cùng 1 sinh viên, cùng ngày, cùng môn học nhưng bị ghi 2 lần |
| `df[df.duplicated(subset=[...], keep=False)]` | In ra **toàn bộ các cặp dòng bị trùng** để đối chiếu | Giúp biết chúng giống hệt nhau hay khác nhau ở cột trạng thái |

---

## 4. Truy cập, Lọc, Chỉnh sửa & Xóa dữ liệu trong Pandas

### 4.1. Chọn cột, Chọn dòng & Lọc điều kiện

| Thao tác | Cú pháp chuẩn | Ghi chú |
| :--- | :--- | :--- |
| Chọn 1 cột (trả về Series 1D) | `df['col']` | Dùng khi tính toán 1 biến hoặc gán làm `y` |
| Chọn nhiều cột (trả về DataFrame 2D) | `df[['col1', 'col2']]` | Bắt buộc dùng `[[]]` khi cần giữ dạng bảng 2D |
| Truy cập theo **tên nhãn / điều kiện** | `df.loc[df['score'] >= 50, ['ID', 'score']]` | Vừa lọc dòng vừa chọn cột cùng lúc |
| Truy cập theo **chỉ số vị trí nguyên** | `df.iloc[0:5, 0:3]` | Lấy 5 dòng đầu, 3 cột đầu |
| Lọc kết hợp nhiều điều kiện | `df[(df['a'] > 0) & (df['b'] < 100)]` | Luôn bọc mỗi điều kiện trong `(...)`; dùng `&` (VÀ), `\|` (HOẶC), `~` (KHÔNG) |
| Lọc theo danh sách giá trị | `df[df['Grade'].isin(['Grade 1', 'Grade 2'])]` | Gọn hơn viết nhiều phép `\|` |
| Lọc trong khoảng | `df[df['score'].between(0, 100)]` | Tương đương `(score >= 0) & (score <= 100)` |

### 4.2. Xóa cột, Xóa dòng & Xóa trùng lặp (`drop`, `drop_duplicates`)

```python
# 1. Xóa một hoặc nhiều cột không cần thiết (dùng columns=[...])
df = df.drop(columns=["col_thua_1", "col_thua_2"])
# Hoặc chỉ giữ lại các cột cần dùng (cách này thường an toàn và rõ ràng hơn):
# df = df[["ID", "Name", "Score"]]

# 2. Xóa dòng theo điều kiện (thực chất là giữ lại các dòng thỏa điều kiện ngược lại)
df = df[df["Score"] >= 0]

# Hoặc xóa dòng theo index cụ thể:
df = df.drop(index=[0, 1, 2])

# 3. Xóa dòng trùng lặp (Duplicates Removal)
# keep='first' : giữ dòng đầu tiên (mặc định)
# keep='last'  : giữ dòng cuối cùng
# keep=False   : xóa sạch tất cả các dòng bị trùng
df = df.drop_duplicates()                                      # Trùng toàn bộ cột
df = df.drop_duplicates(subset=["ID", "Date"], keep="first")   # Trùng theo khóa

# 4. Đổi tên cột, Sắp xếp và Reset lại chỉ số dòng
df = df.rename(columns={"Old_Name": "new_name"})
df = df.sort_values(by=["ID", "Date"], ascending=[True, True]).reset_index(drop=True)
```

---

## 5. Biểu thức chính quy (Regex) & Làm sạch dữ liệu chuỗi (`.str`)

Dữ liệu thu thập thực tế thường dính ký tự đơn vị (`"80%"`, `"100%"`), dấu âm/chữ (`"score: 90"`), khoảng trắng ẩn (`"PRESENT "`, `" "` — một chuỗi toàn dấu cách!), emoji (`"✅"`, `"❌"`) hoặc sai chính tả (`"absnt"`).

### 5.1. Bảng ký hiệu Regex cần nhớ

| Ký hiệu | Ý nghĩa | Ví dụ khớp |
| :--- | :--- | :--- |
| `\d` / `\D` | Chữ số (`0–9`) / **Không phải** chữ số | `\d+` khớp `85`, `2025` |
| `\w` / `\W` | Chữ cái, chữ số, `_` / Ký tự đặc biệt | `\w+` khớp `S00001` |
| `\s` / `\S` | Khoảng trắng (` `, `\t`, `\n`) / Ký tự không phải khoảng trắng | `\s+` khớp nhiều dấu cách liên tiếp |
| `.` | Bất kỳ ký tự nào (trừ xuống dòng) | `a.c` khớp `abc`, `a-c` |
| `+` / `*` / `?` | Lặp **$\ge 1$ lần** / **$\ge 0$ lần** / **$0$ hoặc $1$ lần** | `-?\d+` khớp cả `-5` và `90` |
| `{m,n}` | Lặp từ `m` đến `n` lần | `\d{4}` khớp năm gồm 4 chữ số `2025` |
| `^` / `$` | Bắt đầu chuỗi / Kết thúc chuỗi | `^\s*$` khớp chuỗi rỗng hoặc toàn dấu cách |
| `[]` / `[^]` | Tập ký tự cho phép / **Phủ định** (không nằm trong ngoặc) | `[^0-9.-]` khớp mọi ký tự không phải số, `.`, `-` |
| `\|` | Phép **HOẶC** | `done\|✔\|✅` khớp một trong 3 mẫu |
| `(...)` | Nhóm rút trích (Capturing group) | `r"(-?\d+\.?\d*)"` dùng với `.str.extract()` |

### 5.2. Các hàm `.str` làm sạch chuỗi trong Pandas

```python
# 1. Cắt khoảng trắng đầu/cuối và chuẩn hóa chữ hoa/thường
s = df["col"].astype(str).str.strip().str.lower()    # ngoài ra có .str.upper(), .str.title()

# 2. Biến các chuỗi rỗng hoặc chỉ chứa toàn khoảng trắng (' ', '   ') thành NaN
df["col"] = df["col"].replace(r"^\s*$", np.nan, regex=True)

# 3. Xóa ký tự không phải số (như '%', chữ) bằng .str.replace(..., regex=True)
df["pct_clean"] = df["pct_raw"].astype(str).str.replace(r"[^0-9.-]", "", regex=True)

# 4. Gộp nhiều khoảng trắng liên tiếp thành 1 dấu cách chuẩn
df["name_clean"] = df["name"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

# 5. Rút trích phần số hoặc mã bằng .str.extract()
# Ví dụ: "Grade 3" -> "3", hoặc "score: 85.5%" -> "85.5"
df["grade_num"] = df["Grade_Level"].str.extract(r"(\d+)")

# 6. Kiểm tra chuỗi có chứa mẫu Regex hay không (na=False tránh lỗi khi có NaN)
has_done = s.str.contains(r"done|✔|✅", regex=True, na=False)

# 7. Tách chuỗi thành nhiều cột theo dấu phân tách
# df[["first", "last"]] = df["Full_Name"].str.split(" ", n=1, expand=True)
```

---

## 6. Ép kiểu dữ liệu, Ánh xạ, Ngoại lai & Giá trị thiếu

### 6.1. Ép kiểu số và ngày tháng hỗn hợp

```python
# 1. Ép kiểu sang số (chuỗi lỗi không ép được sẽ tự biến thành NaN nhờ errors='coerce')
df["score"] = pd.to_numeric(df["score_str"], errors="coerce")

# 2. Ép kiểu sang số nguyên / số thực / chuỗi khi cột đã sạch
df["flag"] = df["flag"].astype(int)

# 3. Ép kiểu ngày tháng (QUAN TRỌNG: dùng format='mixed' khi dữ liệu lẫn nhiều định dạng
# như '2025-03-09' và '01/01/2025' hoặc '01-01-2009')
df["date_clean"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")

# Trích xuất thành phần thời gian bằng .dt
df["year"] = df["date_clean"].dt.year
df["year_month"] = df["date_clean"].dt.to_period("M").astype(str)
```

### 6.2. Ánh xạ nhãn (`.map`, `.replace`) & Phân khoảng (`pd.cut`)

```python
# 1. Quy đổi điểm chữ hoặc nhãn phân loại sang số bằng .map()
grade_map = {"A+": 4.0, "A": 3.7, "B": 3.0, "B-": 2.7, "C": 2.0, "C-": 1.7, "D": 1.0, "F": 0.0}
df["gpa"] = df["Grade_Feedback"].str.strip().map(grade_map)

# 2. Thay thế một vài giá trị cụ thể nhưng giữ nguyên các giá trị khác bằng .replace()
df["sig"] = df["Guardian_Signature"].replace({"Yes": 1, "No": 0})

# 3. Rời rạc hóa biến số liên tục thành các nhóm bằng pd.cut()
df["score_tier"] = pd.cut(
    df["Exam_Score"],
    bins=[0, 50, 70, 85, 100],
    labels=["Yếu", "Trung bình", "Khá", "Giỏi"],
    include_lowest=True
)
```

### 6.3. Xử lý ngoại lai (Outliers) & Giá trị thiếu (Missing Values)

```python
# 1. Xử lý ngoại lai biết trước miền giá trị hợp lệ (vd: điểm/tỷ lệ % chỉ từ 0 đến 100)
df["Exam_Score"] = df["Exam_Score"].clip(lower=0, upper=100)

# 2. Xử lý giá trị thiếu:
# Cách A: Xóa các dòng bị thiếu ở cột quan trọng (hoặc xóa cột nếu thiếu quá nhiều)
df_dropped = df.dropna(subset=["important_col"])

# Cách B: Điền giá trị thiếu (Imputation) cho BIẾN ĐẦU VÀO
df["comm_count"] = df["comm_count"].fillna(0)                 # Không có log tương tác -> 0 lần
df["hw_rate"] = df["hw_rate"].fillna(df["hw_rate"].median())  # Điền trung vị khi phân phối lệch
df["category"] = df["category"].fillna("Unknown")             # Biến phân loại -> nhãn "Unknown"
```

> **Nguyên tắc tư duy sống còn khi xử lý `NaN`:**  
> Phải tách bạch giữa **biến đầu vào (features)** và **biến mục tiêu (target)**:
> * Khi ghép bảng thông tin sinh viên với bảng điểm thi, những đối tượng chưa có điểm sẽ có `target = NaN`.
> * **Tuyệt đối không** dùng `.fillna(mean)` cho cột mục tiêu! Phải tách riêng nhóm `df[df["target"].isna()]` thành tập đối tượng mới cần dự báo (`new_data`), và chỉ giữ `df[df["target"].notna()]` làm dữ liệu lịch sử (`historical_data`).

---

## 7. Tổng hợp (`groupby`) & Ghép bảng (`merge`, `concat`)

### 7.1. Tổng hợp dữ liệu từ bảng nhật ký (`groupby` + `agg`)

Khi một đối tượng xuất hiện nhiều lần ở bảng chi tiết (nhiều ngày điểm danh, nhiều bài tập, nhiều môn thi), ta phải gom nhóm về **1 dòng / 1 đối tượng** trước khi ghép bảng:

```python
# Tổng hợp nhiều chỉ số cùng lúc và đặt tên cột mới sạch sẽ (Named Aggregation)
agg_df = df.groupby("Student_ID").agg(
    total_records=("Date", "count"),
    mean_score=("Exam_Score", "mean"),
    max_score=("Exam_Score", "max"),
    min_score=("Exam_Score", "min")
).reset_index()   # Luôn dùng reset_index() để đưa Student_ID từ index trở lại thành cột bình thường
```

### 7.2. Ghép bảng theo chiều ngang (`pd.merge`)

```python
# Cú pháp chuẩn:
# how='left'  : Giữ toàn bộ dòng của bảng bên trái (phổ biến nhất để không làm mất đối tượng chưa có log)
# how='inner' : Chỉ giữ các khóa có mặt ở CẢ hai bảng
# how='outer' : Giữ tất cả khóa từ cả hai bảng
merged = df_left.merge(df_right, on="Student_ID", how="left")

# Trường hợp tên cột khóa ở hai bảng khác nhau:
# merged = df_left.merge(df_right, left_on="ID", right_on="Student_ID", how="left")

# Kiểm tra quan trọng sau mỗi lần merge:
# Số dòng sau left-merge KHÔNG ĐƯỢC tăng lên nếu bảng phải đã gom nhóm duy nhất theo khóa!
assert len(merged) == len(df_left), "Cảnh báo: Bảng bên phải bị trùng khóa làm nhân bản dòng!"
```

### 7.3. Nối bảng theo chiều dọc (`pd.concat`)

Dùng khi cần ghép nối nhiều bảng có **cùng cấu trúc cột** xếp chồng lên nhau (ví dụ dữ liệu học kỳ 1 và học kỳ 2):

```python
combined = pd.concat([df_part1, df_part2], axis=0, ignore_index=True)
```
