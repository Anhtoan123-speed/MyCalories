# MyCalories

**MyCalories** là ứng dụng web viết bằng [Streamlit](https://streamlit.io/) giúp bạn lập thực đơn giảm cân một cách đơn giản: biết mình nên ăn bao nhiêu calo mỗi ngày, chọn món từ nguyên liệu đang có trong bếp, rồi kiểm tra thực đơn đã đạt mục tiêu hay chưa.

## Mục lục

1. [Ứng dụng làm được gì?](#1-ứng-dụng-làm-được-gì)
2. [Chuẩn bị trước khi chạy](#2-chuẩn-bị-trước-khi-chạy)
3. [Cài đặt và chạy trên Windows](#3-cài-đặt-và-chạy-trên-windows)
4. [Hướng dẫn sử dụng từng bước](#4-hướng-dẫn-sử-dụng-từng-bước)
5. [Cấu trúc thư mục và các tệp cần có](#5-cấu-trúc-thư-mục-và-các-tệp-cần-có)
6. [Mô hình nhận diện và dữ liệu huấn luyện](#6-mô-hình-nhận-diện-và-dữ-liệu-huấn-luyện)
7. [Xử lý sự cố thường gặp](#7-xử-lý-sự-cố-thường-gặp)

---

## 1. Ứng dụng làm được gì?

Giao diện được chia thành năm mục. Mỗi mục đảm nhận một bước trong quy trình lập thực đơn:

| Mục | Chức năng | Kết quả bạn nhận được |
| --- | --- | --- |
| **Mục tiêu** | Nhập thông tin hồ sơ cá nhân. | Chỉ số BMI, BMR, TDEE và mục tiêu calo mỗi ngày. |
| **Nhận diện** | Tải ảnh lên hoặc chụp ảnh rau củ. Bạn xác nhận hoặc chỉnh lại kết quả trước khi thêm vào danh sách nguyên liệu. | Danh sách nguyên liệu đang có. |
| **Món ăn** | Tìm món theo nguyên liệu đang có, xem cách nấu, điều chỉnh định lượng và khẩu phần. | Các món phù hợp cùng thông tin dinh dưỡng. |
| **Thực đơn** | Thêm món vào từng bữa trong ngày. | Tổng calo dự kiến, lượng đạm, tinh bột, chất béo; có thể xuất ra tệp CSV. |
| **Cân nặng** | Ghi lại cân nặng theo ngày. | Biểu đồ và tiến độ thay đổi cân nặng. |

**Một vài thuật ngữ:**

- **BMI**: chỉ số khối cơ thể, tính từ chiều cao và cân nặng.
- **BMR**: năng lượng cơ thể tiêu hao khi nghỉ ngơi hoàn toàn.
- **TDEE**: tổng năng lượng tiêu hao trong một ngày, đã tính cả mức độ vận động.
- **Mục tiêu calo**: mức calo nên nạp mỗi ngày để đạt mục tiêu cân nặng của bạn.

> **Lưu ý:** Các chỉ số trong ứng dụng chỉ mang tính ước tính để tham khảo, không thay thế lời khuyên của bác sĩ hoặc chuyên gia dinh dưỡng.

---

## 2. Chuẩn bị trước khi chạy

Bạn cần có:

- Máy tính chạy **Windows** và đã cài **Python** (lệnh `py` dùng được trong PowerShell).
- **VS Code** (hoặc bất kỳ trình soạn thảo nào có Terminal).
- Toàn bộ thư mục dự án, bao gồm các tệp mô hình trong `outputs/` (xem [mục 5](#5-cấu-trúc-thư-mục-và-các-tệp-cần-có)).
- Kết nối Internet trong lần cài đặt đầu tiên để tải các thư viện.

> **Gợi ý:** TensorFlow thường chỉ hỗ trợ một số phiên bản Python nhất định. Nếu bước cài đặt báo lỗi liên quan đến TensorFlow, hãy thử dùng phiên bản Python khác đang được TensorFlow hỗ trợ.

---

## 3. Cài đặt và chạy trên Windows

### Bước 1. Mở dự án

Mở thư mục dự án trong VS Code, sau đó chọn **Terminal → New Terminal**. Hãy chắc chắn Terminal đang dùng **PowerShell**.

### Bước 2. Tạo môi trường ảo

Môi trường ảo giúp các thư viện của dự án tách biệt với phần còn lại của máy.

```powershell
py -m venv .venv
```

> Nếu thư mục `.venv` đã có sẵn trong dự án, bạn có thể **bỏ qua bước này**.

### Bước 3. Cài đặt thư viện

Nếu dự án có tệp `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu bản tải về **chưa có** `requirements.txt`, cài trực tiếp các thư viện cần thiết:

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit numpy pandas pillow tensorflow ultralytics
```

| Thư viện | Dùng để |
| --- | --- |
| `streamlit` | Xây dựng giao diện web. |
| `numpy`, `pandas` | Xử lý số liệu và bảng dữ liệu. |
| `pillow` | Đọc và xử lý ảnh. |
| `tensorflow` | Chạy mô hình CNN (Keras). |
| `ultralytics` | Chạy mô hình YOLO11. |

### Bước 4. Chạy ứng dụng

```powershell
.\.venv\Scripts\python.exe -m streamlit run app_mycalories_olivine.py
```

### Bước 5. Mở ứng dụng trong trình duyệt

Sau khi chạy, Terminal sẽ in ra dòng **Local URL**, thường là:

[http://localhost:8501](http://localhost:8501)

Bấm vào địa chỉ này (hoặc dán vào trình duyệt) để mở ứng dụng. Để dừng ứng dụng, quay lại Terminal và nhấn `Ctrl + C`.

---

## 4. Hướng dẫn sử dụng từng bước

Để thử toàn bộ quy trình, hãy đi theo thứ tự **Mục tiêu → Nhận diện → Món ăn → Thực đơn → Cân nặng**.

### Bước 1. Mục tiêu: nhập hồ sơ

Điền các thông tin cá nhân được yêu cầu. Ứng dụng sẽ tính BMI, BMR, TDEE và đưa ra **mục tiêu calo mỗi ngày**. Con số này sẽ được dùng để so sánh với thực đơn ở các bước sau.

### Bước 2. Nhận diện: xác định nguyên liệu bằng ảnh

1. Tải ảnh lên hoặc chụp ảnh rau củ trực tiếp.
2. Chọn chế độ nhận diện phù hợp:
   - **Một loại rau củ trong ảnh** → dùng mô hình **CNN**.
   - **Nhiều loại rau củ trong cùng một ảnh** → dùng mô hình **YOLO11**.
3. Xem kết quả nhận diện, **xác nhận hoặc sửa lại** nếu chưa đúng.
4. Thêm các nguyên liệu đã xác nhận vào danh sách.

### Bước 3. Món ăn: chọn món từ nguyên liệu có sẵn

1. Tìm món theo danh sách nguyên liệu vừa thêm.
2. Mở món bạn quan tâm để xem **cách nấu**.
3. Điều chỉnh **định lượng** và **khẩu phần** cho phù hợp với nhu cầu.

### Bước 4. Thực đơn: ghép món theo từng bữa

1. Thêm món vào từng bữa trong ngày.
2. Xem **tổng calo dự kiến** và lượng **đạm, tinh bột, chất béo**.
3. Đối chiếu với mục tiêu calo ở Bước 1 để biết thực đơn đang thừa hay thiếu.
4. Xuất thực đơn ra tệp **CSV** nếu muốn lưu hoặc mở bằng Excel.

### Bước 5. Cân nặng: theo dõi tiến độ

Ghi lại cân nặng mỗi ngày để theo dõi bạn có đang tiến gần đến mục tiêu hay không.

> Hồ sơ và thực đơn được lưu trong thư mục `data/` trên chính máy chạy ứng dụng, nên dữ liệu của bạn không bị gửi đi đâu khác.

---

## 5. Cấu trúc thư mục và các tệp cần có

| Tệp / thư mục | Vai trò |
| --- | --- |
| `app_mycalories_olivine.py` | Tệp chính để chạy ứng dụng, chứa giao diện Olivine. |
| `recipes_extra.py` | Các công thức món ăn bổ sung. |
| `assets/healthy-lifestyle.jpg` | Ảnh hiển thị trong phần giới thiệu. |
| `.streamlit/config.toml` | Cấu hình giao diện Streamlit. |
| `outputs/best_model.keras` | Mô hình CNN, phân loại ảnh chứa **một** loại rau củ. |
| `outputs/class_names.json` | Danh sách nhãn tương ứng với đầu ra của CNN. |
| `outputs/vegetable_yolo.pt` | Trọng số YOLO11 đã huấn luyện, nhận diện **nhiều** vật thể trong một ảnh. |
| `data/` | Nơi lưu hồ sơ và thực đơn trên máy chạy ứng dụng. |

**Nếu thiếu tệp mô hình:** hãy đặt đúng tệp vào thư mục `outputs/` trước khi dùng chế độ nhận diện tương ứng.

---

## 6. Mô hình nhận diện và dữ liệu huấn luyện

| | CNN (TensorFlow/Keras) | YOLO11 (Ultralytics) |
| --- | --- | --- |
| **Dùng khi** | Ảnh chỉ có một loại rau củ. | Ảnh có nhiều loại rau củ cùng lúc. |
| **Phạm vi nhận diện** | 15 lớp rau củ. | Bông cải xanh, cà rốt, hành tây, ớt chuông, cà chua. |
| **Tệp cần có** | `best_model.keras`, `class_names.json` | `vegetable_yolo.pt` |
| **Bộ dữ liệu** | *Vegetable Image Dataset* (Kaggle). | Bộ ảnh có nhãn khung (bounding box) từ Roboflow. |

---

## 7. Xử lý sự cố thường gặp

| Vấn đề | Cách xử lý |
| --- | --- |
| Lệnh `py` không được nhận diện | Python chưa được cài hoặc chưa thêm vào PATH. Cài lại Python và tích chọn tùy chọn thêm vào PATH. |
| Lỗi khi cài `tensorflow` | Phiên bản Python có thể chưa được TensorFlow hỗ trợ. Thử dùng phiên bản Python khác. |
| Báo thiếu mô hình khi nhận diện | Kiểm tra các tệp trong `outputs/` theo bảng ở [mục 5](#5-cấu-trúc-thư-mục-và-các-tệp-cần-có). |
| Không mở được `localhost:8501` | Kiểm tra ứng dụng vẫn đang chạy trong Terminal và dùng đúng địa chỉ **Local URL** được in ra. |
| Cổng 8501 đang bị chiếm | Streamlit sẽ tự chọn cổng khác (ví dụ 8502); hãy dùng địa chỉ mới hiển thị trong Terminal. |
| Nhận diện sai nguyên liệu | Chỉnh lại kết quả thủ công ở mục **Nhận diện** trước khi thêm vào danh sách. |
