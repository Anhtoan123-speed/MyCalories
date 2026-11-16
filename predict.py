from pathlib import Path
import json
import sys

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
IMAGE_SIZE = (128, 128)  # phải khớp với train.py

if len(sys.argv) != 2:
    print('Cách chạy: python predict.py "đường dẫn ảnh"')
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.is_file():
    print("Không tìm thấy ảnh:", image_path)
    sys.exit(1)

class_names = json.loads(
    (OUTPUT / "class_names.json").read_text(encoding="utf-8")
)
model = keras.models.load_model(str(OUTPUT / "best_model.keras"))

# Tiền xử lý GIỐNG app.py và GIỐNG lúc huấn luyện:
# image_dataset_from_directory dùng tf.image.resize (bilinear).
# Bản cũ dùng load_img mặc định (nearest) nên lệch so với lúc train.
image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
image = ImageOps.fit(image, (256, 256), method=Image.Resampling.LANCZOS)
array = tf.image.resize(
    np.asarray(image, dtype="float32"), IMAGE_SIZE, method="bilinear"
).numpy()

# Mô hình đã có lớp chuẩn hóa, không chia 255 thêm
scores = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
top_indices = np.argsort(scores)[::-1][:3]

print("\nBa kết quả có điểm dự đoán cao nhất:")
for index in top_indices:
    print(f"{class_names[index]}: {scores[index]:.2%}")

if scores[top_indices[0]] < 0.60:
    print("\nCảnh báo: mô hình chưa chắc chắn, ảnh có thể không thuộc các loại đã học.")