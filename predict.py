from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image
from tensorflow import keras

from preprocess import to_model_array

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
CONFIDENCE_OK = 0.70
MIN_MARGIN = 0.20   # điểm hạng 1 phải hơn hạng 2 ít nhất mức này

if len(sys.argv) != 2:
    print('Cách chạy: python predict.py "đường dẫn ảnh"')
    sys.exit(1)

image_path = Path(sys.argv[1])
if not image_path.is_file():
    print("Không tìm thấy ảnh:", image_path)
    sys.exit(1)

class_names = json.loads((OUTPUT / "class_names.json").read_text(encoding="utf-8"))
model = keras.models.load_model(str(OUTPUT / "best_model.keras"))

array = to_model_array(Image.open(image_path))
scores = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
top = np.argsort(scores)[::-1][:3]

print("\nBa kết quả có điểm dự đoán cao nhất:")
for i in top:
    print(f"{class_names[i]}: {scores[i]:.2%}")

if scores[top[0]] < CONFIDENCE_OK or scores[top[0]] - scores[top[1]] < MIN_MARGIN:
    print("\nCảnh báo: mô hình chưa chắc chắn, hãy kiểm tra lại.")
