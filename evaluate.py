"""Đánh giá mô hình bằng ĐÚNG hàm tiền xử lý của app (preprocess.py).

Cách chạy:
  python evaluate.py              # đánh giá thư mục test/
  python evaluate.py real_test    # đánh giá ảnh thật bạn tự chụp
Cấu trúc: <thư mục>/<tên_lớp>/*.jpg. Chỉ cần có các lớp bạn muốn thử,
không bắt buộc đủ 15 lớp. Tên thư mục phải trùng tên lớp trong class_names.json.
"""
from pathlib import Path
import csv
import json
import sys

import numpy as np
from PIL import Image
from tensorflow import keras

from preprocess import to_model_array

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
FOLDER = sys.argv[1] if len(sys.argv) > 1 else "test"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BATCH = 32

class_names = json.loads((OUTPUT / "class_names.json").read_text(encoding="utf-8"))
model = keras.models.load_model(str(OUTPUT / "best_model.keras"))

root = ROOT / FOLDER
if not root.is_dir():
    sys.exit(f"Không thấy thư mục: {root}")

index_of = {name: i for i, name in enumerate(class_names)}
files, labels = [], []
for sub in sorted(p for p in root.iterdir() if p.is_dir()):
    if sub.name not in index_of:
        print(f"Bỏ qua thư mục không thuộc nhãn đã học: {sub.name}")
        continue
    for f in sorted(sub.iterdir()):
        if f.suffix.lower() in EXTS:
            files.append(f)
            labels.append(index_of[sub.name])

if not files:
    sys.exit("Không có ảnh nào để đánh giá.")

y_true = np.array(labels)
probs = []
for start in range(0, len(files), BATCH):
    batch = []
    for f in files[start:start + BATCH]:
        with Image.open(f) as im:
            batch.append(to_model_array(im))
    probs.append(model.predict(np.stack(batch), verbose=0))
probs = np.concatenate(probs)

y_pred = probs.argmax(axis=1)
confidence = probs.max(axis=1)
accuracy = float((y_pred == y_true).mean())
loss = float(-np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-9, 1)).mean())

# Ma trận nhầm lẫn: hàng = nhãn thật, cột = nhãn dự đoán
n = len(class_names)
matrix = np.zeros((n, n), dtype=int)
np.add.at(matrix, (y_true, y_pred), 1)

per_class = {}
for i, name in enumerate(class_names):
    support = int(matrix[i].sum())
    if support == 0:
        continue  # lớp không có ảnh trong thư mục này
    predicted = int(matrix[:, i].sum())
    tp = int(matrix[i, i])
    recall = tp / support
    precision = tp / predicted if predicted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_class[name] = {"precision": round(precision, 4), "recall": round(recall, 4),
                       "f1": round(f1, 4), "support": support}

pairs = sorted(
    ((int(matrix[i, j]), class_names[i], class_names[j])
     for i in range(n) for j in range(n) if i != j and matrix[i, j] > 0),
    reverse=True,
)

# Độ chính xác theo ngưỡng tin cậy: giúp chọn CONFIDENCE_OK trong app
thresholds = {}
for t in (0.5, 0.6, 0.7, 0.8, 0.9):
    mask = confidence >= t
    thresholds[str(t)] = {
        "coverage": round(float(mask.mean()), 4),
        "accuracy": round(float((y_pred[mask] == y_true[mask]).mean()), 4) if mask.any() else None,
    }

results = {
    "folder": FOLDER, "images": len(files),
    "loss": loss, "accuracy": accuracy,
    "per_class": per_class,
    "top_confusions": [{"count": c, "true": a, "predicted": b} for c, a, b in pairs[:10]],
    "confidence_thresholds": thresholds,
}
tag = "test" if FOLDER == "test" else FOLDER
(OUTPUT / f"{tag}_results.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

with open(OUTPUT / f"{tag}_confusion_matrix.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["thật \\ dự đoán"] + class_names)
    for i, name in enumerate(class_names):
        writer.writerow([name] + matrix[i].tolist())

print(f"\nThư mục: {FOLDER} ({len(files)} ảnh)")
print(f"Độ chính xác: {accuracy:.2%}")
print(f"Sai số (loss): {loss:.4f}")

print("\nLớp yếu nhất (recall thấp):")
for name, m in sorted(per_class.items(), key=lambda kv: kv[1]["recall"])[:5]:
    print(f"  {name}: recall {m['recall']:.2%}, precision {m['precision']:.2%} ({m['support']} ảnh)")

print("\nCác cặp hay bị nhầm:")
for c, a, b in pairs[:5]:
    print(f"  {a} -> {b}: {c} ảnh")

print("\nNgưỡng tin cậy (coverage = tỉ lệ ảnh vượt ngưỡng):")
for t, v in thresholds.items():
    acc = "n/a" if v["accuracy"] is None else f"{v['accuracy']:.2%}"
    print(f"  >= {t}: coverage {v['coverage']:.2%}, accuracy {acc}")
