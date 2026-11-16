from pathlib import Path
import csv
import json

import numpy as np
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"

# Đọc đúng thứ tự nhãn đã dùng khi huấn luyện
class_names = json.loads(
    (OUTPUT / "class_names.json").read_text(encoding="utf-8")
)
model = keras.models.load_model(str(OUTPUT / "best_model.keras"))

# Đọc tập test, không xáo trộn để thứ tự nhãn khớp thứ tự dự đoán
test_ds = keras.utils.image_dataset_from_directory(
    str(ROOT / "test"),
    class_names=class_names,
    image_size=(128, 128),
    batch_size=32,
    label_mode="int",
    shuffle=False,
)

# Mô hình đã có lớp chuẩn hóa, không chia 255 thêm
loss, accuracy = model.evaluate(test_ds)

y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
probs = model.predict(test_ds, verbose=0)
y_pred = probs.argmax(axis=1)
confidence = probs.max(axis=1)

# Ma trận nhầm lẫn: hàng = nhãn thật, cột = nhãn dự đoán
n = len(class_names)
matrix = np.zeros((n, n), dtype=int)
np.add.at(matrix, (y_true, y_pred), 1)

per_class = {}
for i, name in enumerate(class_names):
    support = int(matrix[i].sum())
    predicted = int(matrix[:, i].sum())
    tp = int(matrix[i, i])
    recall = tp / support if support else 0.0
    precision = tp / predicted if predicted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_class[name] = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": support,
    }

# Các cặp nhầm lẫn nhiều nhất
pairs = [
    (int(matrix[i, j]), class_names[i], class_names[j])
    for i in range(n) for j in range(n) if i != j and matrix[i, j] > 0
]
pairs.sort(reverse=True)

# Độ chính xác theo ngưỡng tin cậy: giúp chọn CONFIDENCE_OK trong app.py
thresholds = {}
for t in (0.5, 0.6, 0.7, 0.8, 0.9):
    mask = confidence >= t
    thresholds[str(t)] = {
        "coverage": round(float(mask.mean()), 4),
        "accuracy": round(float((y_pred[mask] == y_true[mask]).mean()), 4) if mask.any() else None,
    }

results = {
    "test_loss": float(loss),
    "test_accuracy": float(accuracy),
    "per_class": per_class,
    "top_confusions": [
        {"count": c, "true": a, "predicted": b} for c, a, b in pairs[:10]
    ],
    "confidence_thresholds": thresholds,
}
(OUTPUT / "test_results.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
)

with open(OUTPUT / "confusion_matrix.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["thật \\ dự đoán"] + class_names)
    for i, name in enumerate(class_names):
        writer.writerow([name] + matrix[i].tolist())

print(f"\nĐộ chính xác trên tập test: {accuracy:.2%}")
print(f"Sai số trên tập test: {loss:.4f}")

print("\nLớp yếu nhất (recall thấp):")
for name, m in sorted(per_class.items(), key=lambda kv: kv[1]["recall"])[:5]:
    print(f"  {name}: recall {m['recall']:.2%}, precision {m['precision']:.2%}")

print("\nCác cặp hay bị nhầm:")
for c, a, b in pairs[:5]:
    print(f"  {a} -> {b}: {c} ảnh")

print("\nNgưỡng tin cậy (coverage = tỉ lệ ảnh vượt ngưỡng):")
for t, v in thresholds.items():
    acc = "n/a" if v["accuracy"] is None else f"{v['accuracy']:.2%}"
    print(f"  >= {t}: coverage {v['coverage']:.2%}, accuracy {acc}")