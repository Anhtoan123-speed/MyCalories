from pathlib import Path
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------
# 1. Cấu hình
# ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)   # phải khớp preprocess.py
BATCH_SIZE = 32
EPOCHS_HEAD = 12          # giai đoạn 1: chỉ train lớp phân loại
EPOCHS_FINE = 15          # giai đoạn 2: tinh chỉnh các lớp cuối của MobileNetV2
FINE_TUNE_FROM = -40      # mở khóa 40 lớp cuối
REAL_REPEAT = 6           # ảnh thật (thư mục real_train/) được lặp lại N lần để model chú ý hơn

keras.utils.set_random_seed(42)

# ---------------------------------------------------------------
# 2. Dữ liệu (cắt vuông giữa ảnh thay vì bóp méo)
# ---------------------------------------------------------------
train_ds = keras.utils.image_dataset_from_directory(
    str(DATA / "train"),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=42,
    crop_to_aspect_ratio=True,
)
class_names = train_ds.class_names
print("Các loại rau củ:", class_names)

val_ds = keras.utils.image_dataset_from_directory(
    str(DATA / "validation"),
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
    crop_to_aspect_ratio=True,
)

(OUTPUT / "class_names.json").write_text(
    json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8"
)

# Ảnh thật tự chụp: real_train/<tên_lớp>/*.jpg (không cần đủ 15 lớp)
REAL_DIR = DATA / "real_train"
if REAL_DIR.is_dir() and any(REAL_DIR.iterdir()):
    present = sorted(p.name for p in REAL_DIR.iterdir() if p.is_dir())
    unknown = [n for n in present if n not in class_names]
    if unknown:
        raise SystemExit(f"Thư mục trong real_train không trùng tên lớp: {unknown}\n"
                         f"Tên hợp lệ: {class_names}")
    real_ds = keras.utils.image_dataset_from_directory(
        str(REAL_DIR),
        class_names=present,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=42,
        crop_to_aspect_ratio=True,
    )
    # đổi nhãn cục bộ sang chỉ số nhãn chung
    label_map = tf.constant([class_names.index(n) for n in present], dtype=tf.int32)
    real_ds = real_ds.unbatch().map(
        lambda x, y: (x, tf.gather(label_map, y)), num_parallel_calls=tf.data.AUTOTUNE
    )
    n_real = int(real_ds.cardinality().numpy()) if real_ds.cardinality() >= 0 else -1
    print(f"Dùng ảnh thật từ real_train: các lớp {present}, lặp {REAL_REPEAT} lần")
    train_ds = (
        train_ds.unbatch()
        .concatenate(real_ds.repeat(REAL_REPEAT))
        .shuffle(1500, seed=42, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
    )

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# ---------------------------------------------------------------
# 3. Mô hình: MobileNetV2 pretrained ImageNet + augmentation mạnh
#    Augmentation chỉ chạy khi huấn luyện, tự tắt khi dự đoán.
# ---------------------------------------------------------------
base = keras.applications.MobileNetV2(
    input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet"
)
base.trainable = False

inputs = keras.Input(shape=IMAGE_SIZE + (3,))          # ảnh 0-255
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.12)(x)
x = layers.RandomZoom((-0.3, 0.1))(x)
x = layers.RandomTranslation(0.1, 0.1)(x)
x = layers.RandomBrightness(0.3, value_range=(0, 255))(x)
x = layers.RandomContrast(0.3)(x)
x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)      # MobileNetV2 cần [-1, 1]
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inputs, outputs)


def make_callbacks(tag: str, best_so_far=None):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(OUTPUT / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            initial_value_threshold=best_so_far,  # không ghi đè model tốt hơn
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        keras.callbacks.CSVLogger(str(OUTPUT / f"training_log_{tag}.csv")),
    ]


# ---------------------------------------------------------------
# 4. Giai đoạn 1: chỉ train phần đầu phân loại
# ---------------------------------------------------------------
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()
history_head = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
                         callbacks=make_callbacks("head"))
best_val_loss = min(history_head.history["val_loss"])

# ---------------------------------------------------------------
# 5. Giai đoạn 2: tinh chỉnh các lớp cuối (giữ BatchNorm đóng băng)
# ---------------------------------------------------------------
base.trainable = True
for layer in base.layers[:FINE_TUNE_FROM]:
    layer.trainable = False
for layer in base.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE,
          callbacks=make_callbacks("fine", best_so_far=best_val_loss))

print("\nĐã hoàn thành huấn luyện.")
print("Mô hình được lưu tại:", OUTPUT / "best_model.keras")
