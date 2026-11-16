from pathlib import Path
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Cấu hình đường dẫn và tham số
ROOT = Path(__file__).resolve().parent
DATA = ROOT

OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30  # Chạy thử 2 lượt trước; chạy ổn thì đổi thành 30

keras.utils.set_random_seed(42)

# 2. Đọc tập huấn luyện
train_ds = keras.utils.image_dataset_from_directory(
    str(DATA / "train"),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=42,
)

class_names = train_ds.class_names
print("Các loại rau củ:", class_names)

# 3. Đọc tập validation, giữ cùng thứ tự nhãn
val_ds = keras.utils.image_dataset_from_directory(
    str(DATA / "validation"),
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

# Lưu nhãn để dùng khi nhận diện ảnh mới
(OUTPUT / "class_names.json").write_text(
    json.dumps(class_names, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# 4. Xây dựng CNN từ đầu, không dùng trọng số có sẵn
model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(len(class_names), activation="softmax"),
])

# 5. Cấu hình huấn luyện
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(OUTPUT / "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
    ),
    keras.callbacks.CSVLogger(
        str(OUTPUT / "training_log.csv")
    ),
]

# 6. Bắt đầu huấn luyện
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

print("\nĐã hoàn thành huấn luyện.")
print("Mô hình được lưu tại:", OUTPUT / "best_model.keras")
print("Nhật ký được lưu tại:", OUTPUT / "training_log.csv")