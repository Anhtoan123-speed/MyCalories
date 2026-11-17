"""Tiền xử lý ảnh DÙNG CHUNG cho predict.py, train.py và app.

Quy tắc phải khớp lúc huấn luyện:
  1. Sửa hướng ảnh theo EXIF, đổi sang RGB
  2. Cắt vuông ở giữa (KHÔNG bóp méo tỉ lệ)
  3. Resize bilinear (có khử răng cưa) về IMAGE_SIZE
  4. Giữ nguyên thang 0-255: mô hình tự chuẩn hóa bên trong (Rescaling)
"""
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

IMAGE_SIZE = (224, 224)


def center_crop_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return image.crop((left, top, left + side, top + side))


def to_model_array(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = center_crop_square(image)
    array = tf.image.resize(
        np.asarray(image, dtype="float32"), IMAGE_SIZE, method="bilinear", antialias=True
    ).numpy()
    return array  # (224, 224, 3), giá trị 0-255


def predict_tta(models, image: Image.Image, weights=None) -> np.ndarray:
    """Nhiều góc nhìn x nhiều model, lấy trung bình có trọng số."""
    base = ImageOps.exif_transpose(image).convert("RGB")
    w, h = base.size

    # bản zoom vào giữa 85%
    cw, ch = int(w * 0.85), int(h * 0.85)
    l, t = (w - cw) // 2, (h - ch) // 2
    zoom = to_model_array(base.crop((l, t, l + cw, t + ch)))

    crop = to_model_array(base)                       # cắt vuông giữa
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (128, 128, 128))
    canvas.paste(base, ((s - w) // 2, (s - h) // 2))
    pad = to_model_array(canvas)                      # đủ khung

    views = np.stack([crop, crop[:, ::-1], zoom, pad, pad[:, ::-1]])
    weights = weights or [1.0] * len(models)
    total = sum(wt * m.predict(views, verbose=0).mean(axis=0)
                for m, wt in zip(models, weights))
    return total / sum(weights)