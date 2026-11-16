# -*- coding: utf-8 -*-
"""Ứng dụng giảm cân.

Luồng sử dụng:
  1. Mục tiêu  : nhập hồ sơ -> BMI, BMR, TDEE, calo mục tiêu mỗi ngày.
  2. Nhận diện : chụp/tải ảnh rau củ -> CNN nhận diện -> thêm vào danh sách nguyên liệu.
  3. Món ăn    : gợi ý món theo nguyên liệu đang có, tính calo + đạm/tinh bột/béo.
  4. Thực đơn  : thêm món vào từng bữa, so với calo mục tiêu, tải CSV.
  5. Cân nặng  : ghi nhận cân nặng theo ngày, xem tiến độ.

Chạy:  python -m streamlit run app_mycalories_olivine.py
"""
from __future__ import annotations

import base64
import csv
import hashlib
import html
import io
import json
import random
import re
import uuid
from datetime import date
from pathlib import Path
from recipes_extra import MORE_RECIPES as EXTRA_RECIPES
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow import keras

# Gộp công thức, bỏ món trùng tên
RECIPES, _seen_names = [], set()
for _recipe in [*EXTRA_RECIPES]:
    if _recipe["name"].casefold() not in _seen_names:
        _seen_names.add(_recipe["name"].casefold())
        RECIPES.append(_recipe)
RECIPE_BY_NAME = {r["name"]: r for r in RECIPES}

# st.set_page_config phải là lệnh Streamlit đầu tiên
st.set_page_config(page_title="MyCalories", page_icon="🥗", layout="wide")

# ============================================================
# 1. CẤU HÌNH VÀ DỮ LIỆU
# ============================================================
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
HERO_IMAGE = ROOT / "assets" / "healthy-lifestyle.jpg"
DATA_DIR = ROOT / "data"
LEGACY_MENU_FILE = ROOT / "daily_menu.json"  # file của phiên bản cũ
MODEL_FILE = OUTPUT / "best_model.keras"
LABEL_FILE = OUTPUT / "class_names.json"
YOLO_FILE = OUTPUT / "vegetable_yolo.pt"
YOLO_LABELS = {
    "broccoli": "Broccoli",
    "carrot": "Carrot",
    "onion": "Onion",
    "pepper": "Capsicum",
    "tomato": "Tomato",
}

IMAGE_SIZE = (128, 128)  # phải khớp với train.py
CONFIDENCE_OK = 0.60     # dưới mức này thì cảnh báo "chưa chắc chắn"
MEALS = ("Bữa sáng", "Bữa trưa", "Bữa tối", "Bữa phụ")
MAIN_MEAL_SHARE = {"Bữa sáng": 0.30, "Bữa trưa": 0.40, "Bữa tối": 0.30}

VI_NAMES = {
    # 15 nhãn của mô hình
    "Bean": "Đậu que",
    "Bitter_Gourd": "Khổ qua",
    "Bottle_Gourd": "Bầu",
    "Brinjal": "Cà tím",
    "Broccoli": "Bông cải xanh",
    "Cabbage": "Bắp cải",
    "Capsicum": "Ớt chuông",
    "Carrot": "Cà rốt",
    "Cauliflower": "Súp lơ trắng",
    "Cucumber": "Dưa chuột",
    "Papaya": "Đu đủ",
    "Potato": "Khoai tây",
    "Pumpkin": "Bí đỏ",
    "Radish": "Củ cải",
    "Tomato": "Cà chua",
    # thực phẩm bổ sung (không do mô hình nhận diện)
    "Rice": "Cơm trắng (đã nấu)",
    "Chicken_Breast": "Ức gà (đã nấu)",
    "Egg": "Trứng gà",
    "Tofu": "Đậu hũ",
    "Oats": "Yến mạch (khô)",
    "Sweet_Potato": "Khoai lang",
    "Shrimp": "Tôm (đã bóc vỏ, sống)",
    "Pork_Lean": "Thịt nạc heo (sống)",
    "Fish_Fillet": "Cá phi lê trắng (sống)",
    "Mushroom": "Nấm",
    "Water_Spinach": "Rau muống",
    "Brown_Rice": "Cơm gạo lứt (đã nấu)",
    "Oil": "Dầu ăn",
    "Onion": "Hành tây",
}
VEGETABLES = tuple(list(VI_NAMES)[:15])
EXTRA_FOODS = (
    "Rice", "Brown_Rice", "Chicken_Breast", "Egg", "Tofu", "Oats", "Sweet_Potato",
    "Shrimp", "Pork_Lean", "Fish_Fillet", "Mushroom", "Water_Spinach",
)
PANTRY_STAPLES = ("Oil",)  # mặc định nhà nào cũng có, không bắt buộc chọn

# Năng lượng và chất đa lượng trên 100 g: (kcal, đạm, tinh bột, béo).
# Rau củ: tính theo khối lượng sống, ăn được. Cơm/ức gà: khối lượng đã nấu.
# Số liệu gần đúng theo USDA; nên đối chiếu Bảng thành phần thực phẩm Việt Nam.
FOODS = {
    "Bean": (31, 1.8, 7.0, 0.2),
    "Bitter_Gourd": (17, 1.0, 3.7, 0.2),
    "Bottle_Gourd": (14, 0.6, 3.4, 0.0),
    "Brinjal": (25, 1.0, 5.9, 0.2),
    "Broccoli": (34, 2.8, 6.6, 0.4),
    "Cabbage": (25, 1.3, 5.8, 0.1),
    "Capsicum": (27, 1.0, 6.3, 0.2),
    "Carrot": (41, 0.9, 9.6, 0.2),
    "Cauliflower": (25, 1.9, 5.0, 0.3),
    "Cucumber": (15, 0.7, 3.6, 0.1),
    "Papaya": (43, 0.5, 10.8, 0.3),
    "Potato": (77, 2.0, 17.5, 0.1),
    "Pumpkin": (26, 1.0, 6.5, 0.1),
    "Radish": (16, 0.7, 3.4, 0.1),
    "Tomato": (18, 0.9, 3.9, 0.2),
    "Rice": (130, 2.7, 28.0, 0.3),
    "Chicken_Breast": (165, 31.0, 0.0, 3.6),
    "Egg": (143, 12.6, 0.7, 9.5),
    "Tofu": (76, 8.1, 1.9, 4.8),
    "Oats": (389, 16.9, 66.3, 6.9),
    "Sweet_Potato": (86, 1.6, 20.1, 0.1),
    "Shrimp": (85, 20.1, 0.0, 0.5),         # sống, đã bóc vỏ
    "Pork_Lean": (120, 20.9, 0.0, 3.5),     # thịt nạc sống
    "Fish_Fillet": (96, 20.1, 0.0, 1.7),    # phi lê cá trắng sống
    "Mushroom": (22, 3.1, 3.3, 0.3),
    "Water_Spinach": (19, 2.6, 3.1, 0.2),
    "Brown_Rice": (123, 2.7, 25.6, 1.0),    # đã nấu
    "Oil": (884, 0.0, 0.0, 100.0),
}

# Công thức nằm trong recipes.py (thêm món mới ở đó).

# ---------- Hồ sơ người dùng ----------
ACTIVITY = {
    1.2: "Ít vận động (ngồi nhiều)",
    1.375: "Nhẹ (tập 1–3 buổi/tuần)",
    1.55: "Vừa (tập 3–5 buổi/tuần)",
    1.725: "Nhiều (tập 6–7 buổi/tuần)",
}
RATES = {
    0.0: "Giữ cân",
    0.25: "Nhẹ nhàng: giảm 0,25 kg/tuần",
    0.5: "Vừa phải: giảm 0,5 kg/tuần (khuyên dùng)",
    0.75: "Nhanh: giảm 0,75 kg/tuần",
}
PROFILE_DEFAULTS = {
    "sex": "Nữ", "age": 25, "height": 160.0, "weight": 60.0,
    "activity": 1.375, "rate": 0.5, "goal_weight": 55.0,
}
PROFILE_KEYS = {
    "sex": "p_sex", "age": "p_age", "height": "p_height", "weight": "p_weight",
    "activity": "p_activity", "rate": "p_rate", "goal_weight": "p_goal",
}


# ============================================================
# 2. HÀM TÍNH TOÁN (thuần, không phụ thuộc giao diện)
# ============================================================
def vi(key: str) -> str:
    return VI_NAMES.get(key, key)


def nutrition(items: dict) -> tuple[dict, list]:
    """Tổng kcal/đạm/tinh bột/béo cho dict {nguyên liệu: gram}."""
    total = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    unknown = []
    for key, grams in items.items():
        if key not in FOODS:
            unknown.append(key)
            continue
        kcal, protein, carbs, fat = FOODS[key]
        factor = grams / 100
        total["kcal"] += kcal * factor
        total["protein"] += protein * factor
        total["carbs"] += carbs * factor
        total["fat"] += fat * factor
    return total, unknown


def per_serving(total: dict, servings: float) -> dict:
    servings = max(servings, 1)
    return {k: v / servings for k, v in total.items()}


def recipe_per_serving(recipe: dict) -> dict:
    total, _ = nutrition(recipe["items"])
    return per_serving(total, recipe["servings"])


def recipe_status(recipe: dict, available: set) -> tuple[list, list]:
    """Trả về (nguyên liệu chính cần có, nguyên liệu còn thiếu)."""
    needed = [k for k in recipe["items"] if k not in PANTRY_STAPLES]
    missing = [k for k in needed if k not in available]
    return needed, missing


def bmi_category(bmi: float) -> str:
    """Phân loại theo chuẩn WHO khu vực châu Á - Thái Bình Dương."""
    if bmi < 18.5:
        return "Gầy"
    if bmi < 23:
        return "Bình thường"
    if bmi < 25:
        return "Thừa cân"
    return "Béo phì"


def compute_plan(profile: dict) -> dict:
    weight, height, age = profile["weight"], profile["height"], profile["age"]
    bmi = weight / ((height / 100) ** 2)
    # Công thức Mifflin-St Jeor
    bmr = 10 * weight + 6.25 * height - 5 * age + (5 if profile["sex"] == "Nam" else -161)
    tdee = bmr * profile["activity"]

    notes = []
    rate = profile["rate"]
    if bmi < 18.5 and rate > 0:
        rate = 0.0
        notes.append("BMI của bạn dưới 18,5 nên ứng dụng không đề xuất giảm cân. "
                     "Mức calo bên dưới là mức duy trì.")

    deficit = rate * 7700 / 7  # ~7700 kcal cho 1 kg mỡ
    floor = 1500 if profile["sex"] == "Nam" else 1200
    target = tdee - deficit
    floored = False
    if rate > 0 and target < floor:
        target, floored = floor, True
        notes.append(f"Mức calo tính ra thấp hơn ngưỡng an toàn {floor} kcal nên đã được nâng lên "
                     f"{floor} kcal. Tốc độ giảm thực tế sẽ chậm hơn dự kiến.")

    height_m2 = (height / 100) ** 2
    goal = profile["goal_weight"]
    weeks = None
    if rate > 0 and goal < weight:
        weeks = (weight - goal) / rate
    if rate > 0 and goal >= weight:
        notes.append("Cân nặng mục tiêu chưa thấp hơn cân nặng hiện tại.")
    if goal / height_m2 < 18.5:
        notes.append("Cân nặng mục tiêu có BMI dưới 18,5 (mức gầy). Hãy đặt mục tiêu cao hơn.")

    return {
        "bmi": bmi, "category": bmi_category(bmi), "bmr": bmr, "tdee": tdee,
        "target": round(target / 10) * 10, "rate": rate, "floored": floored,
        "healthy_low": 18.5 * height_m2, "healthy_high": 22.9 * height_m2,
        "protein_ref": round(weight * 1.2), "weeks": weeks, "notes": notes,
    }


def make_entry(meal: str, name: str, portions: float, per: dict) -> dict:
    return {
        "id": uuid.uuid4().hex[:8], "meal": meal, "name": name,
        "portions": round(portions, 2),
        "kcal": round(per["kcal"] * portions, 1),
        "protein": round(per["protein"] * portions, 1),
        "carbs": round(per["carbs"] * portions, 1),
        "fat": round(per["fat"] * portions, 1),
    }


GRID_MULTIPLIERS = (1.0,)  # gợi ý luôn là 1 phần


def _candidates(meal: str, budget: float, available: set, excluded: set, used_items: set) -> list:
    """Các món phù hợp bữa `meal`, xếp theo độ gần với calo còn lại (tốt nhất đứng đầu)."""
    found = []
    for recipe in RECIPES:
        if recipe["name"] in excluded or meal not in recipe["meals"]:
            continue
        if recipe_status(recipe, available)[1]:
            continue
        per = recipe_per_serving(recipe)
        main_items = set(recipe["items"]) - set(PANTRY_STAPLES)
        overlap = len(main_items & used_items) / max(len(main_items), 1)
        scored = []
        for m in GRID_MULTIPLIERS:
            diff = per["kcal"] * m - budget
            scored.append((abs(diff) * (1.5 if diff > 0 else 1.0), m))
        score, mult = min(scored)
        found.append((score + overlap * budget * 0.35, recipe, mult, per))
    found.sort(key=lambda c: c[0])
    return found


def suggest_day(available: set, target_kcal: float, menu: list | None = None,
                exclude: set | None = None, seed: str = "") -> list:
    """Chọn món cho sáng/trưa/tối sao cho calo gần với phần ngân sách của từng bữa.

    Không gợi ý lại món đã có trong thực đơn hoặc món nằm trong `exclude` (các lần gợi ý trước);
    trừ calo đã ăn của từng bữa. Có `seed` thì chọn ngẫu nhiên trong vài món phù hợp nhất (xáo trộn).
    """
    menu = menu or []
    rng = random.Random(seed)
    used = {e["name"] for e in menu} | set(exclude or ())
    eaten = {}
    for e in menu:
        eaten[e["meal"]] = eaten.get(e["meal"], 0) + e["kcal"]
    used_items = set()
    for name in used:
        if name in RECIPE_BY_NAME:
            used_items |= set(RECIPE_BY_NAME[name]["items"]) - set(PANTRY_STAPLES)
    plan = []
    for meal, share in MAIN_MEAL_SHARE.items():
        full = target_kcal * share
        budget = full - eaten.get(meal, 0)
        if budget < full * 0.25:  # bữa này đã đủ calo
            continue
        cands = _candidates(meal, budget, available, used, used_items)
        if not cands:
            continue
        _, recipe, multiplier, per = rng.choice(cands[:3]) if seed else cands[0]
        used.add(recipe["name"])
        used_items |= set(recipe["items"]) - set(PANTRY_STAPLES)
        plan.append((meal, recipe["name"], multiplier, per))
    return plan


APP_STYLE = """
<style>
[data-testid="stTabs"] [role="tab"] {
    padding: 0.6rem 0.35rem;
    font-size: 19px;
}

[data-testid="stTabs"] [role="tab"] p {
    font-size: 18px !important;
    line-height: 1.45;
}

.stMainBlockContainer, [data-testid="stAppViewBlockContainer"] {
    padding-top: 4.4rem;
}

    .mc-hero {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 0.86fr);
        align-items: center;
        gap: clamp(1.25rem, 4vw, 4.5rem);
        padding: clamp(1.4rem, 4vw, 3.2rem);
        margin-bottom: 0.5rem;
        border: 1px solid #D3DFC5;
        border-radius: 1.5rem;
        background: linear-gradient(115deg, #F0F5E6 20%, #E1EDCF 100%);
    }

    .mc-hero .mc-title {
        margin: 0;
        padding: 0;
        color: #273423;
        font-size: clamp(2.1rem, 4.8vw, 4.6rem);
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.045em;
        overflow-wrap: anywhere;
    }

    .mc-hero .mc-tagline {
        margin: 1rem 0 0;
        color: #465A38;
        font-size: clamp(1rem, 1.55vw, 1.2rem);
        line-height: 1.55;
    }

    .mc-hero .mc-tagline::before {
        display: block;
        width: 3rem;
        height: 0.24rem;
        margin: 0 0 0.9rem;
        border-radius: 1rem;
        background: #A2CA6C;
        content: "";
    }

    .mc-hero-art {
        display: block;
        width: 100%;
        max-width: 28rem;
        height: auto;
        justify-self: end;
        border: 1px solid #D3DFC5;
        border-radius: 1rem;
        box-shadow: 0 12px 28px #42552C24;
    }

    [data-testid="stSidebar"] .mc-sidebar-title {
        margin: 0 0 0.6rem;
        padding: 0;
        color: #304525;
        font-size: 1.55rem;
    }

    [data-testid="stMetric"] {
        padding: 0.9rem 1rem;
        background: #F0F5E7;
        border: 1px solid #D3DFC5;
        border-radius: 0.9rem;
    }

    [data-testid="stBaseButton-primary"] {
        background: #A2CA6C;
        border-color: #8DB452;
        color: #21301B;
    }

    [data-testid="stBaseButton-primary"]:hover {
        background: #B4D484;
        border-color: #789A47;
        color: #21301B;
    }

    @media (max-width: 700px) {
        .mc-hero {
            grid-template-columns: minmax(0, 1fr);
            padding: 1.35rem;
            border-radius: 1.15rem;
        }

        .mc-hero-art {
            justify-self: start;
        }
    }
</style>
"""


MENU_GRID_CSS = """<style>
.mg-title{text-align:center;color:#49632E;font-weight:800;font-size:1.7rem;letter-spacing:.5px;margin:.2rem 0 .8rem;box-sizing: border-box;
padding-inline-start: 14%;}
table.mg{width:100%;table-layout:fixed;border-collapse:separate;border-spacing:12px 0}
table.mg th{color:#49632E;font-weight:700;padding:4px 6px 10px;text-align:center}
table.mg td{background:#EAF1DF;color:#273423;text-align:center;padding:14px 12px;border-bottom:2px solid #F8FAF4;
  vertical-align:middle;font-size:1rem;line-height:1.35}
table.mg td.lbl{background:transparent;color:#49632E;font-weight:700;border-bottom:0;width:13%}
table.mg td.top{border-radius:26px 26px 0 0}
table.mg td.bot{border-radius:0 0 26px 26px;border-bottom:0}
table.mg td.tot{background:#D8E8C1;font-weight:700}
table.mg td.none{color:#56654F;font-style:italic}
table.mg small{display:block;color:#465A38;margin-top:3px}
</style>"""


def render_menu_grid(headers: list, rows: list, title: str = "") -> str:
    """Bảng thực đơn dạng thẻ xanh. rows: [(nhãn bữa, [ô ...])]; ô: {"text", "kcal", "state"} hoặc None."""
    out = [MENU_GRID_CSS]
    if title:
        out.append(f'<div class="mg-title">{html.escape(title)}</div>')
    out.append('<table class="mg"><colgroup><col style="width:13%"></colgroup>')
    if headers:
        out.append("<tr><th></th>")
        out += [f"<th>{html.escape(h)}</th>" for h in headers]
        out.append("</tr>")
    for i, (label, cells) in enumerate(rows):
        pos = " top" if i == 0 else (" bot" if i == len(rows) - 1 else "")
        out.append(f'<tr><td class="lbl">{html.escape(label)}</td>')
        for cell in cells:
            cls, text, sub = pos, "—", ""
            if cell and cell.get("text"):
                text = html.escape(cell["text"])
                if cell.get("kcal"):
                    sub = f"<small>≈ {cell['kcal']:.0f} kcal</small>"
            else:
                cls += " none"
            if cell and cell.get("state") == "total":
                cls += " tot"
            out.append(f'<td class="{cls.strip()}">{text}{sub}</td>')
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


# ============================================================
# 3. LƯU TRỮ DỮ LIỆU (tự động lưu, mỗi hồ sơ một file)
# ============================================================
def slugify(name: str) -> str:
    # \W (Unicode) giữ chữ cái tiếng Việt, loại bỏ dấu chấm và dấu gạch chéo
    return re.sub(r"\W+", "_", name.strip(), flags=re.UNICODE).strip("_")[:40] or "mac_dinh"


def sanitize_profile(raw) -> dict:
    p = dict(PROFILE_DEFAULTS)
    if isinstance(raw, dict):
        p.update({k: v for k, v in raw.items() if k in p})
    try:
        p["age"] = min(max(int(p["age"]), 18), 100)
        p["height"] = min(max(float(p["height"]), 120.0), 220.0)
        p["weight"] = min(max(float(p["weight"]), 30.0), 300.0)
        p["goal_weight"] = min(max(float(p["goal_weight"]), 30.0), 300.0)
        p["activity"] = float(p["activity"])
        p["rate"] = float(p["rate"])
    except (TypeError, ValueError):
        return dict(PROFILE_DEFAULTS)
    if p["sex"] not in ("Nữ", "Nam"):
        p["sex"] = PROFILE_DEFAULTS["sex"]
    if p["activity"] not in ACTIVITY:
        p["activity"] = PROFILE_DEFAULTS["activity"]
    if p["rate"] not in RATES:
        p["rate"] = PROFILE_DEFAULTS["rate"]
    return p


def clean_entry(item) -> dict | None:
    """Chuẩn hóa một món trong thực đơn, kể cả định dạng của phiên bản cũ."""
    if not isinstance(item, dict):
        return None
    try:
        return {
            "id": str(item.get("id") or uuid.uuid4().hex[:8]),
            "meal": item.get("meal") or item.get("Bữa") or "Bữa phụ",
            "name": str(item.get("name") or item.get("Món ăn") or "Món"),
            "portions": float(item.get("portions", item.get("Số khẩu phần", 1))),
            "kcal": float(item.get("kcal", item.get("Năng lượng (kcal)", 0))),
            "protein": float(item.get("protein", 0)),
            "carbs": float(item.get("carbs", 0)),
            "fat": float(item.get("fat", 0)),
        }
    except (TypeError, ValueError):
        return None


def load_data(path: Path) -> dict:
    data = {"profile": dict(PROFILE_DEFAULTS), "pantry": [], "menus": {}, "weights": []}
    try:
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            data["profile"] = sanitize_profile(saved.get("profile"))
            data["pantry"] = [x for x in saved.get("pantry", []) if isinstance(x, str)]
            for day, items in dict(saved.get("menus", {})).items():
                cleaned = [e for e in map(clean_entry, items) if e]
                if cleaned:
                    data["menus"][day] = cleaned
            data["weights"] = [
                {"date": str(w["date"]), "kg": float(w["kg"])}
                for w in saved.get("weights", []) if "date" in w and "kg" in w
            ]
        elif LEGACY_MENU_FILE.is_file() and not any(DATA_DIR.glob("*.json")):
            # Nhập thực đơn của phiên bản cũ vào ngày hôm nay
            old = json.loads(LEGACY_MENU_FILE.read_text(encoding="utf-8"))
            cleaned = [e for e in map(clean_entry, old) if e]
            if cleaned:
                data["menus"][date.today().isoformat()] = cleaned
    except (OSError, ValueError, TypeError, AttributeError):
        st.warning("Không đọc được dữ liệu đã lưu, ứng dụng dùng dữ liệu trống.")
    return data


def save_data() -> None:
    path = Path(st.session_state["loaded_path"])
    try:
        DATA_DIR.mkdir(exist_ok=True)
        tmp = path.with_suffix(".tmp")  # ghi file tạm rồi thay thế để tránh file ghi dở
        tmp.write_text(
            json.dumps(st.session_state["data"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError as error:
        st.error(f"Không lưu được dữ liệu: {error}")


def push_profile_to_state(profile: dict) -> None:
    for field, key in PROFILE_KEYS.items():
        st.session_state[key] = profile[field]
    st.session_state["log_weight"] = profile["weight"]


# ============================================================
# 4. MÔ HÌNH NHẬN DIỆN
# ============================================================
@st.cache_resource(show_spinner="Đang nạp mô hình...")
def load_resources():
    model = keras.models.load_model(str(MODEL_FILE))
    names = json.loads(LABEL_FILE.read_text(encoding="utf-8"))
    return model, names


def load_image(raw: bytes) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")


@st.cache_data(show_spinner=False, max_entries=128)
def predict_scores(raw: bytes) -> list:
    model, _ = load_resources()
    image = load_image(raw)

    array = tf.image.resize(
        np.asarray(image, dtype="float32"),
        IMAGE_SIZE,
        method="bilinear",
    ).numpy()

    return model.predict(array[None, ...], verbose=0)[0].tolist()


@st.cache_resource(show_spinner="Đang nạp mô hình YOLO...")
def load_yolo(weights_version: int):
    from ultralytics import YOLO

    return YOLO(str(YOLO_FILE), task="detect")


@st.cache_data(show_spinner=False, max_entries=32)
def predict_objects(raw: bytes, weights_version: int) -> tuple[np.ndarray, dict[str, float]]:
    image = load_image(raw)
    detector = load_yolo(weights_version)
    result = detector.predict(image, imgsz=640, conf=0.25, device="cpu", verbose=False)[0]

    confidence_by_class = {}
    if result.boxes is not None:
        for class_id, confidence in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            label = result.names[int(class_id)]
            confidence_by_class[label] = max(confidence_by_class.get(label, 0.0), confidence)

    return result.plot()[..., ::-1].copy(), confidence_by_class


# ============================================================
# 5. CALLBACK (chạy trước khi giao diện vẽ lại nên trạng thái luôn mới)
# ============================================================
def flash(message: str) -> None:
    st.session_state["flash"] = message


def add_entries_cb(day_key: str, entries: list, message: str, skip_dupes: bool = False) -> None:
    menu = st.session_state["data"]["menus"].setdefault(day_key, [])
    if skip_dupes:
        existing = {(e["meal"], e["name"]) for e in menu}
        entries = [e for e in entries if (e["meal"], e["name"]) not in existing]
        if not entries:
            flash("Các món gợi ý đã có trong thực đơn.")
            return
    menu.extend(entries)
    save_data()
    flash(message)


def clear_menu_cb(day_key: str, widget_key: str) -> None:
    menus = st.session_state["data"]["menus"]
    count = len(menus.pop(day_key, []))
    st.session_state[widget_key] = []
    save_data()
    flash(f"Đã xóa toàn bộ {count} món trong thực đơn ngày này.")


def shuffle_suggestion_cb(state_key: str, shown: list) -> None:
    """Ghi nhớ các món đã gợi ý rồi sang lần gợi ý kế tiếp (không lặp lại món cũ)."""
    seen = st.session_state.setdefault("suggest_seen", {}).setdefault(state_key, set())
    seen.update(shown)
    rounds = st.session_state.setdefault("suggest_round", {})
    rounds[state_key] = rounds.get(state_key, 0) + 1


def remove_entries_cb(day_key: str, widget_key: str) -> None:
    ids = set(st.session_state.get(widget_key, []))
    menus = st.session_state["data"]["menus"]
    menus[day_key] = [e for e in menus.get(day_key, []) if e["id"] not in ids]
    if not menus[day_key]:
        del menus[day_key]
    st.session_state[widget_key] = []
    save_data()
    flash(f"Đã xóa {len(ids)} món.")


def add_detected_cb(fix_keys: list) -> None:
    current = list(st.session_state.get("ingredients", []))
    for key in fix_keys:
        value = st.session_state.get(key)
        labels = [value] if isinstance(value, str) else value or []
        for label in labels:
            if label not in current:
                current.append(label)
    st.session_state["ingredients"] = current
    flash("Đã thêm nguyên liệu nhận diện được vào danh sách.")


def clear_pantry_cb() -> None:
    st.session_state["ingredients"] = []


def log_weight_cb(day_key: str) -> None:
    kg = round(float(st.session_state["log_weight"]), 1)
    weights = [w for w in st.session_state["data"]["weights"] if w["date"] != day_key]
    weights.append({"date": day_key, "kg": kg})
    st.session_state["data"]["weights"] = sorted(weights, key=lambda w: w["date"])
    save_data()
    flash(f"Đã ghi nhận {kg} kg.")


def delete_weight_cb(day_key: str) -> None:
    data = st.session_state["data"]
    data["weights"] = [w for w in data["weights"] if w["date"] != day_key]
    save_data()


def use_latest_weight_cb() -> None:
    weights = st.session_state["data"]["weights"]
    if weights:
        latest = max(weights, key=lambda w: w["date"])
        st.session_state["p_weight"] = min(max(float(latest["kg"]), 30.0), 300.0)
        flash("Đã cập nhật cân nặng trong hồ sơ.")


# ============================================================
# 6. KHỞI TẠO: nạp mô hình, hồ sơ, dữ liệu
# ============================================================
model, class_names = None, list(VEGETABLES)
if MODEL_FILE.is_file() and LABEL_FILE.is_file():
    try:
        model, class_names = load_resources()
    except Exception as error:  # lỗi phiên bản Keras/TensorFlow rất đa dạng
        st.sidebar.error(f"Không nạp được mô hình: {error}")
else:
    st.sidebar.warning("Thiếu best_model.keras hoặc class_names.json trong thư mục outputs. "
                       "Chạy train.py trước để dùng tính năng nhận diện ảnh.")

PANTRY_OPTIONS = list(dict.fromkeys(list(class_names) + list(EXTRA_FOODS)))

st.markdown(APP_STYLE, unsafe_allow_html=True)
st.sidebar.markdown('<h2 class="mc-sidebar-title">Hồ sơ</h2>', unsafe_allow_html=True)
profile_name = st.sidebar.text_input(
    "Tên hồ sơ", value="Mặc định", key="profile_name",
    help="Mỗi tên hồ sơ lưu dữ liệu riêng trong thư mục data/.",
)
view_date = st.sidebar.date_input("Ngày ghi thực đơn", value=date.today(), key="view_date", format="DD/MM/YYYY")
if not isinstance(view_date, date):
    view_date = date.today()
day_key = view_date.isoformat()

profile_path = DATA_DIR / f"{slugify(profile_name)}.json"
if st.session_state.get("loaded_path") != str(profile_path):
    st.session_state["loaded_path"] = str(profile_path)
    st.session_state["data"] = load_data(profile_path)
    push_profile_to_state(st.session_state["data"]["profile"])
    st.session_state["ingredients"] = [
        x for x in st.session_state["data"]["pantry"] if x in PANTRY_OPTIONS
    ]
data = st.session_state["data"]
sex = st.sidebar.selectbox("Giới tính", ["Nữ", "Nam"], key="p_sex")
height = st.sidebar.number_input("Chiều cao (cm)", min_value=120.0, max_value=220.0, step=0.5, key="p_height")
age = st.sidebar.number_input("Tuổi (từ 18 trở lên)", min_value=18, max_value=100, step=1, key="p_age")
weight = st.sidebar.number_input("Cân nặng hiện tại (kg)", min_value=30.0, max_value=300.0, step=0.1, key="p_weight")
st.sidebar.divider()
st.sidebar.caption(
    "⚠️ Ứng dụng chỉ mang tính tham khảo, không thay thế tư vấn của bác sĩ hoặc chuyên gia dinh dưỡng. "
    "Nếu bạn đang mang thai, cho con bú, có bệnh nền hoặc rối loạn ăn uống, hãy hỏi chuyên gia trước khi giảm cân."
)

hero_art = ""
if HERO_IMAGE.is_file():
    encoded_image = base64.b64encode(HERO_IMAGE.read_bytes()).decode("ascii")
    hero_art = (
        f'<img class="mc-hero-art" src="data:image/jpeg;base64,{encoded_image}" '
        'alt="Healthy lifestyle: minh họa cô gái ngồi thiền giữa rau củ">'
    )

st.markdown(
    '<section class="mc-hero" aria-labelledby="mc-title">'
    '<div><h1 class="mc-title" id="mc-title">MyCalories</h1>'
    '<p class="mc-tagline">Ăn thông minh - sống khỏe mỗi ngày</p></div>'
    f'{hero_art}</section>',
    unsafe_allow_html=True,
)
message = st.session_state.pop("flash", None)
if message:
    st.toast(message, icon="✅")

tab_profile, tab_detect, tab_recipes, tab_menu, tab_weight = st.tabs(
    ["🎯 Mục tiêu", "📷 Nhận diện", "🍳 Món ăn", "📅 Thực đơn", "⚖️ Cân nặng"]
)

# ============================================================
# TAB 1: MỤC TIÊU
# ============================================================
def suggest_rate_on_activity_change() -> None:
    profile = {
        field: st.session_state[key]
        for field, key in PROFILE_KEYS.items()
    }

    recommended_rate = 0.0
    goal_bmi = profile["goal_weight"] / (profile["height"] / 100) ** 2
    candidates = (0.5, 0.25) if profile["activity"] == 1.725 else (0.25,)

    if profile["goal_weight"] < profile["weight"] and goal_bmi >= 18.5:
        for candidate in candidates:
            plan = compute_plan({**profile, "rate": candidate})
            if plan["rate"] == candidate and not plan["floored"]:
                recommended_rate = candidate
                break

    st.session_state["p_rate"] = recommended_rate
with tab_profile:
    st.subheader("Mục tiêu calo")
    activity = st.selectbox(
        "Mức vận động",
        list(ACTIVITY),
        format_func=ACTIVITY.get,
        key="p_activity",
        on_change=suggest_rate_on_activity_change,
    )
    rate = st.selectbox("Mục tiêu", list(RATES), format_func=RATES.get, key="p_rate")
    goal_weight = st.number_input("Cân nặng mục tiêu (kg)", min_value=30.0, max_value=300.0, step=0.1, key="p_goal")

    profile = {"sex": sex, "age": int(age), "height": float(height), "weight": float(weight),
               "activity": float(activity), "rate": float(rate), "goal_weight": float(goal_weight)}
    if profile != data["profile"]:  # tự động lưu khi hồ sơ thay đổi
        data["profile"] = profile
        save_data()

    plan = compute_plan(profile)

    m1, m2, m3 = st.columns(3)
    m1.metric("BMI", f"{plan['bmi']:.1f}", plan["category"], delta_color="off")
    m2.metric("BMR (năng lượng nghỉ)", f"{plan['bmr']:.0f} kcal")
    m3.metric("TDEE (tiêu hao mỗi ngày)", f"{plan['tdee']:.0f} kcal")

    st.success(f"**Mục tiêu ăn mỗi ngày: khoảng {plan['target']:.0f} kcal** "
               f"(đạm tham khảo ≥ {plan['protein_ref']} g/ngày)")
    st.caption(f"Cân nặng khỏe mạnh theo BMI châu Á (18,5–22,9) với chiều cao của bạn: "
               f"{plan['healthy_low']:.1f}–{plan['healthy_high']:.1f} kg.")
    if plan["weeks"]:
        st.info(f"Ước tính cần khoảng {plan['weeks']:.0f} tuần để đạt cân nặng mục tiêu. ")
    for note in plan["notes"]:
        st.warning(note)
    st.caption("Công thức: BMR theo Mifflin-St Jeor, TDEE = BMR × hệ số vận động, "
               "thâm hụt ≈ 7.700 kcal cho mỗi kg. Ngưỡng tối thiểu 1.200 kcal (nữ) / 1.500 kcal (nam).")

target_kcal = plan["target"]

# ============================================================
# TAB 2: NHẬN DIỆN VÀ DANH SÁCH NGUYÊN LIỆU
# ============================================================
with tab_detect:
    st.subheader("Nhận diện rau củ bằng ảnh")
    mode = st.radio(
        "Chế độ nhận diện",
        ["Nhiều loại (YOLO)", "Một loại (CNN)"],
        index=0 if YOLO_FILE.is_file() else 1,
        horizontal=True,
    )
    use_yolo = mode == "Nhiều loại (YOLO)"
    if use_yolo and not YOLO_FILE.is_file():
        st.warning("Thiếu file outputs/vegetable_yolo.pt. Bạn có thể chọn nguyên liệu thủ công bên dưới.")
    elif not use_yolo and model is None:
        st.error("Chưa dùng được mô hình CNN. Bạn có thể chọn nguyên liệu thủ công bên dưới.")
    else:
        files = st.file_uploader("Chọn ảnh rau củ (có thể chọn nhiều ảnh)",
                                 type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        with st.expander("Hoặc chụp ảnh bằng camera"):
            shot = st.camera_input("Chụp ảnh rau củ", key="camera")
        sources = list(files or [])
        if shot is not None:
            sources.append(shot)

        fix_keys = []
        for source_index, source in enumerate(sources):
            raw = source.getvalue()
            image_key = f"{source_index}_{hashlib.md5(raw).hexdigest()[:10]}"
            if use_yolo:
                try:
                    annotated, detections = predict_objects(raw, YOLO_FILE.stat().st_mtime_ns)
                except ImportError:
                    st.error("Chưa cài ultralytics trong môi trường Python đang chạy ứng dụng.")
                    break
                except UnidentifiedImageError:
                    st.error(f"Không đọc được ảnh {source.name}. Hãy chọn ảnh khác.")
                    continue
                except Exception as error:
                    st.error(f"YOLO không xử lý được ảnh {source.name}: {error}")
                    continue

                st.image(annotated, caption=source.name, width=480)
                if not detections:
                    st.info("YOLO chưa phát hiện rau củ trong ảnh này.")
                    continue

                detected = {YOLO_LABELS.get(label, label): score for label, score in detections.items()}
                st.caption("Đã phát hiện " + " · ".join(
                    f"{vi(label)} ({score:.0%})" for label, score in detected.items()
                ))
                unsupported = [vi(label) for label in detected if label not in PANTRY_OPTIONS]
                if unsupported:
                    st.caption("Chưa có trong danh sách nguyên liệu của ứng dụng: " + ", ".join(unsupported))

                fix_key = f"fix_yolo_{image_key}"
                st.multiselect(
                    "Xác nhận hoặc sửa nguyên liệu",
                    options=PANTRY_OPTIONS,
                    default=[label for label in detected if label in PANTRY_OPTIONS],
                    format_func=vi,
                    key=fix_key,
                )
                fix_keys.append(fix_key)
                continue

            try:
                scores = np.asarray(predict_scores(raw))
            except (UnidentifiedImageError, OSError):
                st.error(f"Không đọc được ảnh {source.name}. Hãy chọn ảnh khác.")
                continue

            order = np.argsort(scores)[::-1][:3]
            best = int(order[0])
            left, right = st.columns([1, 3])
            left.image(raw, width=170)
            right.markdown(f"**Dự đoán: {vi(class_names[best])}** ({scores[best]:.0%})")
            right.caption("Top 3: " + " · ".join(
                f"{vi(class_names[i])} {scores[i]:.0%}" for i in order))
            if scores[best] < CONFIDENCE_OK:
                right.warning("Mô hình chưa chắc chắn. Có thể đây không phải loại rau củ mô hình đã học, "
                              "hãy kiểm tra và chọn lại bên dưới.")
            fix_key = f"fix_cnn_{image_key}"
            right.selectbox("Xác nhận hoặc sửa lại:", options=class_names, index=best,
                            format_func=vi, key=fix_key)
            fix_keys.append(fix_key)

        if fix_keys:
            st.button("➕ Thêm vào danh sách nguyên liệu", type="primary",
                      on_click=add_detected_cb, args=(fix_keys,),
                      disabled=not any(st.session_state.get(key) for key in fix_keys))
        if use_yolo:
            st.caption("YOLO nhận diện bông cải xanh, cà rốt, hành tây, ớt chuông và cà chua. "
                       "Kiểm tra danh sách trước khi thêm nguyên liệu.")
        else:
            st.caption("CNN chọn một trong 15 loại rau củ cho mỗi ảnh. "
                       "Chụp một loại rau củ, nền gọn, đủ sáng.")

    st.divider()
    st.subheader("Nguyên liệu bạn đang có")
    selected = st.multiselect(
        "Thêm trực tiếp hoặc bấm × để bỏ:", options=PANTRY_OPTIONS, format_func=vi, key="ingredients",
    )
    st.button("Xóa toàn bộ danh sách", on_click=clear_pantry_cb, disabled=not selected)
    if sorted(selected) != sorted(data["pantry"]):  # tự động lưu danh sách
        data["pantry"] = list(selected)
        save_data()
    available = set(selected)

# ============================================================
# TAB 3: MÓN ĂN
# ============================================================
with tab_recipes:
    st.subheader("Gợi ý món từ nguyên liệu bạn có")
    st.caption(f"Kiểm tra: {sorted(available)} | Tổng công thức: {len(RECIPES)}")
    ranked = []
    for recipe in RECIPES:
        needed, missing = recipe_status(recipe, available)
        if not available or set(needed) != available:
            continue

        _, missing = recipe_status(recipe, available)
        per = recipe_per_serving(recipe)
        ranked.append((len(missing), per["kcal"], recipe, missing, per))

    ranked.sort(key=lambda row: (row[0], row[1]))

    if not available:
        st.info("Hãy nhận diện hoặc chọn nguyên liệu để xem món gợi ý.")
    elif not ranked:
        st.info("Chưa có công thức dùng đúng các nguyên liệu đã chọn.")
    st.dataframe(
        [
            {
                "Món": recipe["name"],
                "Còn thiếu": ", ".join(vi(k) for k in missing) or "Đủ nguyên liệu",
                "Thời gian (phút)": recipe["time"],
                "kcal/khẩu phần": round(per["kcal"]),
                "Đạm (g)": round(per["protein"], 1),
            }
            for _, _, recipe, missing, per in ranked
        ],
        hide_index=True,
    )
    st.divider()
    st.subheader("Cách nấu món ăn")
    suggested_recipes = {row[2]["name"]: row[2] for row in ranked}

    if not suggested_recipes:
        st.info("Chưa có món phù hợp để hiển thị cách nấu.")
    else:
        recipe_names = list(suggested_recipes)
        if st.session_state.get("calc_recipe") not in suggested_recipes:
            st.session_state["calc_recipe"] = recipe_names[0]

        chosen_name = st.selectbox(
            "Chọn món:", options=recipe_names, key="calc_recipe"
        )
        chosen = suggested_recipes[chosen_name]
        _, missing_now = recipe_status(chosen, available)
        if missing_now:
            st.warning("Bạn cần chuẩn bị thêm: " + ", ".join(vi(k) for k in missing_now))
        with st.expander("📖 Nguyên liệu và cách làm chi tiết", expanded=True):
            base = recipe_per_serving(chosen)
            st.caption(f"⏱ {chosen['time']} phút · 👥 {chosen['servings']} khẩu phần · "
                    f"≈ {base['kcal']:.0f} kcal/khẩu phần (theo định lượng mẫu)")
            st.markdown("**Nguyên liệu**")
            st.markdown("\n".join(f"- {line}" for line in chosen["ingredients"]))
            st.markdown("**Cách làm**")
            for number, step in enumerate(chosen["steps"], start=1):
                st.markdown(f"{number}. {step}")
            if chosen["tips"]:
                st.markdown("**Lưu ý**")
                st.markdown("\n".join(f"- {tip}" for tip in chosen["tips"]))

        st.caption("Khối lượng cho CẢ MÓN: rau củ tính trước khi nấu, cơm và ức gà tính sau khi nấu. "
                "Hãy chỉnh theo lượng thực tế.")
        grams_by_item = {}
        for key, default in chosen["items"].items():
            grams_by_item[key] = st.number_input(
                f"{vi(key)} (g)", min_value=0, max_value=5000, value=int(default), step=5,
                key=f"g_{chosen_name}_{key}",
            )
        extra_kcal = st.number_input(
            "Năng lượng bổ sung khác như đường, nước sốt, nước chấm (kcal)",
            min_value=0.0, max_value=5000.0, value=0.0, step=5.0, key=f"extra_{chosen_name}",
        )
        servings = st.number_input("Chia món thành bao nhiêu khẩu phần bằng nhau?",
                                min_value=1, max_value=20, value=chosen["servings"], step=1,
                                key=f"servings_{chosen_name}")

        total, _ = nutrition(grams_by_item)
        total["kcal"] += extra_kcal
        per = per_serving(total, servings)

        st.dataframe([{
            "Nguyên liệu": vi(k), "Khối lượng (g)": g,
            "kcal": round(nutrition({k: g})[0]["kcal"], 1),
        } for k, g in grams_by_item.items()], hide_index=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("kcal / khẩu phần", f"{per['kcal']:.0f}")
        c2.metric("Đạm (g)", f"{per['protein']:.1f}")
        c3.metric("Tinh bột (g)", f"{per['carbs']:.1f}")
        c4.metric("Béo (g)", f"{per['fat']:.1f}")
        st.caption(f"Cả món: {total['kcal']:.0f} kcal.")

        a1, a2 = st.columns(2)
        default_meal = chosen["meals"][0]
        meal_pick = a1.selectbox("Thêm vào bữa nào?", MEALS, index=MEALS.index(default_meal),
                                key=f"meal_{chosen_name}")
        eaten = a2.number_input("Bạn ăn bao nhiêu khẩu phần?", min_value=0.25, max_value=10.0,
                                value=1.0, step=0.25, key=f"eaten_{chosen_name}")
        st.button("Thêm vào thực đơn", type="primary", disabled=total["kcal"] <= 0,
                on_click=add_entries_cb,
                args=(day_key, [make_entry(meal_pick, chosen_name, eaten, per)],
                        f"Đã thêm {chosen_name} vào {meal_pick.lower()}."))

# ============================================================
# TAB 4: THỰC ĐƠN
# ============================================================
with tab_menu:
    st.subheader(f"Thực đơn ngày {view_date.strftime('%d/%m/%Y')}")
    menu = data["menus"].get(day_key, [])
    eaten_kcal = sum(e["kcal"] for e in menu)

    st.progress(min(eaten_kcal / target_kcal, 1.0) if target_kcal else 0.0,
                text=f"Calo dự kiến {eaten_kcal:.0f} / {target_kcal:.0f} kcal")
    remaining = target_kcal - eaten_kcal
    if eaten_kcal > target_kcal * 1.05:
        st.warning(f"Đã vượt mục tiêu khoảng {-remaining:.0f} kcal.")
    elif menu:
        st.info(f"Còn khoảng {max(remaining, 0):.0f} kcal để đạt mục tiêu hôm nay.")

    if menu:
        ordered = sorted(menu, key=lambda e: MEALS.index(e["meal"]) if e["meal"] in MEALS else 99)
        meals_in_menu = {}
        for entry in ordered:
            meals_in_menu.setdefault(entry["meal"], []).append(entry)

        grid_rows = []
        for meal in ("Bữa sáng", "Bữa phụ", "Bữa trưa", "Bữa tối"):
            entries = meals_in_menu.get(meal, [])
            cell = {"text": " + ".join(f"{x['name']} ({x['portions']:g} phần)" for x in entries),
                    "kcal": sum(x["kcal"] for x in entries)} if entries else None
            grid_rows.append((meal, [cell]))
        st.markdown(
            render_menu_grid([], grid_rows, "Thực đơn hôm nay của bạn"),
            unsafe_allow_html=True,
        )

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Tổng kcal", f"{eaten_kcal:.0f}")
        t2.metric("Đạm (g)", f"{sum(e['protein'] for e in menu):.0f}")
        t3.metric("Tinh bột (g)", f"{sum(e['carbs'] for e in menu):.0f}")
        t4.metric("Béo (g)", f"{sum(e['fat'] for e in menu):.0f}")

        by_meal = {}
        for e in menu:
            by_meal[e["meal"]] = by_meal.get(e["meal"], 0) + e["kcal"]
        st.dataframe([{"Bữa": m, "kcal": round(by_meal[m], 1)} for m in MEALS if m in by_meal],
                     hide_index=True)

        remove_key = f"rm_{day_key}"
        labels = {e["id"]: f"{e['meal']} — {e['name']} ({e['kcal']:.0f} kcal)" for e in menu}
        st.multiselect("Chọn món muốn xóa:", options=list(labels), format_func=labels.get, key=remove_key)
        st.button("Xóa các món đã chọn", on_click=remove_entries_cb, args=(day_key, remove_key),
                  disabled=not st.session_state.get(remove_key))
        st.button("🗑️ Xóa tất cả thực đơn hôm nay", on_click=clear_menu_cb, args=(day_key, remove_key))

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["Ngày", "Bữa", "Món ăn", "Khẩu phần", "kcal", "Đạm (g)", "Tinh bột (g)", "Béo (g)"])
        for e in ordered:
            writer.writerow([day_key, e["meal"], e["name"], e["portions"], e["kcal"],
                             e["protein"], e["carbs"], e["fat"]])
        st.download_button("Tải thực đơn CSV", data=buffer.getvalue().encode("utf-8-sig"),
                           file_name=f"thuc_don_{day_key}.csv", mime="text/csv")
    else:
        st.info("Thực đơn ngày này chưa có món nào.")

    st.divider()
    st.subheader("Gợi ý 3 bữa chính theo mục tiêu calo")
    sg_key = f"{profile_name}|{day_key}"
    seen = st.session_state.setdefault("suggest_seen", {}).setdefault(sg_key, set())
    round_map = st.session_state.setdefault("suggest_round", {})
    round_no = round_map.get(sg_key, 0)
    day_plan = suggest_day(available, target_kcal, menu, seen, f"{sg_key}|{round_no}" if round_no else "")
    restarted = False
    if not day_plan and seen:  # đã gợi ý hết món phù hợp -> bắt đầu vòng mới
        seen.clear()
        round_map[sg_key] = round_no = 0
        day_plan = suggest_day(available, target_kcal, menu)
        restarted = True
    if not day_plan:
        st.info("Hãy thêm nguyên liệu ở tab Nhận diện để nhận gợi ý.")
    else:
        if restarted:
            st.info("Đã gợi ý hết các món phù hợp, ứng dụng bắt đầu lại từ đầu.")
        st.caption(f"Gợi ý lần {round_no + 1} (bấm Xáo trộn để xem lần tiếp theo, các lần không trùng món nhau)")
        st.dataframe([{
            "Bữa": meal, "Món gợi ý": name, "Khẩu phần": mult,
            "kcal": round(per["kcal"] * mult),
        } for meal, name, mult, per in day_plan], hide_index=True)
        planned = sum(per["kcal"] * mult for _, _, mult, per in day_plan)
        st.caption(f"Tổng gợi ý khoảng {planned:.0f} kcal so với mục tiêu {target_kcal:.0f} kcal. "
                   "Các món rau củ thường ít calo, phần còn thiếu nên bổ sung từ tinh bột (cơm, yến mạch) "
                   "và đạm (trứng, ức gà, đậu hũ) ở mục thêm thực phẩm bên dưới.")
        st.button("Thêm các món gợi ý vào thực đơn", on_click=add_entries_cb,
                  args=(day_key,
                        [make_entry(meal, name, mult, per) for meal, name, mult, per in day_plan],
                        "Đã thêm các món gợi ý.", True))
        st.button("🔀 Xáo trộn món gợi ý", on_click=shuffle_suggestion_cb,
                  args=(sg_key, [name for _, name, _, _ in day_plan]))

    with st.expander("➕ Thêm thực phẩm lẻ hoặc món tự nhập"):
        st.markdown("**Theo khối lượng**")
        g1, g2, g3 = st.columns(3)
        food_key = g1.selectbox("Thực phẩm", options=list(FOODS), format_func=vi, key="quick_food")
        food_grams = g2.number_input("Khối lượng (g)", min_value=1, max_value=3000, value=100, step=10,
                                     key="quick_grams")
        food_meal = g3.selectbox("Bữa", MEALS, key="quick_meal")
        food_per = per_serving(nutrition({food_key: food_grams})[0], 1)
        st.button(f"Thêm {food_grams} g {vi(food_key).lower()} (≈ {food_per['kcal']:.0f} kcal)",
                  on_click=add_entries_cb,
                  args=(day_key, [make_entry(food_meal, f"{vi(food_key)} {food_grams} g", 1, food_per)],
                        "Đã thêm thực phẩm."))

        st.markdown("**Món tự nhập**")
        m1c, m2c, m3c = st.columns(3)
        custom_name = m1c.text_input("Tên món", key="custom_name")
        custom_kcal = m2c.number_input("kcal", min_value=0.0, max_value=5000.0, value=0.0, step=10.0,
                                       key="custom_kcal")
        custom_meal = m3c.selectbox("Bữa", MEALS, key="custom_meal")
        st.button("Thêm món tự nhập", disabled=not custom_name.strip() or custom_kcal <= 0,
                  on_click=add_entries_cb,
                  args=(day_key,
                        [make_entry(custom_meal, custom_name.strip(), 1,
                                    {"kcal": custom_kcal, "protein": 0.0, "carbs": 0.0, "fat": 0.0})],
                        "Đã thêm món tự nhập."))
        st.caption("Món tự nhập không có số liệu đạm/tinh bột/béo.")

    st.caption("Dữ liệu được lưu tự động trong thư mục data/ theo từng hồ sơ và từng ngày. "
               "Sửa công thức ở tab Món ăn không làm thay đổi món đã thêm.")

# ============================================================
# TAB 5: CÂN NẶNG
# ============================================================
with tab_weight:
    st.subheader("Theo dõi cân nặng")
    st.number_input(f"Cân nặng ngày {view_date.strftime('%d/%m/%Y')} (kg)",
                    min_value=30.0, max_value=300.0, step=0.1, key="log_weight")
    w1, w2 = st.columns(2)
    w1.button("Ghi nhận cân nặng", type="primary", on_click=log_weight_cb, args=(day_key,))
    w2.button("Xóa bản ghi của ngày này", on_click=delete_weight_cb, args=(day_key,),
              disabled=not any(w["date"] == day_key for w in data["weights"]))

    weights = sorted(data["weights"], key=lambda w: w["date"])
    if not weights:
        st.info("Chưa có bản ghi cân nặng. Hãy cân vào buổi sáng, cùng điều kiện mỗi lần để so sánh chính xác.")
    else:
        frame = pd.DataFrame({"Ngày": [w["date"] for w in weights],
                              "Cân nặng (kg)": [w["kg"] for w in weights]})
        st.line_chart(frame.set_index("Ngày"))

        first, last = weights[0]["kg"], weights[-1]["kg"]
        k1, k2 = st.columns(2)
        k1.metric("Cân nặng gần nhất", f"{last:.1f} kg",
                  f"{last - first:+.1f} kg so với lần đầu", delta_color="inverse")
        if first > goal_weight:
            progress = (first - last) / (first - goal_weight)
            k2.metric("Tiến độ đến mục tiêu", f"{min(max(progress, 0), 1):.0%}")
            st.progress(min(max(progress, 0.0), 1.0))
        st.button("Cập nhật cân nặng trong hồ sơ theo lần ghi gần nhất", on_click=use_latest_weight_cb)
        st.caption("Cập nhật cân nặng hồ sơ định kỳ để mục tiêu calo luôn sát thực tế.")
