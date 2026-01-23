import io
import os
import sys
import hashlib
import subprocess

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from rembg import remove
from rembg.session_factory import new_session


# =========================
# Streamlit config（最初に必須）
# =========================
st.set_page_config(page_title="WearSnap", layout="wide")


# =========================
# rembg session
# =========================
REMBG_SESSION = new_session("u2net")


# =========================
# Paths
# =========================
OUT_FINAL = "outputs/tryon_top_final.jpg"

PERSON_RGB = "assets/uploaded_person.jpg"
PERSON_RGBA = "assets/uploaded_person_rgba.png"
AUTO_TOP_PATH = "assets/uploaded_top_rgba.png"


# =========================
# Session init
# =========================
if "boot_done" not in st.session_state:
    st.session_state.boot_done = True
    st.session_state.has_generated = False
    if os.path.exists(OUT_FINAL):
        os.remove(OUT_FINAL)

if "top_sig" not in st.session_state:
    st.session_state.top_sig = None
if "top_path" not in st.session_state:
    st.session_state.top_path = None


# =========================
# Utils
# =========================
def apply_watermark_any(
    path: str,
    text: str = "WearSnap",
    opacity_pct: float = 0.22,
    angle: float = 18.0,
):
    """PNG/JPG両対応：白文字+黒縁取りの透かし（明るい背景でも確実に見える）"""
    if not os.path.exists(path):
        return

    img = Image.open(path).convert("RGBA")
    W, H = img.size

    font_size = max(26, int(min(W, H) * 0.10))

    font = None
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "Arial.ttf",
    ]
    for fp in candidates:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)

    margin = int(min(W, H) * 0.04)

    positions = [
        (margin, margin),
        (W - tw - margin, H - th - margin),
    ]

    alpha = int(255 * max(0.0, min(1.0, opacity_pct)))
    fill = (255, 255, 255, alpha)
    stroke = (0, 0, 0, int(alpha * 0.85))
    stroke_width = max(2, int(font_size * 0.06))

    for (x, y) in positions:
        try:
            draw.text(
                (x, y), text,
                font=font,
                fill=fill,
                stroke_width=stroke_width,
                stroke_fill=stroke
            )
        except TypeError:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    draw.text((x + dx, y + dy), text, font=font, fill=stroke)
            draw.text((x, y), text, font=font, fill=fill)

    overlay = overlay.rotate(angle, resample=Image.BICUBIC, expand=False)
    out = Image.alpha_composite(img, overlay)

    if path.lower().endswith(".png"):
        out.save(path, format="PNG")
    else:
        out.convert("RGB").save(path, format="JPEG", quality=95)


def auto_rgba_with_rembg(uploaded_bytes: bytes, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    inp = Image.open(io.BytesIO(uploaded_bytes)).convert("RGBA")
    buf = io.BytesIO()
    inp.save(buf, format="PNG")
    out_bytes = remove(buf.getvalue(), session=REMBG_SESSION)
    Image.open(io.BytesIO(out_bytes)).convert("RGBA").save(out_path)
    return out_path


def run_tryon(
    person_path: str,
    top_path: str,
    cx: float,
    y: float,
    w: float,
    angle: float,
    alpha: float,
    out_path: str,
    person_rgba_path: str = None,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cmd = [
        sys.executable, "step_top_overlay.py",
        "--person", person_path,
        "--top", top_path,
        "--cx", f"{cx:.4f}",
        "--y", f"{y:.4f}",
        "--w", f"{w:.4f}",
        "--angle", f"{angle:.4f}",
        "--alpha", f"{alpha:.4f}",
        "--out", out_path,
    ]

    if person_rgba_path and os.path.exists(person_rgba_path):
        cmd += ["--person_rgba", person_rgba_path]

    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr, " ".join(cmd)


def estimate_cx_w_from_mask(person_rgba_path: str):
    import cv2
    import numpy as np

    cx, w = 0.50, 0.90

    if not person_rgba_path or not os.path.exists(person_rgba_path):
        return cx, w

    rgba = cv2.imread(person_rgba_path, cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.shape[2] != 4:
        return cx, w

    alpha = rgba[:, :, 3]
    mask = (alpha > 10).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return cx, w

    x0, x1 = xs.min(), xs.max()
    cx = ((x0 + x1) / 2) / mask.shape[1]
    w = ((x1 - x0 + 1) / mask.shape[1]) * 1.10

    return float(cx), float(max(0.7, min(1.25, w)))


# =========================
# UI
# =========================
st.title("👕 WearSnap")
st.caption("写真1枚で、服の試着イメージをすぐ確認")

plan = st.radio("無料 / 有料", ["無料（透かしあり）", "有料（透かしなし）"], horizontal=True)
is_free = plan.startswith("無料")


# --- Step 1: Person ---
st.markdown("## 1) 人物写真")
person_upload = st.file_uploader("人物写真", type=["jpg", "jpeg", "png"])
person_path = person_rgba_path = None

if person_upload:
    raw = person_upload.getvalue()
    Image.open(io.BytesIO(raw)).convert("RGB").save(PERSON_RGB, quality=95)
    person_path = PERSON_RGB

    buf = io.BytesIO()
    Image.open(io.BytesIO(raw)).convert("RGBA").save(buf, format="PNG")
    Image.open(io.BytesIO(remove(buf.getvalue()))).save(PERSON_RGBA)
    person_rgba_path = PERSON_RGBA
    st.success("人物写真を読み込みました")


# --- Step 2: Top ---
st.markdown("## 2) トップス画像")
top_upload = st.file_uploader("服画像", type=["jpg", "jpeg", "png"])

if top_upload:
    raw = top_upload.getvalue()
    sig = hashlib.md5(raw).hexdigest()
    if sig != st.session_state.top_sig:
        st.session_state.top_sig = sig
        st.session_state.top_path = auto_rgba_with_rembg(raw, AUTO_TOP_PATH)
        if os.path.exists(OUT_FINAL):
            os.remove(OUT_FINAL)
        st.success("服の背景を透過しました")

top_path = st.session_state.top_path


# --- Step 3 ---
st.markdown("## 3) 試着設定")
mode = st.radio("体型モード", ["大人", "子供"])
is_child = mode.startswith("子供")

auto_fit = st.checkbox("自動位置合わせ", value=True)

cx = st.slider("cx", 0.0, 1.0, 0.5)
y = st.slider("y", 0.0, 0.4, 0.1)
w = st.slider("w", 0.5, 1.25, 0.9)
angle = st.slider("angle", -10.0, 10.0, -1.5)
alpha = st.slider("alpha", 0.1, 1.0, 1.0)

if st.button("👕 試着する"):
    if auto_fit and person_rgba_path:
        cx_use, w_use = estimate_cx_w_from_mask(person_rgba_path)
        y_use = y
        w_use = min(max(w_use, 1.00 if not is_child else 0.98), 1.06 if not is_child else 1.02)
        last_mode = "AUTO"
    else:
        cx_use, y_use, w_use = cx, y, w
        last_mode = "MANUAL"

    rc, *_ = run_tryon(
        person_path, top_path,
        cx_use, y_use, w_use,
        angle, alpha,
        OUT_FINAL,
        person_rgba_path
    )

    if rc == 0:
        if is_free:
            apply_watermark_any(OUT_FINAL)
            st.sidebar.warning("無料プラン：透かしを適用しました ✅")
        st.image(OUT_FINAL, width=900)
