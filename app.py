import io
import os
import subprocess
import sys
import time

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from rembg import remove

# ---- paths (必ず先に定義) ----
DEFAULT_PERSON = "assets/person2_fixed.jpg"
DEFAULT_TOP = "assets/tshirt_rgba.png"
OUT_TMP = "outputs/tryon_tmp.jpg"
OUT_FINAL = "outputs/tryon_top_final.jpg"

# ---- person cache ----
PERSON_RGB = "assets/uploaded_person.jpg"
PERSON_RGBA = "assets/uploaded_person_rgba.png"

# ===== セッション初期化 =====
if "boot_done" not in st.session_state:
    st.session_state.boot_done = True
    st.session_state.has_generated = False

    # 起動時は前回の生成結果を表示しない
    if os.path.exists(OUT_FINAL):
        os.remove(OUT_FINAL)

if "has_generated" not in st.session_state:
    st.session_state.has_generated = False

def apply_watermark_any(
    path: str,
    text: str = "TRY-ON MVP  FREE",
    opacity: int = 70,
    step: int = 220,
):
    """PNG/JPGどちらでも透かしを入れる（無料版用）"""
    img = Image.open(path).convert("RGBA")
    W, H = img.size

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

def apply_watermark_any(path: str, text: str = "TRY-ON MVP  FREE", opacity: int = 70, step: int = 220):
    """
    PNG / JPG 両対応の透かし処理
    """
    img = Image.open(path).convert("RGBA")
    W, H = img.size

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("Arial.ttf", size=max(28, int(min(W, H) * 0.05)))
    except Exception:
        font = ImageFont.load_default()

    for yy in range(-H, H * 2, step):
        for xx in range(-W, W * 2, step):
            draw.text((xx, yy), text, font=font, fill=(255, 255, 255, opacity))

    overlay = overlay.rotate(-22, expand=False)
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
    out_bytes = remove(buf.getvalue())
    out = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
    out.save(out_path)
    return out_path
from PIL import Image, ImageDraw, ImageFont

# ---- paths ----
DEFAULT_PERSON = "assets/person2_fixed.jpg"
DEFAULT_TOP = "assets/tshirt_rgba.png"
OUT_TMP = "outputs/tryon_tmp.jpg"
OUT_FINAL = "outputs/tryon_top_final.jpg"

# ---- person cache ----
PERSON_RGBA = None
PERSON_RGB = "assets/uploaded_person.jpg"
def apply_watermark_jpg(path: str, text: str = "TRY-ON MVP  FREE", opacity: int = 70, step: int = 220):
    """
    JPGに薄い透かしを斜めに入れる（無料版用）
    opacity: 0~255（小さいほど薄い）
    step: 文字の間隔（大きいほど疎）
    """
    if not os.path.exists(path):
        return

    base = Image.open(path).convert("RGBA")
    W, H = base.size

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay) 
    # フォント（環境依存なのでデフォルトで）
    font = ImageFont.load_default()
    
    # 斜めタイル状に文字を敷き詰める
    for y in range(-H, H * 2, step):
        for x in range(-W, W * 2, step):
            draw.text((x, y), text, font=font, fill=(255, 255, 255, opacity))

    overlay = overlay.rotate(-22, expand=False)
    out = Image.alpha_composite(base, overlay).convert("RGB")
    out.save(path, quality=95)

def run_tryon(person_path: str, top_path: str, cx: float, y: float, w: float, angle: float, alpha: float, out_path: str):
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
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr, " ".join(cmd)

def estimate_cx_y_w(person_rgb_path: str, person_rgba_path: str):
    import cv2
    import numpy as np

    img = cv2.imread(person_rgb_path)
    H, W = img.shape[:2]

    # y：顔検出（顔の下＝首元目安）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    y = 0.32  # fallback（今の当たり値）
    if len(faces) > 0:
        x, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        y = (fy + fh) / H + 0.03

    # cx / w：人物マスクの横幅から推定
    cx = 0.51
    w = 1.02
    if person_rgba_path and os.path.exists(person_rgba_path):
        rgba = cv2.imread(person_rgba_path, cv2.IMREAD_UNCHANGED)  # BGRA
        alpha = rgba[:, :, 3]
        ys, xs = np.where(alpha > 10)
        if len(xs) > 0:
            x0, x1 = xs.min(), xs.max()
            cx = ((x0 + x1) / 2) / W
            w = ((x1 - x0) / W) * 1.10  # ちょい大きめ

    cx = float(max(0.0, min(1.0, cx)))
    y = float(max(0.0, min(1.0, y)))
    w = float(max(0.3, min(1.3, w)))
    return cx, y, w

st.set_page_config(page_title="Virtual Try-on MVP (Top)", layout="wide")
st.title("👕 Virtual Try-on MVP（トップス）")
st.caption("※雰囲気確認用の試着です（サイズ厳密再現はしません）")

# ---- Sidebar ----
st.sidebar.header("入力")
st.sidebar.header("プラン")
plan = st.sidebar.radio("無料 / 有料", ["無料（透かしあり）", "有料（透かしなし）"], index=0)
is_free = plan.startswith("無料")
st.sidebar.subheader("人物（アップロード）")
person_upload = st.sidebar.file_uploader(
    "人物写真をアップロード（jpg/png）",
    type=["jpg", "jpeg", "png"]
)

from PIL import UnidentifiedImageError

# 保存先（固定）
PERSON_RGB = "assets/uploaded_person.jpg"
PERSON_RGBA = "assets/uploaded_person_rgba.png"

person_path = None
person_rgba_path = None

if person_upload is not None:
    try:
        os.makedirs("assets", exist_ok=True)
        raw = person_upload.getvalue()

        # 表示用（RGB）
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.save(PERSON_RGB, quality=95)
        person_path = PERSON_RGB

        # 自動位置合わせ用（人物マスクRGBA）
        buf = io.BytesIO()
        img.convert("RGBA").save(buf, format="PNG")
        out_bytes = remove(buf.getvalue())
        Image.open(io.BytesIO(out_bytes)).convert("RGBA").save(PERSON_RGBA)
        person_rgba_path = PERSON_RGBA

    except UnidentifiedImageError:
        st.sidebar.error("人物写真が読み込めません（JPEG/PNGで再アップロード。HEIC不可）")
        person_path = None
        person_rgba_path = None

st.sidebar.subheader("トップス（アップロード）")
import hashlib

import hashlib

top_upload = st.sidebar.file_uploader(
    "服画像をアップロード（jpg/png）",
    type=["jpg", "jpeg", "png"]
)

AUTO_TOP_PATH = "assets/uploaded_top_rgba.png"

if "top_sig" not in st.session_state:
    st.session_state.top_sig = None
if "top_path" not in st.session_state:
    st.session_state.top_path = None

if top_upload is not None:
    raw = top_upload.getvalue()
    sig = hashlib.md5(raw).hexdigest()

    if sig != st.session_state.top_sig:
        st.session_state.top_sig = sig
        st.session_state.top_path = auto_rgba_with_rembg(raw, AUTO_TOP_PATH)
        st.sidebar.success("背景を自動で透過しました ✅")

        st.session_state.has_generated = False

        if os.path.exists(OUT_FINAL):
            os.remove(OUT_FINAL)

top_path = st.session_state.top_path
st.sidebar.header("パラメータ（スライダー）")
# あなたの当たり値をデフォルトにしておく
cx = st.sidebar.slider("cx（中心X）", 0.00, 1.00, 0.51, 0.01)
y = st.sidebar.slider("y（上端Y）", 0.00, 1.00, 0.32, 0.01)
w = st.sidebar.slider("w（幅）", 0.30, 1.30, 1.02, 0.01)
angle = st.sidebar.slider("angle（回転）", -10.0, 10.0, -1.5, 0.5)
alpha = st.sidebar.slider("alpha（透過）", 0.10, 1.00, 1.00, 0.01)

st.sidebar.markdown("---")
gen_btn = st.sidebar.button("試着する", disabled=(top_path is None), width="stretch")
save_btn = st.sidebar.button("✅ これで確定保存（final）", use_container_width=True)

st.sidebar.markdown("---")
# --- last run info (debug) ---
if "last_mode" not in st.session_state:
    st.session_state.last_mode = "-"
if "last_used" not in st.session_state:
    st.session_state.last_used = "-"

st.sidebar.caption(f"MODE: {st.session_state.last_mode}")
st.sidebar.caption(st.session_state.last_used)

st.sidebar.subheader("実行コマンド")
st.sidebar.caption("下に表示されるコマンドはコピペ可能です（デバッグ用）")

# ---- Main layout ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("入力プレビュー")

    # --- 人物 ---
    st.caption("人物")
    if person_path is None:
        st.empty()
    elif os.path.exists(person_path):
        st.image(person_path, width="stretch")
    else:
        st.warning("人物写真が見つかりません。")

    # --- トップス ---
    st.caption("トップス（透過PNG）")
    if top_path is None:
        st.empty()
    elif os.path.exists(top_path):
        st.image(top_path, width="stretch")
    else:
        st.warning("トップス画像が見つかりません。")

# ---- Actions ----
# ---- Actions ----
# ---- Actions ----
with col2:
    st.subheader("結果")
    st.caption("final（確定）")

    if not os.path.exists(OUT_FINAL):
        st.info("左の「試着する」を押してください。")
    else:
        with open(OUT_FINAL, "rb") as f:
            img_bytes = f.read()
        st.image(img_bytes, width="stretch")

def do_generate(
    out_path: str,
    label: str,
    top_path_in: str,
    cx_in: float,
    y_in: float,
    w_in: float,
    angle_in: float,
    alpha_in: float,
):
    with st.spinner(f"{label}..."):
        st.sidebar.info(
            f"USED: cx={cx_in:.2f} y={y_in:.2f} w={w_in:.2f} "
            f"ang={angle_in:.1f} a={alpha_in:.2f}"
        )

        rc, out, err, cmdline = run_tryon(
            person_path,
            top_path_in,
            cx_in, y_in, w_in,
            angle_in, alpha_in,
            out_path
        )

        st.sidebar.markdown("### 実行ログ（デバッグ）")
        st.sidebar.code(cmdline)
        st.sidebar.code(out if out else "(stdout empty)")
        st.sidebar.code(err if err else "(stderr empty)")

        if rc != 0:
            st.error("生成に失敗しました。エラーを確認してください。")
        else:
            if is_free:
                apply_watermark_any(out_path)
            st.success(f"Saved: {out_path}")

        return rc
# ---- Actions ----
auto_fit = st.sidebar.checkbox("自動位置合わせ（おすすめ）", value=True)

if gen_btn:
    if (
        auto_fit
        and person_upload is not None
        and person_rgba_path
        and os.path.exists(person_rgba_path)
    ):
        st.session_state.last_mode = "AUTO"

        cx2, y2, w2 = estimate_cx_y_w(person_path, person_rgba_path)
        w2 = min(1.30, w2 + 0.15)  # 子供対策

        rc = do_generate(
            OUT_FINAL,
            "生成中（final）",
            top_path,
            cx2, y2, w2,
            angle,
            alpha
        )
    else:
        st.session_state.last_mode = "MANUAL"

        st.sidebar.caption(
            f"AUTO_CHECK: upload={person_upload is not None}, "
            f"rgba={person_rgba_path}, "
            f"exists={os.path.exists(person_rgba_path) if person_rgba_path else False}"
        )

        rc = do_generate(
            OUT_FINAL,
            "生成中（final）",
            top_path,
            cx, y, w,
            angle,
            alpha
        )

    st.sidebar.caption(
        f"MODE={st.session_state.last_mode} / rc={rc} / final exists={os.path.exists(OUT_FINAL)}"
    )

    st.session_state.has_generated = (rc == 0)
    if rc == 0:
        st.rerun()
