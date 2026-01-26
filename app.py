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
# Streamlit config (MUST be first st.* call)
# =========================
st.set_page_config(page_title="WearSnap", layout="wide")


# =========================
# rembg session (global)
# =========================
REMBG_SESSION = new_session("u2net")  # or "u2netp"


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
        try:
            os.remove(OUT_FINAL)
        except Exception:
            pass

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
    """PNG/JPG両対応：白文字+黒縁取りの透かし（明るい背景でも見える）"""
    if not os.path.exists(path):
        return

    img = Image.open(path).convert("RGBA")
    W, H = img.size

    font_size = max(26, int(min(W, H) * 0.10))

    font = None
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Streamlit Cloud
        "/System/Library/Fonts/SFNS.ttf",                   # macOS
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

    # ✅ positions を必ずここで定義（NameError回避）
    positions = [
        (margin, margin),                      # 左上
        (W - tw - margin, H - th - margin),    # 右下
    ]

    alpha = int(255 * max(0.0, min(1.0, opacity_pct)))
    fill = (255, 255, 255, alpha)                 # 白
    stroke = (0, 0, 0, int(alpha * 0.85))          # 黒縁
    stroke_width = max(2, int(font_size * 0.06))   # 2〜6px

    for (x, y) in positions:
        try:
            draw.text(
                (x, y),
                text,
                font=font,
                fill=fill,
                stroke_width=stroke_width,
                stroke_fill=stroke,
            )
        except TypeError:
            # 古いPillow対策
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
    """アップロード画像bytes → rembgで透過PNG(RGBA)にして保存"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    inp = Image.open(io.BytesIO(uploaded_bytes)).convert("RGBA")

    buf = io.BytesIO()
    inp.save(buf, format="PNG")
    out_bytes = remove(buf.getvalue(), session=REMBG_SESSION)

    out = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
    out.save(out_path)
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


def do_generate(
    out_path: str,
    label: str,
    person_path: str,
    top_path: str,
    person_rgba_path: str,
    cx_in: float,
    y_in: float,
    w_in: float,
    angle_in: float,
    alpha_in: float,
    is_free: bool,
):
    with st.spinner(f"{label}..."):
        rc, out, err, cmdline = run_tryon(
            person_path,
            top_path,
            cx_in, y_in, w_in,
            angle_in, alpha_in,
            out_path,
            person_rgba_path=person_rgba_path,
        )

        with st.expander("🛠 実行ログ（デバッグ）", expanded=False):
            st.code(cmdline)
            st.code(out if out else "(stdout empty)")
            st.code(err if err else "(stderr empty)")

        if rc != 0:
            st.error("生成に失敗しました。エラーを確認してください。")
            return rc

        if is_free:
            apply_watermark_any(out_path)
            st.sidebar.warning("無料プラン：透かしを適用しました ✅")
        else:
            st.sidebar.success("有料プラン：透かしなし ✅")

        st.success(f"Saved: {out_path}")
        return rc


# =========================
# UI
# =========================
st.title("👕 WearSnap")
st.caption("写真1枚で、服の試着イメージをすぐ確認（大人モード）")

st.markdown("## プラン")
plan = st.radio("無料 / 有料", ["無料（透かしあり）", "有料（透かしなし）"], index=0, horizontal=True)
is_free = plan.startswith("無料")

st.subheader("🧭 WearSnap：かんたん3ステップ")

# -------------------------
# Step 1) Person
# -------------------------
st.markdown("## 1) 人物写真")
person_upload = st.file_uploader(
    "人物写真をアップロード（jpg / png）",
    type=["jpg", "jpeg", "png"],
    key="person_upload_main",
)

person_path = None
person_rgba_path = None

if person_upload is not None:
    try:
        os.makedirs("assets", exist_ok=True)
        raw = person_upload.getvalue()

        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.save(PERSON_RGB, quality=95)
        person_path = PERSON_RGB

        # 人物マスクRGBA（首推定/下地化用）
        buf = io.BytesIO()
        img.convert("RGBA").save(buf, format="PNG")
        out_bytes = remove(buf.getvalue(), session=REMBG_SESSION)
        Image.open(io.BytesIO(out_bytes)).convert("RGBA").save(PERSON_RGBA)
        person_rgba_path = PERSON_RGBA

        st.success("人物写真を読み込みました ✅")

    except UnidentifiedImageError:
        st.error("人物写真が読み込めません（JPEG/PNGで再アップロード。HEIC不可）")
        person_path = None
        person_rgba_path = None

# -------------------------
# Step 2) Top
# -------------------------
st.markdown("## 2) 服画像（トップス）")
top_upload = st.file_uploader(
    "服画像をアップロード（jpg / png）",
    type=["jpg", "jpeg", "png"],
    key="top_upload_main",
)

if top_upload is not None:
    raw = top_upload.getvalue()
    sig = hashlib.md5(raw).hexdigest()

    if sig != st.session_state.top_sig:
        st.session_state.top_sig = sig
        st.session_state.top_path = auto_rgba_with_rembg(raw, AUTO_TOP_PATH)

        st.session_state.has_generated = False
        if os.path.exists(OUT_FINAL):
            try:
                os.remove(OUT_FINAL)
            except Exception:
                pass

        st.success("服の背景を自動で透過しました ✅")

top_path = st.session_state.top_path

# -------------------------
# Ready check
# -------------------------
ready_person = person_path is not None and os.path.exists(person_path)
ready_top = top_path is not None and os.path.exists(top_path)
ready_all = ready_person and ready_top

st.markdown("### ✅ 入力チェック")
c1, c2 = st.columns(2)
with c1:
    st.write("人物：", "OK ✅" if ready_person else "未アップロード ❌")
with c2:
    st.write("服：", "OK ✅" if ready_top else "未アップロード ❌")

# -------------------------
# Step 3) Settings + Run
# -------------------------
st.markdown("## 3) 設定して試着（大人）")

with st.expander("微調整（上級者向け）", expanded=False):
    cx = st.slider("cx（中心X）", 0.00, 1.00, 0.50, 0.01)

    # ✅ 大人はここが命：小さくすると上がる / 大きくすると下がる
    y = st.slider("y（首から下へ）", 0.00, 0.40, 0.08, 0.01)

    w = st.slider("w（幅）", 0.70, 1.25, 1.03, 0.01)
    angle = st.slider("angle（回転）", -10.0, 10.0, -1.5, 0.5)
    alpha = st.slider("alpha（透過）", 0.10, 1.00, 1.00, 0.01)

btn1, _ = st.columns(2)
with btn1:
    gen_btn = st.button("👕 試着する", disabled=(not ready_all), use_container_width=True)

# -------------------------
# Main layout (Preview / Result)
# -------------------------
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("入力プレビュー")

        st.markdown("**人物**")
        if ready_person:
            st.image(person_path, width=560)
        else:
            st.info("① 人物写真をアップロードしてください")

        st.markdown("---")

        st.markdown("**トップス**")
        if ready_top:
            st.image(top_path, width=720)
        else:
            st.info("② トップス画像をアップロードしてください")

with col2:
    with st.container(border=True):
        st.subheader("✨ 試着結果")

        if os.path.exists(OUT_FINAL):
            st.success("試着が完了しました")
            st.image(OUT_FINAL, width=900)

            with open(OUT_FINAL, "rb") as f:
                st.download_button(
                    label="📥 画像を保存する",
                    data=f.read(),
                    file_name="wearsnap_result.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )
        else:
            st.info("③ 「試着する」を押すと、ここに結果が表示されます")

# -------------------------
# Action
# -------------------------
if gen_btn:
    # 大人は余計な補正をしない（ブレの原因になる）
    cx_use, y_use, w_use = cx, y, w
    last_mode = "MANUAL"

    rc = do_generate(
        out_path=OUT_FINAL,
        label=f"生成中（{last_mode}）",
        person_path=person_path,
        top_path=top_path,
        person_rgba_path=person_rgba_path,
        cx_in=cx_use,
        y_in=y_use,
        w_in=w_use,
        angle_in=angle,
        alpha_in=alpha,
        is_free=is_free,
    )

    st.session_state.has_generated = (rc == 0)
    if rc == 0:
        st.rerun()
