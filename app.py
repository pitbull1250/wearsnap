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
# rembg session (stability + faster on Cloud)
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
    opacity_pct: float = 0.22,   # 0.16だと薄いことがあるので少し濃く
    angle: float = 18.0,
):
    """PNG/JPG両対応：白文字+黒縁取りの透かし（明るい背景でも見える）"""
    if not os.path.exists(path):
        return

    img = Image.open(path).convert("RGBA")
    W, H = img.size

    # 画像サイズに応じてフォントを決める
    font_size = max(26, int(min(W, H) * 0.10))

    # フォント選択
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

    # 文字サイズ計測
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

    # 白文字 + 黒縁取り（明るい背景でも確実に見える）
    alpha = int(255 * max(0.0, min(1.0, opacity_pct)))
    fill = (255, 255, 255, alpha)                 # 白
    stroke = (0, 0, 0, int(alpha * 0.85))          # 黒縁
    stroke_width = max(2, int(font_size * 0.06))   # だいたい2〜6pxくらい

    for (x, y) in positions:
        # 縁取り
        try:
            draw.text((x, y), text, font=font, fill=fill,
                      stroke_width=stroke_width, stroke_fill=stroke)
        except TypeError:
            # 古いPillow対策（strokeが使えない場合）
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    draw.text((x + dx, y + dy), text, font=font, fill=stroke)
            draw.text((x, y), text, font=font, fill=fill)

    # 回転して合成
    overlay = overlay.rotate(angle, resample=Image.BICUBIC, expand=False)
    out = Image.alpha_composite(img, overlay)

    # 保存（PNG/JPG両対応）
    if path.lower().endswith(".png"):
        out.save(path, format="PNG")
    else:
        out.convert("RGB").save(path, format="JPEG", quality=95)


def auto_rgba_with_rembg(uploaded_bytes: bytes, out_path: str):
    """アップロード画像bytes → rembgで透過PNG(RGBA)にして保存"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # PILで開けるか確認
    inp = Image.open(io.BytesIO(uploaded_bytes)).convert("RGBA")

    # PNG bytesにしてから remove() に渡す（安定）
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
        "--y", f"{y:.4f}",   # 首から下へ（H比）
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
    """
    マスクから「胴体っぽい範囲」を取り、cxとwだけ推定
    yは首基準なのでスライダー/固定値が安定
    """
    import cv2
    import numpy as np

    cx = 0.50
    w = 0.90

    if not person_rgba_path or not os.path.exists(person_rgba_path):
        return cx, w

    rgba = cv2.imread(person_rgba_path, cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
        return cx, w

    alpha = rgba[:, :, 3]
    mask = (alpha > 10).astype(np.uint8) * 255
    H, W = mask.shape[:2]

    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return cx, w

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bbox_h = max(1, (y1 - y0 + 1))

    # 胴体帯（頭と脚を避ける）
    t0 = int(y0 + bbox_h * 0.25)
    t1 = int(y0 + bbox_h * 0.75)

    rows = np.arange(H)[:, None]
    torso = (mask > 0) & (rows >= t0) & (rows <= t1)
    ys2, xs2 = np.where(torso)

    if xs2.size > 0:
        x0, x1 = int(xs2.min()), int(xs2.max())

    cx = ((x0 + x1) / 2) / float(W)

    # ここが “幅の係数”
    w = ((x1 - x0 + 1) / float(W)) * 1.10

    cx = float(max(0.0, min(1.0, cx)))
    w = float(max(0.70, min(1.25, w)))
    return cx, w


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
st.caption("写真1枚で、服の試着イメージをすぐ確認")

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
    except Exception as e:
        st.error(f"人物処理でエラー: {e}")
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
        try:
            st.session_state.top_path = auto_rgba_with_rembg(raw, AUTO_TOP_PATH)

            st.session_state.has_generated = False
            if os.path.exists(OUT_FINAL):
                try:
                    os.remove(OUT_FINAL)
                except Exception:
                    pass

            st.success("服の背景を自動で透過しました ✅")
        except Exception as e:
            st.error(f"服の透過でエラー: {e}")
            st.session_state.top_path = None

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
st.markdown("## 3) 設定して試着")

mode = st.radio("体型モード", ["大人", "子供（小学生以下）"], index=0, horizontal=True)
is_child = mode.startswith("子供")

auto_fit = st.checkbox("自動位置合わせ（おすすめ）", value=True)

with st.expander("微調整（上級者向け）", expanded=False):
    cx = st.slider("cx（中心X）", 0.00, 1.00, 0.50, 0.01)

    # ★ 首基準：首から下へ（H比）
    y = st.slider("y（首から下へ）", 0.00, 0.40, 0.10, 0.01)

    w = st.slider("w（幅）", 0.50, 1.25, 0.90, 0.01)
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
    if auto_fit and person_rgba_path and os.path.exists(person_rgba_path):
        cx_use, w_use = estimate_cx_w_from_mask(person_rgba_path)
        y_use = y
        last_mode = "AUTO"

        # ★AUTOで小さくなりすぎるのを防ぐ（下限）
        if not is_child:
            w_use = max(w_use, 1.00)   # 大人
        else:
            w_use = max(w_use, 0.98)   # 子供

        # ★AUTOでデカくなりすぎるのも防ぐ（上限）
        # ※ 子供が大きすぎ問題は「この上限」が効く
        if not is_child:
            w_use = min(w_use, 1.06)   # 大人
        else:
            w_use = min(w_use, 1.02)   # 子供

    else:
        cx_use, y_use, w_use = cx, y, w
        last_mode = "MANUAL"

    # 体型モード補正（軽め）
    if is_child:
        y_use = min(0.40, max(0.06, y_use + 0.02))
        w_use = min(1.25, w_use + 0.02)
    else:
        y_use = min(0.40, max(0.04, y_use - 0.02))
        w_use = min(1.25, w_use + 0.03)

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
