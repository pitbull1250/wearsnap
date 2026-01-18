import io
import os
import subprocess
import sys
import time
import hashlib

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
    text: str = "WearSnap",
    opacity_pct: float = 0.16,
    angle: float = 18.0,
):
    """
    PNG/JPG両対応：オシャレ系の「1〜2箇所だけ」透かし
    - 文字は大きめ（画像幅に比例）
    - 薄め（opacity_pct）
    - 斜め（angle）
    """
    if not os.path.exists(path):
        return

    img = Image.open(path).convert("RGBA")
    W, H = img.size

    font_size = max(28, int(min(W, H) * 0.10))

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

    alpha = int(255 * max(0.0, min(1.0, opacity_pct)))
    fill = (0, 0, 0, alpha)  # 黒（白服でも見える）

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

    for (x, y) in positions:
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

    # fallback（今の当たり値）
    cx = 0.51
    y = 0.32
    w = 1.02

    # まずは人物マスク（RGBAのalpha）から “胴体っぽいbbox” を推定
    if person_rgba_path and os.path.exists(person_rgba_path):
        rgba = cv2.imread(person_rgba_path, cv2.IMREAD_UNCHANGED)  # BGRA
        if rgba is not None and rgba.shape[2] == 4:
            alpha = rgba[:, :, 3]
            mask = (alpha > 10).astype(np.uint8) * 255

            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                H, W = mask.shape[:2]
                y0, y1 = ys.min(), ys.max()
                x0, x1 = xs.min(), xs.max()
                bbox_h = y1 - y0 + 1

                # 胴体っぽい範囲（頭と足を切る）
                t0 = int(y0 + bbox_h * 0.25)
                t1 = int(y0 + bbox_h * 0.80)

                rows = np.arange(H)[:, None]
                torso = (mask > 0) & (rows >= t0) & (rows <= t1)
                ys2, xs2 = np.where(torso)

                # 胴体が取れたらそれを優先
                if len(xs2) > 0:
                    x0, x1 = xs2.min(), xs2.max()
                    y0, y1 = ys2.min(), ys2.max()

                # ここから推定
                cx = ((x0 + x1) / 2) / W
                y = (y0 / H) + 0.02         # 上端オフセット（少し下げる）
                w = ((x1 - x0 + 1) / W) * 1.15

    # 値の安全クリップ
    cx = float(max(0.0, min(1.0, cx)))
    y = float(max(0.0, min(1.0, y)))
    w = float(max(0.3, min(1.3, w)))

    return cx, y, w

st.set_page_config(page_title="WearSnap", layout="wide")

st.title("👕 WearSnap")
st.caption("写真1枚で、服の試着イメージをすぐ確認")

st.markdown("## プラン")
plan = st.radio("無料 / 有料", ["無料（透かしあり）", "有料（透かしなし）"], index=0, horizontal=True)
is_free = plan.startswith("無料")

# =========================
# WearSnap Wizard (Main UI)
# =========================

st.subheader("🧭 WearSnap：かんたん3ステップ")

# Step 1) 人物
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

        st.success("人物写真を読み込みました ✅")

    except UnidentifiedImageError:
        st.error("人物写真が読み込めません（JPEG/PNGで再アップロード。HEIC不可）")
        person_path = None
        person_rgba_path = None

# Step 2) 服
st.markdown("## 2) 服画像（トップス）")
top_upload = st.file_uploader(
    "服画像をアップロード（jpg / png）",
    type=["jpg", "jpeg", "png"],
    key="top_upload_main",
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
        st.session_state.has_generated = False
        if os.path.exists(OUT_FINAL):
            os.remove(OUT_FINAL)
        st.success("服の背景を自動で透過しました ✅")

top_path = st.session_state.top_path

# 入力が揃ったか
ready_person = person_path is not None and os.path.exists(person_path)
ready_top = top_path is not None and os.path.exists(top_path)
ready_all = ready_person and ready_top

st.markdown("### ✅ 入力チェック")
c1, c2 = st.columns(2)
with c1:
    st.write("人物：", "OK ✅" if ready_person else "未アップロード ❌")
with c2:
    st.write("服：", "OK ✅" if ready_top else "未アップロード ❌")

# Step 3) 設定 + 実行
st.markdown("## 3) 設定して試着")

mode = st.radio("体型モード", ["大人", "子供（小学生以下）"], index=0, horizontal=True)
is_child = mode.startswith("子供")

auto_fit = st.checkbox("自動位置合わせ（おすすめ）", value=True)

with st.expander("微調整（上級者向け）", expanded=False):
    cx = st.slider("cx（中心X）", 0.00, 1.00, 0.51, 0.01)
    y = st.slider("y（上端Y）", 0.00, 1.00, 0.32, 0.01)
    w = st.slider("w（幅）", 0.30, 1.30, 1.02, 0.01)
    angle = st.slider("angle（回転）", -10.0, 10.0, -1.5, 0.5)
    alpha = st.slider("alpha（透過）", 0.10, 1.00, 1.00, 0.01)

# 微調整を開いてない人向けのデフォルト値（expander内変数が未定義になるのを防ぐ）
if "cx" not in locals():
    cx, y, w, angle, alpha = 0.51, 0.32, 1.02, -1.5, 1.00

btn1, btn2 = st.columns(2)
with btn1:
    gen_btn = st.button("👕 試着する", disabled=(not ready_all), use_container_width=True)

# last info (debug値)
if "last_mode" not in st.session_state:
    st.session_state.last_mode = "-"
if "last_used" not in st.session_state:
    st.session_state.last_used = "-"

# ---- Main layout ----
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("入力プレビュー")

        st.markdown("**人物**")
        if person_path and os.path.exists(person_path):
            st.image(person_path, width=560)
        else:
            st.info("① 人物写真をアップロードしてください")

        st.markdown("---")

        st.markdown("**トップス**")
        if top_path and os.path.exists(top_path):
            st.image(top_path, width=720)  # ← あえて固定
        else:
            st.info("② トップス画像をアップロードしてください")

with col2:
    with st.container(border=True):
        st.subheader("✨ 試着結果")

        if os.path.exists(OUT_FINAL):
            st.success("試着が完了しました")

            # 結果画像表示
            st.image(OUT_FINAL, width=900)

            # 📥 ダウンロードボタン
            with open(OUT_FINAL, "rb") as f:
                st.download_button(
                    label="📥 画像を保存する",
                    data=f.read(),              # ←安定のため read() 推奨
                    file_name="wearsnap_result.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )

        else:
            st.info("③ 「試着する」を押すと、ここに結果が表示されます")

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
                st.sidebar.warning("WATERMARK APPLIED ✅")
            else:
                st.sidebar.success("NO WATERMARK (PAID)")

            st.success(f"Saved: {out_path}")

    return rc
# ---- Actions ----
mode = st.sidebar.radio(
    "体型モード",
    ["大人", "子供（小学生以下）"],
    index=0
)
is_child = mode.startswith("子供")
auto_fit = st.sidebar.checkbox("自動位置合わせ（おすすめ）", value=True)

if gen_btn:
    # ① AUTO / MANUAL で基準値を決める
    if (
        auto_fit
        and person_upload is not None
        and person_rgba_path
        and os.path.exists(person_rgba_path)
    ):
        st.session_state.last_mode = "AUTO"
        cx_use, y_use, w_use = estimate_cx_y_w(person_path, person_rgba_path)
    else:
        st.session_state.last_mode = "MANUAL"
        cx_use, y_use, w_use = cx, y, w

    # ② 共通の安全補正（弱め）
    w_use = min(1.25, w_use + 0.05)

    # ③ モード補正（※ここ重要）
    if is_child:
        # 子供：上げすぎると顎下に来るので弱め
        y_use = max(0.10, y_use - 0.02)
        w_use = min(1.25, w_use + 0.03)
    else:
        # 大人：もう少し上げたい
        y_use = max(0.10, y_use - 0.05)
        w_use = min(1.25, w_use + 0.05)

    # ④ 生成（★必ず if gen_btn の中）
    rc = do_generate(
        OUT_FINAL,
        "生成中（final）",
        top_path,
        cx_use,
        y_use,
        w_use,
        angle,
        alpha,
    )

    st.session_state.has_generated = (rc == 0)
    if rc == 0:
        st.rerun()
