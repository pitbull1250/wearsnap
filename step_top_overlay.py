import argparse
import os
import cv2
import numpy as np


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def alpha_mask_from_rgba(rgba, thr=10):
    """RGBAのalphaから人物マスク(0/255)を作る"""
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
        return None
    a = rgba[:, :, 3]
    return (a > thr).astype(np.uint8) * 255


def estimate_neck_y_from_mask(mask_u8):
    """
    人物マスク(0/255)から首ラインYを推定
    - 上から下に走査して、マスク幅が急に増える位置＝首→肩の境界を探す
    """
    H, W = mask_u8.shape[:2]
    rows = (mask_u8 > 0).sum(axis=1).astype(np.float32)

    # 平滑化（縦方向）
    rows_s = cv2.GaussianBlur(rows.reshape(-1, 1), (1, 31), 0).reshape(-1)

    ys = np.where(rows_s > (W * 0.01))[0]
    if ys.size == 0:
        return int(H * 0.20)
    y_top = int(ys[0])

    y_end = int(H * 0.55)  # 首は上半分にある想定
    diff = np.diff(rows_s)
    diff[:y_top] = 0
    diff[y_end:] = 0

    neck_y = int(np.argmax(diff))
    neck_y = max(y_top + 5, min(neck_y, y_end))
    return neck_y


def alpha_bbox(rgba, thr=10):
    """rgba(H,W,4) の alpha>thr の領域bboxを返す (left, top, right, bottom)"""
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
        return None
    a = rgba[:, :, 3]
    ys, xs = np.where(a > thr)
    if xs.size == 0 or ys.size == 0:
        return None
    left = int(xs.min())
    right = int(xs.max())
    top = int(ys.min())
    bottom = int(ys.max())
    return left, top, right, bottom


def rotate_rgba_premult(top_rgba_premult_f32, angle_deg):
    """
    premultiply済みRGBA(float32 0..255)を回転して返す（premultiply前提）
    - RGBはpremultiply済みなので黒フチが出にくい
    """
    if abs(angle_deg) < 1e-6:
        return top_rgba_premult_f32

    h, w = top_rgba_premult_f32.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rgb_p = top_rgba_premult_f32[:, :, :3]
    a_u8 = np.clip(top_rgba_premult_f32[:, :, 3], 0, 255).astype(np.uint8)

    rgb_rot = cv2.warpAffine(
        rgb_p, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    a_rot = cv2.warpAffine(
        a_u8, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)

    out = np.dstack([rgb_rot, a_rot]).astype(np.float32)
    return out


def overlay_rgba(base_bgr, top_rgba_u8, x, y, alpha_scale=1.0):
    """base_bgr(H,W,3) に top_rgba(h,w,4) を (x,y) に合成"""
    H, W = base_bgr.shape[:2]
    h, w = top_rgba_u8.shape[:2]

    # クリップ
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, W)
    y2 = min(y + h, H)
    if x1 >= x2 or y1 >= y2:
        return base_bgr

    top_x1 = x1 - x
    top_y1 = y1 - y
    top_x2 = top_x1 + (x2 - x1)
    top_y2 = top_y1 + (y2 - y1)

    roi = base_bgr[y1:y2, x1:x2].astype(np.float32)
    top_crop = top_rgba_u8[top_y1:top_y2, top_x1:top_x2].astype(np.float32)

    # --- 黒フチ除去（エッジRGBをinpaint + alpha収縮 + フェザー） ---
    top_u8 = np.clip(top_crop, 0, 255).astype(np.uint8)
    a = top_u8[:, :, 3]

    rim = ((a > 0) & (a < 250)).astype(np.uint8) * 255
    if rim.any():
        rgb = top_u8[:, :, :3]
        rgb_fixed = cv2.inpaint(rgb, rim, 11, cv2.INPAINT_TELEA)
        top_u8[:, :, :3] = rgb_fixed

    top_crop = top_u8.astype(np.float32)

    alpha = (top_crop[:, :, 3] / 255.0) * float(alpha_scale)

    k = np.ones((3, 3), np.uint8)
    a8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    a8 = cv2.erode(a8, k, iterations=4)
    a8 = cv2.GaussianBlur(a8, (0, 0), sigmaX=1.5)

    alpha = a8.astype(np.float32) / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha3 = np.dstack([alpha, alpha, alpha])

    top_rgb = top_crop[:, :, :3]
    out = roi * (1.0 - alpha3) + top_rgb * alpha3
    base_bgr[y1:y2, x1:x2] = out.astype(np.uint8)
    return base_bgr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--person", default="assets/person.jpg")
    p.add_argument("--top", default="assets/top_rgba.png")
    p.add_argument("--cx", type=float, default=0.50, help="中心X (0..1)")

    # ✅ 首基準：首から下へどれくらい下げるか（H比）
    p.add_argument("--y", type=float, default=0.10, help="首から下へのYオフセット（H比）例:0.06〜0.12")

    p.add_argument("--w", type=float, default=1.00, help="幅（人物比）")
    p.add_argument("--angle", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--out", default="outputs/tryon_tmp.jpg")
    p.add_argument("--person_rgba", default=None)
    args = p.parse_args()

    person = cv2.imread(args.person, cv2.IMREAD_COLOR)
    if person is None:
        raise FileNotFoundError(f"Person image not found: {args.person}")
    H, W = person.shape[:2]

    # --- neck_y 推定 ---
    neck_y = int(H * 0.20)  # fallback
    mask_u8 = None
    if args.person_rgba and os.path.exists(args.person_rgba):
        rgba = cv2.imread(args.person_rgba, cv2.IMREAD_UNCHANGED)
        if rgba is not None and rgba.shape[2] == 4:
            mask_u8 = alpha_mask_from_rgba(rgba, thr=10)
            if mask_u8 is not None:
                neck_y = estimate_neck_y_from_mask(mask_u8)

    # 推定クリップ（下すぎると顎に被りやすい / 上すぎると不自然）
    neck_y = min(neck_y, int(H * 0.24))
    neck_y = max(neck_y, int(H * 0.12))

    print("DEBUG neck_y(px) =", neck_y, "/ H =", H, "=> ratio =", round(neck_y / H, 3))

    # --- 胴体だけ「元の服を消す（下地化）」処理（rembgマスク使用） ---
    if args.person_rgba and os.path.exists(args.person_rgba):
        rgba = cv2.imread(args.person_rgba, cv2.IMREAD_UNCHANGED)
        if rgba is not None and rgba.ndim == 3 and rgba.shape[2] == 4:
            a = rgba[:, :, 3]
            person_mask = (a > 10).astype(np.uint8) * 255

            ys, xs = np.where(person_mask > 0)
            if ys.size > 0:
                y0, y1 = int(ys.min()), int(ys.max())
                x0, x1 = int(xs.min()), int(xs.max())
                bbox_h = max(1, (y1 - y0 + 1))

                # 胴体帯（25%〜75%）
                t0 = int(y0 + bbox_h * 0.25)
                t1 = int(y0 + bbox_h * 0.75)

                t0 = max(0, min(H - 1, t0))
                t1 = max(0, min(H, t1))
                if t1 > t0:
                    torso = np.zeros((H, W), dtype=np.uint8)
                    torso[t0:t1, x0:x1] = person_mask[t0:t1, x0:x1]

                    torso = cv2.GaussianBlur(torso, (0, 0), sigmaX=18)
                    m = torso.astype(np.float32) / 255.0
                    m3 = np.dstack([m, m, m])

                    blur = cv2.GaussianBlur(person, (0, 0), sigmaX=6)

                    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV).astype(np.float32)
                    hsv[:, :, 1] *= 0.35
                    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.92 + 12, 0, 255)
                    neutral = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

                    neutral_f = neutral.astype(np.float32)
                    neutral_f = neutral_f * 0.90 + 15.0

                    person_f = person.astype(np.float32)
                    person = (person_f * (1.0 - m3) + neutral_f * m3).astype(np.uint8)

                    # --- 裾ゾーン強化（60%〜80%） ---
                    hem0 = int(y0 + bbox_h * 0.60)
                    hem1 = int(y0 + bbox_h * 0.80)
                    hem0 = max(0, min(H - 1, hem0))
                    hem1 = max(0, min(H, hem1))

                    if hem1 > hem0:
                        hem = np.zeros((H, W), dtype=np.uint8)
                        hem[hem0:hem1, x0:x1] = person_mask[hem0:hem1, x0:x1]

                        hem = cv2.GaussianBlur(hem, (0, 0), sigmaX=22)
                        mh = hem.astype(np.float32) / 255.0
                        mh3 = np.dstack([mh, mh, mh])

                        neutral2 = neutral.astype(np.float32)
                        neutral2 = neutral2 * 0.88 + 22.0

                        person = (person.astype(np.float32) * (1.0 - mh3) + neutral2 * mh3).astype(np.uint8)

    # --- top 読み込み ---
    top = cv2.imread(args.top, cv2.IMREAD_UNCHANGED)
    if top is None:
        raise FileNotFoundError(f"Top image not found: {args.top}")
    if top.ndim != 3 or top.shape[2] != 4:
        raise ValueError("Top image must be RGBA (transparent PNG).")

    # --- 事前のalpha軽処理（軽め） ---
    a0 = top[:, :, 3].copy()
    k0 = np.ones((3, 3), np.uint8)
    a0 = cv2.erode(a0, k0, iterations=1)
    a0 = cv2.GaussianBlur(a0, (0, 0), sigmaX=0.8)
    top[:, :, 3] = a0

    # --- リサイズ（幅を人物W*args.wに合わせる） ---
    target_w = int(W * float(args.w))
    scale = target_w / max(1, top.shape[1])
    target_h = int(top.shape[0] * scale)
    interp = cv2.INTER_AREA if target_w < top.shape[1] else cv2.INTER_CUBIC

    # premultiply → resize → rotate(premult) → unpremultiply
    top_f = top.astype(np.float32)
    a1 = top_f[:, :, 3:4] / 255.0
    top_f[:, :, :3] *= a1

    top_resized = cv2.resize(top_f, (target_w, target_h), interpolation=interp)
    top_rot_f = rotate_rgba_premult(top_resized, float(args.angle))

    # ★ divide-by-zero warning を出さない unpremultiply
    a2 = top_rot_f[:, :, 3:4] / 255.0
    eps = 1e-6
    rgb = top_rot_f[:, :, :3]
    out_rgb = np.zeros_like(rgb)
    np.divide(rgb, a2, out=out_rgb, where=(a2 > eps))
    top_rot_f[:, :, :3] = out_rgb

    # 仕上げ：alphaを少しだけ縮めて境界をきれいに
    alpha_u8 = np.clip(top_rot_f[:, :, 3], 0, 255).astype(np.uint8)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha_u8 = cv2.erode(alpha_u8, k2, iterations=1)
    alpha_u8 = cv2.GaussianBlur(alpha_u8, (0, 0), 0.8)
    top_rot_f[:, :, 3] = alpha_u8.astype(np.float32)

    top_rot = np.clip(top_rot_f, 0, 255).astype(np.uint8)

    # --- bbox基準で配置 ---
    th, tw = top_rot.shape[:2]
    bb = alpha_bbox(top_rot, thr=10)

    # ★襟位置補正：鎖骨に落ちやすいのを「少し上げる」
    collar_lift = int(H * 0.015)  # 1.5%H 上げ（必要なら 0.010〜0.025）

    if bb is None:
        x = int(W * float(args.cx) - tw / 2)
        y = int(neck_y + H * float(args.y) - collar_lift)
    else:
        left, top_y, right, bottom = bb
        bb_cx = (left + right) / 2.0
        x = int(W * float(args.cx) - bb_cx)
        y = int(neck_y + H * float(args.y) - top_y - collar_lift)

    comp = overlay_rgba(person, top_rot, x, y, alpha_scale=float(args.alpha))

    ensure_dir(args.out)
    cv2.imwrite(args.out, comp)
    print(f"Saved: {args.out}")
    print(f"Params: cx={args.cx}, y={args.y}, w={args.w}, angle={args.angle}, alpha={args.alpha}")


if __name__ == "__main__":
    main()
