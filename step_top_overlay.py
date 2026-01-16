import argparse
import os
import cv2
import numpy as np


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def overlay_rgba(base_bgr, top_rgba, x, y, alpha_scale=1.0):
    """base_bgr(H,W,3) に top_rgba(h,w,4) を (x,y) に合成"""
    H, W = base_bgr.shape[:2]
    h, w = top_rgba.shape[:2]

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
    top_crop = top_rgba[top_y1:top_y2, top_x1:top_x2].astype(np.float32)

    alpha = (top_crop[:, :, 3] / 255.0) * float(alpha_scale)
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha3 = np.dstack([alpha, alpha, alpha])

    top_rgb = top_crop[:, :, :3]
    out = roi * (1.0 - alpha3) + top_rgb * alpha3
    base_bgr[y1:y2, x1:x2] = out.astype(np.uint8)
    return base_bgr


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


def rotate_rgba(img_rgba, angle_deg):
    """RGBA画像を中心回転して、全体が入るサイズで返す"""
    if abs(angle_deg) < 1e-6:
        return img_rgba
    h, w = img_rgba.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        img_rgba,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0, 0),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--person", default="assets/person.jpg")
    p.add_argument("--top", default="assets/top_rgba.png")
    p.add_argument("--cx", type=float, default=0.50, help="中心X (0..1)")
    p.add_argument("--y", type=float, default=0.28, help="上端Y (0..1)")
    p.add_argument("--w", type=float, default=1.00, help="幅（人物比）")
    p.add_argument("--angle", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--out", default="outputs/tryon_tmp.jpg")
    args = p.parse_args()

    person = cv2.imread(args.person, cv2.IMREAD_COLOR)
    if person is None:
        raise FileNotFoundError(f"Person image not found: {args.person}")
    H, W = person.shape[:2]

    top = cv2.imread(args.top, cv2.IMREAD_UNCHANGED)
    if top is None:
        raise FileNotFoundError(f"Top image not found: {args.top}")
    if top.ndim != 3 or top.shape[2] != 4:
        raise ValueError("Top image must be RGBA (transparent PNG).")

    # リサイズ（幅を人物W*args.wに合わせる）
    target_w = int(W * float(args.w))
    scale = target_w / top.shape[1]
    target_h = int(top.shape[0] * scale)
    interp = cv2.INTER_AREA if target_w < top.shape[1] else cv2.INTER_CUBIC
    top_resized = cv2.resize(top, (target_w, target_h), interpolation=interp)

    # 回転
    top_rot = rotate_rgba(top_resized, float(args.angle))

    # bbox基準で配置
    th, tw = top_rot.shape[:2]
    bb = alpha_bbox(top_rot, thr=10)
    if bb is None:
        x = int(W * float(args.cx) - tw / 2)
        y = int(H * float(args.y))
    else:
        left, top_y, right, bottom = bb
        bb_cx = (left + right) / 2.0
        x = int(W * float(args.cx) - bb_cx)
        y = int(H * float(args.y) - top_y)

    comp = overlay_rgba(person, top_rot, x, y, alpha_scale=float(args.alpha))

    ensure_dir(args.out)
    cv2.imwrite(args.out, comp)
    print(f"Saved: {args.out}")
    print(f"Params: cx={args.cx}, y={args.y}, w={args.w}, angle={args.angle}, alpha={args.alpha}")


if __name__ == "__main__":
    main()
