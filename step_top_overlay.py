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
    # --- 黒フチ対策：境界（rim）だけを処理する版 ---
    top_u8 = np.clip(top_crop, 0, 255).astype(np.uint8)
    a = top_u8[:, :, 3]

    # 1) まず「不透明マスク」
    m = (a > 0).astype(np.uint8)  # 0/1

    # 2) 境界だけ作る（dilate - erode）
    k = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(m, k, iterations=1)
    ero = cv2.erode(m, k, iterations=1)
    rim = ((dil - ero) > 0).astype(np.uint8) * 255  # 0/255

    # 3) rim 部分だけ RGB を inpaint（黒マットを消す）
    if rim.any():
        rgb = top_u8[:, :, :3]
        rgb_fixed = cv2.inpaint(rgb, rim, 3, cv2.INPAINT_TELEA)
        top_u8[:, :, :3] = rgb_fixed

    top_crop = top_u8.astype(np.float32)

    # alpha（服の透明度）
    alpha = (top_crop[:, :, 3] / 255.0) * float(alpha_scale)

    # 4) alpha収縮（黒フチ削り）※やりすぎると服が細るので 1〜2推奨
    a8 = (alpha * 255).astype(np.uint8)
    a8 = cv2.erode(a8, k, iterations=1)

    # 5) フェザー（自然に）
    a8 = cv2.GaussianBlur(a8, (0, 0), sigmaX=0.8)

    alpha = a8.astype(np.float32) / 255.0
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
    """RGBA画像を中心回転（premultiplyで黒フチ防止）"""
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

    rgba = img_rgba.astype(np.float32)
    a = rgba[:, :, 3:4] / 255.0
    rgb = rgba[:, :, :3]

    # premultiply
    rgb_p = rgb * a

    # rotate premultiplied RGB + alpha separately
    rgb_rot = cv2.warpAffine(
        rgb_p, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    a_rot = cv2.warpAffine(
        (a * 255.0).astype(np.uint8), M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32) / 255.0
    a_rot = a_rot[:, :, None]

    # un-premultiply（0割回避）
    eps = 1e-6
    rgb_out = rgb_rot / np.maximum(a_rot, eps)

    out = np.dstack([
        np.clip(rgb_out, 0, 255),
        np.clip(a_rot * 255.0, 0, 255),
    ]).astype(np.uint8)

    return out


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
    p.add_argument("--person_rgba", default=None)
    args = p.parse_args()

    print("DEBUG person_rgba arg =", args.person_rgba)

    person = cv2.imread(args.person, cv2.IMREAD_COLOR)
    if person is None:
        raise FileNotFoundError(f"Person image not found: {args.person}")
    H, W = person.shape[:2]

    # --- 胴体だけ元の服を弱める処理（rembgマスク使用） ---
    if args.person_rgba and os.path.exists(args.person_rgba):
        rgba = cv2.imread(args.person_rgba, cv2.IMREAD_UNCHANGED)
        if rgba is not None and rgba.shape[2] == 4:
            alpha = rgba[:, :, 3]

            mask = (alpha > 10).astype(np.uint8) * 255
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                y0, y1 = ys.min(), ys.max()
                x0, x1 = xs.min(), xs.max()

                bbox_h = y1 - y0 + 1
                t0 = int(y0 + bbox_h * 0.25)
                t1 = int(y0 + bbox_h * 0.80)

                torso_mask = np.zeros((H, W), dtype=np.uint8)
                torso_mask[t0:t1, x0:x1] = mask[t0:t1, x0:x1]

                torso_mask = cv2.GaussianBlur(torso_mask, (31, 31), 0)

                person_dark = (person * 0.75).astype(np.uint8)
                m = torso_mask.astype(np.float32) / 255.0
                m3 = np.dstack([m, m, m])

                person = (person * (1 - m3) + person_dark * m3).astype(np.uint8)

    # ↓↓↓ この下に既存の top 合成処理が続く（触らなくてOK）

    top = cv2.imread(args.top, cv2.IMREAD_UNCHANGED)
    if top is None:
        raise FileNotFoundError(f"Top image not found: {args.top}")
    if top.ndim != 3 or top.shape[2] != 4:
        raise ValueError("Top image must be RGBA (transparent PNG).")
    # --- B) 黒フチ（ハロ）対策：alpha を収縮して少しぼかす ---
    a = top[:, :, 3].copy()

    # 透明境界を少し削る（やりすぎ注意）
    k = np.ones((3, 3), np.uint8)
    a = cv2.erode(a, k, iterations=1)

    # 境界をなめらかに
    a = cv2.GaussianBlur(a, (0, 0), sigmaX=0.8)

    # alpha を戻す
    top[:, :, 3] = a

    # リサイズ（幅を人物W*args.wに合わせる）
    target_w = int(W * float(args.w))
    scale = target_w / top.shape[1]
    target_h = int(top.shape[0] * scale)
    interp = cv2.INTER_AREA if target_w < top.shape[1] else cv2.INTER_CUBIC

    # ========= 黒フチ対策：premultiply → resize/rotate → unpremultiply =========
    top_f = top.astype(np.float32)
    a = top_f[:, :, 3:4] / 255.0                      # (H,W,1) alpha 0..1
    top_f[:, :, :3] *= a                               # premultiply RGB

    top_resized = cv2.resize(top_f, (target_w, target_h), interpolation=interp)

    top_rot_f = rotate_rgba(top_resized, float(args.angle))  # floatのまま回転

    # unpremultiply（alphaがある所だけRGBを戻す）
    a2 = top_rot_f[:, :, 3:4] / 255.0
    eps = 1e-6
    top_rot_f[:, :, :3] = np.where(a2 > eps, top_rot_f[:, :, :3] / a2, 0.0)

    # 仕上げ：alphaを少しだけ縮めて、境界の汚れを抑える（任意だけど効く）
    alpha_u8 = np.clip(top_rot_f[:, :, 3], 0, 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha_u8 = cv2.erode(alpha_u8, k, iterations=1)
    alpha_u8 = cv2.GaussianBlur(alpha_u8, (0, 0), 0.8)
    top_rot_f[:, :, 3] = alpha_u8.astype(np.float32)

    top_rot = np.clip(top_rot_f, 0, 255).astype(np.uint8)
    # ========= 黒フチ対策ここまで =========

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
