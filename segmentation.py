# ==============================
# stage3::segmentation.py
# ==============================
# 实验目的：实现基于边缘检测的图像结构感知分割系统，包括：
# - 边缘检测（Canny/Sobel/Prewitt/Laplacian）
# - 形态学处理（开运算、膨胀）提升边缘连续性
# - 分水岭分割
# - 多指标评估（IoU、F1、Accuracy）

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import sobel, prewitt
from sklearn.metrics import jaccard_score, f1_score, accuracy_score

def detect_edges(gray, method):
    # 边缘检测算法选择
    if method == "canny":
        return cv2.Canny(gray, 100, 200)
    elif method == "sobel":
        return cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    elif method == "prewitt":
        return (prewitt(gray) * 255).astype(np.uint8)
    elif method == "laplacian":
        return cv2.Laplacian(gray, cv2.CV_64F)
    else:
        raise ValueError("Unsupported edge method")

def segment_with_watershed(img, edge):
    # 分水岭分割前预处理，包括形态学操作构造 marker
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 形态学开运算去噪
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 膨胀获取背景区域（增强连续性）
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 距离变换提取前景区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # 计算未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 标记前景区域并构造 marker
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 应用分水岭分割
    markers = cv2.watershed(img, markers)
    return markers

def evaluate_all_metrics(pred_mask, gt_mask):
    # 多指标评估：IoU, F1, Accuracy
    pred = (pred_mask > 0).astype(np.uint8).flatten()
    gt = (gt_mask > 0).astype(np.uint8).flatten()
    return {
        "IoU": jaccard_score(gt, pred),
        "F1": f1_score(gt, pred),
        "Accuracy": accuracy_score(gt, pred)
    }

def process_dataset(image_dir, mask_dir, prefix_img, prefix_mask, ext_img, ext_mask, log_file):
    methods = ["canny", "sobel", "prewitt", "laplacian"]
    os.makedirs("results/segmentation", exist_ok=True)

    with open(log_file, 'a') as f_log:
        f_log.write("Filename\tMethod\tIoU\tF1\tAccuracy\n")

        for fname in os.listdir(image_dir):
            if not fname.endswith(ext_img):
                continue

            idx = fname.replace(prefix_img, "").replace(ext_img, "")
            img_path = os.path.join(image_dir, fname)
            mask_path = os.path.join(mask_dir, f"{prefix_mask}{idx}{ext_mask}")

            if not os.path.exists(mask_path):
                continue

            img = cv2.imread(img_path)
            gt_mask = cv2.imread(mask_path, 0)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            for method in methods:
                edge = detect_edges(gray, method)
                markers = segment_with_watershed(img.copy(), edge)
                pred_mask = (markers > 1).astype(np.uint8) * 255
                metrics = evaluate_all_metrics(pred_mask, gt_mask)

                print(f"{fname} | {method}: IoU = {metrics['IoU']:.4f}, F1 = {metrics['F1']:.4f}, Acc = {metrics['Accuracy']:.4f}")
                f_log.write(f"{fname}\t{method}\t{metrics['IoU']:.4f}\t{metrics['F1']:.4f}\t{metrics['Accuracy']:.4f}\n")

                out_path = f"results/segmentation/{method}_{idx}.png"
                cv2.imwrite(out_path, pred_mask)

def run_all():
    log_path = "results/segmentation/eval_metrics.txt"
    if os.path.exists(log_path):
        os.remove(log_path)

    process_dataset(
        image_dir="./dataset/horses/images",
        mask_dir="./dataset/horses/masks",
        prefix_img="image-",
        prefix_mask="mask-",
        ext_img=".png",
        ext_mask=".png",
        log_file=log_path
    )

    process_dataset(
        image_dir="./dataset/hw2data/imgs",
        mask_dir="./dataset/hw2data/gt",
        prefix_img="",
        prefix_mask="",
        ext_img=".jpg",
        ext_mask=".png",
        log_file=log_path
    )

if __name__ == "__main__":
    run_all()
