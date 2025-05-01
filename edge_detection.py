# ==============================
# stage1_edge_detection.py
# ==============================
# 实验目的：实现多种经典边缘检测算法，对图像进行边缘提取，并结合 ground truth 进行精度评估。
# 对比指标包括：Precision（查准率）、Recall（查全率）、F1-score（调和平均）、IoU（交并比）。

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from skimage.filters import prewitt
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import matplotlib.gridspec as gridspec

def edge_detection_methods(img_gray):
    # 执行多种边缘检测算法，返回二值化后的边缘图像字典
    sobel = cv2.Sobel(img_gray, cv2.CV_64F, 1, 1, ksize=3)  # Sobel 算子提取梯度边缘
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)         # Laplacian 算子
    canny = cv2.Canny(img_gray, 100, 200)                   # Canny 边缘检测
    prewitt_edge = prewitt(img_gray)                       # Prewitt 滤波器
    prewitt_edge = (prewitt_edge * 255).astype(np.uint8)   # 归一化到 0-255

    # 返回处理后的二值图像（可调阈值）用于后续评估
    return {
        "Sobel": np.uint8(np.abs(sobel) > 50),
        "Laplacian": np.uint8(np.abs(laplacian) > 50),
        "Canny": canny // 255,
        "Prewitt": np.uint8(prewitt_edge > 30)
    }

def visualize_edges(edges, img_id, split, gt_mask=None):
    # 将每种方法的边缘检测结果绘制为图像并保存，方便人工观察对比
    plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1])

    # Ground Truth 放在左侧，占据两行
    if gt_mask is not None:
        ax = plt.subplot(gs[:, 0])  # 占两行
        ax.imshow(gt_mask, cmap='gray')
        ax.set_title("Ground Truth")
        ax.axis('off')

    # 右侧 2x2 布局显示算法结果
    method_names = ["Sobel", "Laplacian", "Prewitt", "Canny"]
    positions = [(0, 1), (0, 2), (1, 1), (1, 2)]

    for (method, pos) in zip(method_names, positions):
        edge = edges[method]
        ax = plt.subplot(gs[pos[0], pos[1]])
        ax.imshow(edge, cmap='gray')
        ax.set_title(f"{method} Edge")
        ax.axis('off')

    plt.suptitle(f"{split.upper()} Image {img_id} Edge Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs("results/edges", exist_ok=True)
    plt.savefig(f"results/edges/edge_{img_id}.png")
    plt.close()

def load_bsds500_image(image_path):
    # 从指定路径读取图像，并转换为灰度图
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def load_ground_truth_mask(mat_path):
    mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    gts = mat['groundTruth']

    # 确保 gts 是列表
    if not isinstance(gts, (list, np.ndarray)):
        gts = [gts]

    masks = []
    for gt in gts:
        # gt 是一个 groundTruth 对象，有属性 Boundaries
        if hasattr(gt, 'Boundaries'):
            mask = np.array(gt.Boundaries)
        else:
            raise ValueError("groundTruth entry missing 'Boundaries' field")
        masks.append(mask)

    combined = np.max(np.stack(masks, axis=0), axis=0)
    return (combined * 255).astype(np.uint8)


def evaluate_edge(pred, gt):
    # 计算四种边缘检测评估指标，输入为 flatten 后的一维二值数组
    pred = pred.flatten() > 0
    gt = gt.flatten() > 0
    return {
        "Precision": precision_score(gt, pred, zero_division=0),
        "Recall": recall_score(gt, pred, zero_division=0),
        "F1": f1_score(gt, pred, zero_division=0),
        "IoU": jaccard_score(gt, pred, zero_division=0)
    }

def process_bsds500_dataset():
    # 遍历 BSDS500 数据集下的 train/val/test 子集，评估各方法边缘性能
    base_path = "./dataset/BSDS500"
    splits = ["train", "val", "test"]
    log_path = "results/edge_eval_metrics.txt"  # 保存每张图像评估结果
    summary_path = "results/edge_eval_summary.txt"  # 保存平均指标汇总

    # 用于累加各方法的所有指标值
    all_metrics = {
        "Sobel": {"Precision": [], "Recall": [], "F1": [], "IoU": []},
        "Laplacian": {"Precision": [], "Recall": [], "F1": [], "IoU": []},
        "Prewitt": {"Precision": [], "Recall": [], "F1": [], "IoU": []},
        "Canny": {"Precision": [], "Recall": [], "F1": [], "IoU": []},
    }

    total_images = sum(
                len([f for f in os.listdir(os.path.join(base_path, "images", split)) if f.endswith(".jpg")])
                for split in splits if os.path.exists(os.path.join(base_path, "images", split))
            )
    processed_count = 0

    with open(log_path, 'w') as log:
        log.write("ImageID\tMethod\tPrecision\tRecall\tF1\tIoU\n")

        for split in splits:
            image_dir = os.path.join(base_path, "images", split)
            gt_dir = os.path.join(base_path, "ground_truth", split)
            if not os.path.exists(image_dir):
                continue

            for fname in os.listdir(image_dir):
                if not fname.endswith(".jpg"):
                    continue

                img_id = os.path.splitext(fname)[0]
                img_path = os.path.join(image_dir, fname)
                gt_path = os.path.join(gt_dir, f"{img_id}.mat")
                if not os.path.exists(gt_path):
                    continue

                processed_count += 1
                percent = processed_count / total_images * 100
                print(f"[{percent:.1f}%] Processing {split}/{img_id}...")
                img, gray = load_bsds500_image(img_path)
                gt_mask = load_ground_truth_mask(gt_path)
                edges = edge_detection_methods(gray)
                visualize_edges(edges, img_id, split, gt_mask)

                for method, edge in edges.items():
                    metrics = evaluate_edge(edge, gt_mask)

                    # 写入单图像日志
                    log.write(f"{img_id}\t{method}\t{metrics['Precision']:.4f}\t{metrics['Recall']:.4f}\t{metrics['F1']:.4f}\t{metrics['IoU']:.4f}\n")

                    # 累计每个指标值
                    for key in metrics:
                        all_metrics[method][key].append(metrics[key])

    # 计算每种方法的平均指标并写入 summary 文件
    with open(summary_path, 'w') as fout:
        fout.write("Method\tAvg_Precision\tAvg_Recall\tAvg_F1\tAvg_IoU\n")
        for method in all_metrics:
            avg_prec = np.mean(all_metrics[method]["Precision"])
            avg_recall = np.mean(all_metrics[method]["Recall"])
            avg_f1 = np.mean(all_metrics[method]["F1"])
            avg_iou = np.mean(all_metrics[method]["IoU"])
            fout.write(f"{method}\t{avg_prec:.4f}\t{avg_recall:.4f}\t{avg_f1:.4f}\t{avg_iou:.4f}\n")
        print(f"\n✅ 统计完成，结果保存至 {summary_path}")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    process_bsds500_dataset()
