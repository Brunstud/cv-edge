# ==============================
# stage2::pyramid.py
# ==============================
# 实验目的：构建图像的高斯金字塔和拉普拉斯金字塔，并从拉普拉斯金字塔中恢复图像。
# 比较不同层数下图像重建质量（PSNR、SSIM、MSE），验证多尺度金字塔在图像压缩与融合中的应用基础。

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

def build_gaussian_pyramid(img, levels):
    # 构建高斯金字塔：逐层下采样图像并保存到列表中
    pyramid = [img]  # 第0层为原图
    for _ in range(1, levels):
        img = cv2.pyrDown(img)  # 下采样，尺寸减半
        pyramid.append(img)  # 添加到金字塔列表
    return pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    # 构建拉普拉斯金字塔：每一层为相邻两层高斯图的差值
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        up = cv2.pyrUp(gaussian_pyramid[i + 1])  # 上采样下一层
        up = cv2.resize(up, (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))  # 对齐尺寸
        lap = cv2.subtract(gaussian_pyramid[i], up)  # 当前层减上采样结果，得到细节图
        laplacian_pyramid.append(lap)
    laplacian_pyramid.append(gaussian_pyramid[-1])  # 顶层图像无可比，直接保留
    return laplacian_pyramid

def reconstruct_from_laplacian(laplacian_pyramid):
    # 从拉普拉斯金字塔重建图像，逐层向上还原
    img = laplacian_pyramid[-1]  # 从顶层开始
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        img = cv2.pyrUp(img)  # 上采样还原尺寸
        img = cv2.resize(img, (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]))
        img = cv2.add(img, laplacian_pyramid[i])  # 加上该层细节图
    return img

def psnr(img1, img2):
    # 计算 PSNR（峰值信噪比），用于衡量还原质量
    return cv2.PSNR(img1, img2)

def compute_ssim(img1, img2):
    # 计算 SSIM（结构相似性指数），考虑图像感知质量
    return ssim(img1, img2, channel_axis=2, data_range=255)

def compute_mse(img1, img2):
    # 计算 MSE（均方误差），衡量像素差异
    return mean_squared_error(img1.flatten(), img2.flatten())

def process_kodak_dataset():
    # 处理 Kodak 数据集中的所有图像，进行金字塔重建与评估
    data_dir = "./dataset/Kodak"  # 数据路径
    out_dir = "results/kodak_pyramid"  # 结果输出路径
    os.makedirs(out_dir, exist_ok=True)
    image_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]

    result_log_path = os.path.join(out_dir, "metrics_results.txt")  # 指标输出文件
    with open(result_log_path, "w") as f_log:
        f_log.write("Filename\tLevels\tPSNR(dB)\tSSIM\tMSE\n")

        for fname in image_files:
            img_path = os.path.join(data_dir, fname)
            img = cv2.imread(img_path)
            print(f"Processing {fname}...")

            for levels in [3, 4, 5]:  # 实验不同金字塔层数对重建效果的影响
                g_pyr = build_gaussian_pyramid(img, levels)  # 构建高斯金字塔
                l_pyr = build_laplacian_pyramid(g_pyr)       # 构建拉普拉斯金字塔
                recon = reconstruct_from_laplacian(l_pyr)    # 图像重建

                # 多指标评估
                score_psnr = psnr(img, recon)
                score_ssim = compute_ssim(img, recon)
                score_mse = compute_mse(img, recon)

                print(f"  Levels: {levels}, PSNR: {score_psnr:.2f} dB, SSIM: {score_ssim:.4f}, MSE: {score_mse:.2f}")
                f_log.write(f"{fname}\t{levels}\t{score_psnr:.2f}\t{score_ssim:.4f}\t{score_mse:.2f}\n")

                # 保存重建图像
                recon_path = os.path.join(out_dir, f"{fname[:-4]}_recon_L{levels}.png")
                cv2.imwrite(recon_path, recon)

if __name__ == "__main__":
    process_kodak_dataset()
