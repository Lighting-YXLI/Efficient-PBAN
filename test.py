# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from scipy.stats import spearmanr, pearsonr,kendalltau
from sklearn.metrics import mean_squared_error
import argparse
import os
from multidataloader import test_loader  # 确保 val_loader 加载你的 test.txt 数据
from EfficientPBAN import Net

def compute_metrics(gt_scores, pred_scores):
    srocc = spearmanr(gt_scores, pred_scores)[0]
    plcc = pearsonr(gt_scores, pred_scores)[0]
    krcc = kendalltau(gt_scores, pred_scores)[0]
    rmse = np.sqrt(mean_squared_error(gt_scores, pred_scores))
    return srocc, plcc, krcc,rmse

def test(model_path, output_txt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    checkpoint = torch.load(model_path, map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data1, data2, target in test_loader:
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            output = model(data1,data2)

            all_preds.extend(output.cpu().numpy().tolist())
            all_labels.extend(target.cpu().numpy().tolist())

    # 计算指标
    srocc, plcc, krcc,rmse = compute_metrics(np.array(all_labels), np.array(all_preds))
    print(f"SROCC: {srocc:.4f}, PLCC: {plcc:.4f}, KRCC: {krcc:.4f},RMSE: {rmse:.4f}")

    # 写入预测文件
    with open(output_txt, "w") as f:
        f.write("GT_Score\tPredicted_Score\n")
        for gt, pred in zip(all_labels, all_preds):
            f.write(f"{gt:.4f}\t{pred:.4f}\n")
        f.write(f"\nSROCC: {srocc:.4f}, PLCC: {plcc:.4f}, KRCC: {krcc:.4f},RMSE: {rmse:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test IQA model on test set")
    parser.add_argument('--model_path', type=str, default='C:/Users/18810/Desktop/PBAN/nriqa/results/FRnewres50_123_ISRGenQ.pth',
                        help='Path to trained model (.pth)')
    parser.add_argument('--output_txt', type=str, default='C:/Users/18810/Desktop/PBAN/nriqa/results/FRnewres50_123_ISRGenQ_test.txt',
                        help='File to save predictions and metrics')
    args = parser.parse_args()

    test(args.model_path, args.output_txt)
