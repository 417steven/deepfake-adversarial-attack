# Adversarial Attacks on DINOv3-based Deepfake Detection

研究 DINOv3 自監督特徵在 deepfake 偵測任務上的**對抗穩健性 (adversarial robustness)**。

核心問題：若 deepfake 偵測器建立在 DINOv3 的特徵空間之上，只要在偽造人臉上加入人眼難以察覺的擾動，
把它的特徵向量推向「真實人臉特徵質心」，偵測器是否就會失效？

## 方法

1. **特徵質心基準線** — 用全部 real 樣本計算 DINOv3 特徵空間中的「絕對真實質心」。
2. **攻擊目標函數** — 最小化 `fake 樣本特徵` 與 `真實質心` 的距離，讓偽造樣本在特徵空間中偽裝成真實樣本。
3. **兩種攻擊設定**
   - *Per-sample (PGD)* — 針對單張影像最佳化專屬擾動（`attack_dinov3_batch.py`）。
   - *Universal (UAP)* — 最佳化一個對整個資料集通用的單一擾動（`universal_attack.py`）。
4. **代理分類器評估** — 用 DINOv3 特徵訓練 SVM 作為下游偵測器，量測攻擊前後的分類準確率落差（`proxy_classifier.py`）。
5. **擾動預算消融** — 比較 L∞ 預算 ε = 8/255、12/255、16/255 的攻擊強度與視覺可察覺度。

## 檔案說明

| 檔案 | 用途 |
|------|------|
| `extract_faces.py` | 用 MTCNN 從 FaceForensics++ 影片均勻抽幀並裁切人臉 (224×224) |
| `attack_dinov3_batch.py` | Per-sample PGD 攻擊，批次跑 50 張隨機樣本 |
| `universal_attack.py` | Universal adversarial perturbation 訓練 |
| `proxy_classifier.py` | DINOv3 特徵 + SVM 代理偵測器，評估攻擊成效 |
| `universal_loss_combined_ablation.py` | ε 預算消融實驗圖 |
| `plot_dynamics.py` / `plot_histogram_dark.py` / `plot_combined_dark.py` | 結果視覺化 |
| `universal_noise_dino.pt` | 訓練完成的 universal perturbation 張量 |

## 結果圖表

- `optimization_dynamics_dark.png` — 攻擊最佳化過程的 loss 收斂曲線
- `universal_loss_combined_ablation.png` — 不同 ε 預算下的 universal loss 比較
- `batch_attack_histogram_dark_final.png` — 攻擊前後特徵距離分佈
- `universal_noise_visualization*_255.png` — 各 ε 預算下的擾動視覺化

## 執行環境

需要 PyTorch + CUDA、`facenet-pytorch`、`scikit-learn`、`matplotlib`。
DINOv3 權重透過 `torch.hub` 載入 `facebookresearch/dinov3` 架構，再讀取本地 `dinov3_vits16.pth`。

## 未包含在此 repo 的檔案

以下因體積或授權因素未上傳（見 `.gitignore`）：

- `ffpp_videos/` — FaceForensics++ 原始影片（需自行向原作者申請授權取得）
- `dataset_faces/` — 抽取出的人臉資料集，可用 `extract_faces.py` 重新產生
- `dinov3_vits16.pth` — DINOv3 預訓練權重（86MB，由 Meta 官方發布）
- `deepfake_env/` — 本機虛擬環境

## 用途聲明

本專案為學術性的對抗穩健性研究，目的在於揭露以自監督特徵為基礎的 deepfake 偵測器之弱點，
以協助後續設計更穩健的偵測方法。請勿用於規避真實世界的內容真實性驗證系統。
