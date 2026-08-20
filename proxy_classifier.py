import torch
import numpy as np
from sklearn import svm
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

# 1. 載入 DINOv3 模型 (與你之前跑攻擊時一樣)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16', pretrained=False)
model.load_state_dict(torch.load('dinov3_vits16.pth', map_location='cpu'))
model = model.to(device)
model.eval()

# 2. 影像預處理
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_features(image_paths):
    features = []
    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            feat = model(img_t)
            features.append(feat.cpu().numpy().flatten())
    return np.array(features)

# ---------------------------------------------------------
# 3. 準備數據路徑 (請填入你的資料夾路徑)
# ---------------------------------------------------------
import os

# 1. 設定資料夾路徑
REAL_DIR = "dataset_faces/real"
FAKE_DIR = "dataset_faces/fake"
ADV_DIR  = "dataset_faces/adv"  # 這是你存 PGD 攻擊後影像的地方




# 2. 自動抓取「檔案清單」 (這步是關鍵！)
# 這裡用 sorted 確保順序固定，以免訓練集跟測試集搞混
all_real_files = sorted([os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) if f.endswith('.jpg')])
all_fake_files = sorted([os.path.join(FAKE_DIR, f) for f in os.listdir(FAKE_DIR) if f.endswith('.jpg')])
all_adv_files  = sorted([os.path.join(ADV_DIR, f) for f in os.listdir(ADV_DIR) if f.endswith('.jpg')])

# 3. 分配給 Proxy Classifier 的變數
# 建議做法：
# 用前 800 張 Real + 800 張 Fake 來「訓練」SVM
# 用剩下的影像來測試 ASR (攻擊成功率)

real_paths = all_real_files
fake_paths = all_fake_files

# 測試用：直接拿你那 1000 張攻擊後的圖片來跑
adv_paths = all_adv_files 

print(f"📊 數據載入完畢：")
print(f" - 真臉特徵提取總數: {len(real_paths)} 張")
print(f" - 假臉特徵提取總數: {len(fake_paths)} 張")
print(f" - 測試用攻擊臉: {len(adv_paths)} 張")

print("🏗️ 正在提取特徵...")
feat_real = get_features(real_paths)
feat_fake = get_features(fake_paths)
feat_adv  = get_features(adv_paths)




from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter

# ---------------------------------------------------------
# 4. 終極修正版：Proxy Classifier 診斷與訓練
# ---------------------------------------------------------

# A. 數據準備 (先 StandardScale 放大細微特徵，再 L2 正規化)
scaler = StandardScaler()
# 為了避免資料洩漏，嚴格來說只能用訓練集 fit
train_features = np.vstack([feat_real[:800], feat_fake[:800]])
scaler.fit(train_features)

feat_real_l2 = normalize(scaler.transform(feat_real), norm='l2')
feat_fake_l2 = normalize(scaler.transform(feat_fake), norm='l2')
feat_adv_l2  = normalize(scaler.transform(feat_adv), norm='l2')

train_x = np.vstack([feat_real_l2[:800], feat_fake_l2[:800]])
train_y = np.array([0]*800 + [1]*800)

test_real_raw = feat_real_l2[800:]
test_fake_raw = feat_fake_l2[800:]
test_x = np.vstack([test_real_raw, test_fake_raw])
test_y = np.array([0]*len(test_real_raw) + [1]*len(test_fake_raw))

# B. 改用更強大的多層感知機 (MLP) 也就是小型神經網路
print("🤖 正在訓練 Proxy Classifier (使用 MLP 神經網路)...")
clf = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),       # 稍微簡化網路，避免在小數據集上過擬合
    activation='relu',
    solver='adam',
    alpha=0.1,                          # 提高 L2 正規化強度，防止死記硬背
    batch_size=16,
    learning_rate_init=0.0005,           # 稍微降低初始學習率，讓收斂更穩定
    max_iter=800,                        # 給它足夠的迭代次數
    early_stopping=False,                # ❌ 關閉早停，強迫模型去挖掘細微特徵
    random_state=42,
    tol=1e-4,                            # ✅ 損失函數優化容忍度
    n_iter_no_change=30
)
clf.fit(train_x, train_y)
print(f"🔍 MLP 訓練完成，實際迭代次數: {clf.n_iter_}")

# C. 診斷預測分佈 
test_preds = clf.predict(test_x)
adv_preds = clf.predict(feat_adv_l2)

print(f"DEBUG - 測試集預測分佈: {Counter(test_preds)}")
print(f"DEBUG - 攻擊集預測分佈: {Counter(adv_preds)}")

# D. 最終數據
clean_acc = clf.score(test_x, test_y)
asr = np.mean(adv_preds == 0) # 假臉變真臉

print("\n" + "="*30)
print("📊 修正後的實驗報告")
print("="*30)
print(f"✅ 原始測試集準確率: {clean_acc*100:.2f}%")
print(f"🎯 攻擊成功率 (ASR): {asr*100:.2f}%")
print("="*30)