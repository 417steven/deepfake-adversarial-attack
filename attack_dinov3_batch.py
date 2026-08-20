import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from tqdm import tqdm 
from torchvision.utils import save_image

# ==========================================
# 1. 環境設定與模型載入
# ==========================================
REAL_DIR = "dataset_faces/real"
FAKE_DIR = "dataset_faces/fake"
NUM_SAMPLES = 50  # 隨機抽樣 50 張

print("正在載入 DINOv3 本地權重...")
model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16', pretrained=False)
model.load_state_dict(torch.load('dinov3_vits16.pth', map_location='cpu'))
model.eval().cuda() # A6000 準備就緒

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return preprocess(img).unsqueeze(0).cuda()

# ==========================================
# 2. 準備基準線：計算穩定的真實特針質心
# ==========================================


all_reals = [os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) if f.endswith('.jpg')]
all_fakes = [os.path.join(FAKE_DIR, f) for f in os.listdir(FAKE_DIR) if f.endswith('.jpg')]

# 💥 升級 1：使用所有的 Real 圖片來計算「絕對真實質心」
print(f"正在計算 {len(all_reals)} 張 Real 樣本的絕對真實質心...")
real_features = []
with torch.no_grad():
    # 這裡也可以加上 tqdm，讓你看到計算質心的進度
    for path in tqdm(all_reals, desc="計算質心中"):
        feat = model(normalize(load_image(path)))
        real_features.append(feat)
    centroid_feature = torch.mean(torch.cat(real_features, dim=0), dim=0, keepdim=True)

# 💥 升級 2：攻擊所有的 Fake 圖片 (不再隨機抽樣)
# 注意：跑 1000 張大約需要 30~40 分鐘，請確保伺服器連線穩定
iters = 40
epsilon = 8/255
alpha = 2/255

initial_losses = []
final_losses = []

print(f"\n🚀 開始對全部 {len(all_fakes)} 張 Fake 進行 PGD 地毯式攻擊...")

# 使用 tqdm 包住 all_fakes，這樣就會有漂亮的進度條跟預估剩餘時間 (ETA)
for fake_path in tqdm(all_fakes, desc="PGD 攻擊進度"):
    fake_img = load_image(fake_path)
    
    # 計算初始 Loss
    with torch.no_grad():
        init_feat = model(normalize(fake_img))
        init_loss = F.mse_loss(init_feat, centroid_feature).item()
        initial_losses.append(init_loss)
        
    # PGD 迭代
    adv_image = fake_img.clone().detach().requires_grad_(True)
    for i in range(iters):
        adv_feature = model(normalize(adv_image))
        loss = F.mse_loss(adv_feature, centroid_feature)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            adv_image = adv_image - alpha * adv_image.grad.sign()
            eta = torch.clamp(adv_image - fake_img, min=-epsilon, max=epsilon)
            adv_image = torch.clamp(fake_img + eta, min=0, max=1).requires_grad_(True)
            
    # 計算最終 Loss
    with torch.no_grad():
        final_feat = model(normalize(adv_image))
        final_loss = F.mse_loss(final_feat, centroid_feature).item()
        final_losses.append(final_loss)
    save_name = os.path.basename(fake_path) # 抓取原始假臉的檔名 (例如 001.jpg)
    save_path = os.path.join("dataset_faces/adv", save_name)
    save_image(adv_image, save_path) # 存檔！



# ==========================================
# 4. 統計結果與分佈圖儲存
# ==========================================
print("\n" + "="*30)
print("📊 批量攻擊實驗統計報告")
print("="*30)
print(f"攻擊前 (Original Fake) 平均 Loss: {np.mean(initial_losses):.6f} (±{np.std(initial_losses):.6f})")
print(f"攻擊後 (Poisoned Fake) 平均 Loss: {np.mean(final_losses):.6f} (±{np.std(final_losses):.6f})")
print(f"最大特徵縮減幅度: {(1 - np.mean(final_losses)/np.mean(initial_losses))*100:.2f}%")
print("="*30)

# 繪製學術直方圖 (Histogram)
plt.figure(figsize=(10, 6))

# 使用透明度 alpha 讓重疊區域清晰可見
plt.hist(initial_losses, bins=15, alpha=0.6, label='Before Attack (Original Fake)', edgecolor='black')
plt.hist(final_losses, bins=15, alpha=0.6, label='After Attack (Adversarial Fake)', edgecolor='black')

plt.title('Distribution of Feature MSE Loss Before & After Batch PGD Attack (DINOv3)', fontsize=14, fontweight='bold')
plt.xlabel('Feature MSE Loss (Distance to Real Centroid)', fontsize=12)
plt.ylabel('Frequency (Number of Images)', fontsize=12)
plt.axvline(np.mean(initial_losses), color='darkred', linestyle='--', linewidth=2, label=f'Mean Before ({np.mean(initial_losses):.3f})')
plt.axvline(np.mean(final_losses), color='darkblue', linestyle='--', linewidth=2, label=f'Mean After ({np.mean(final_losses):.3f})')

plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# 儲存圖片
plt.savefig('dinov3_batch_attack_histogram.png', dpi=300)
print("\n✅ 統計完成！高解析度分佈圖已儲存為 dinov3_batch_attack_histogram.png")