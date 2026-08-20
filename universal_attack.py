import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm 

# ==========================================
# 1. 環境設定與模型載入
# ==========================================
REAL_DIR = "dataset_faces/real"
FAKE_DIR = "dataset_faces/fake"

print("正在載入 DINOv3 本地權重...")
model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16', pretrained=False)
model.load_state_dict(torch.load('dinov3_vits16.pth', map_location='cpu'))
model.eval().cuda()

# 凍結模型權重，節省記憶體
for param in model.parameters():
    param.requires_grad = False

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return preprocess(img).unsqueeze(0).cuda()

# ==========================================
# 2. 計算穩定的真實特徵質心 (Centroid)
# ==========================================
all_reals = [os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) if f.endswith('.jpg')]
all_fakes = [os.path.join(FAKE_DIR, f) for f in os.listdir(FAKE_DIR) if f.endswith('.jpg')]

print(f"正在計算 {len(all_reals)} 張 Real 樣本的絕對真實質心...")
centroid_sum = None
for path in tqdm(all_reals, desc="計算質心中"):
    feat = model(normalize(load_image(path)))
    if centroid_sum is None:
        centroid_sum = feat.clone()
    else:
        centroid_sum += feat
centroid_feature = centroid_sum / len(all_reals)

# ==========================================
# 3. 訓練通用對抗雜訊 (Universal Adversarial Perturbation)
# ==========================================
epsilon = 16/255 #8/255  或 12/255
alpha = 1/255      # 學習率 (因為要看很多張圖，步伐可以稍微小一點)
epochs = 30      # 因為更新次數變少了，我們可以多跑幾輪讓它收斂
accumulation_steps = 32 # 🌟 累積 32 張圖片的梯度後才更新一次雜訊

print(f"\n🚀 開始訓練 Universal Noise，使用 {len(all_fakes)} 張 Fake 圖片...")

# 宣告全局的 Universal Noise (初始化為 0)，並開啟梯度追蹤
universal_noise = torch.zeros((1, 3, 224, 224), device='cuda', requires_grad=True)

# 確保一開始梯度是乾淨的
if universal_noise.grad is not None:
    universal_noise.grad.zero_()

loss_history = []

for epoch in range(epochs):
    epoch_loss = 0.0
    # 打亂 Fake 圖片順序，有助於泛化
    np.random.shuffle(all_fakes) 
    
    progress_bar = tqdm(enumerate(all_fakes), total=len(all_fakes), desc=f"Epoch {epoch+1}/{epochs} 訓練 UAP")
    for step, fake_path in progress_bar:
        fake_img = load_image(fake_path)
        
        # 將全局雜訊疊加到當前圖片上，並限制在 0~1 的合法圖片範圍
        adv_image = torch.clamp(fake_img + universal_noise, min=0, max=1)
        
        # 計算特徵距離 Loss
        adv_feature = model(normalize(adv_image))
        loss = F.mse_loss(adv_feature, centroid_feature)
        epoch_loss += loss.item()
        
        # 反向傳播 (此時梯度會自動「累加」到 universal_noise.grad 裡面)
        loss.backward()
        
        # 🌟 當累積滿 accumulation_steps 張，或是到了最後一張圖時，才執行更新
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(all_fakes):
            with torch.no_grad():
                # PGD 更新法則：這時候的 grad 是 32 張圖片梯度的總和 (多數決方向)
                universal_noise -= alpha * universal_noise.grad.sign()
                # 限制雜訊強度在 [-epsilon, epsilon] 之間
                universal_noise.clamp_(min=-epsilon, max=epsilon)
                
            # 更新完畢後，清空梯度，準備下一批累積
            universal_noise.grad.zero_()
            
        # 計算並顯示「當前 Epoch 的平均 Loss」，這才是我們判斷收斂的依據
        avg_loss = epoch_loss / (step + 1)
        progress_bar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})

    # 紀錄每個 Epoch 的最終平均 Loss 以供畫圖
    loss_history.append(avg_loss)

print("\n✅ Universal Noise 訓練完成！")

# 將訓練好的 Universal Noise 存下來，以後遇到新圖片直接掛上去就好！
torch.save(universal_noise.detach().cpu(), "universal_noise_dino.pt")

# ==========================================
# 4. 繪製並儲存訓練 Loss 曲線圖
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), loss_history, marker='o', linestyle='-', color='b', linewidth=2)
plt.title('Universal Attack Training Loss over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Average Feature MSE Loss', fontsize=12)
plt.xticks(range(1, epochs + 1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('universal_loss_curve.png', dpi=300)
print("📈 訓練 Loss 曲線圖已儲存為 universal_loss_curve.png")

# ==========================================
# 5. 視覺化這個「通用雜訊」長什麼樣子
# ==========================================
noise_np = universal_noise.detach().squeeze().cpu().permute(1, 2, 0).numpy()
# 將 [-epsilon, epsilon] 的雜訊映射到 [0, 1] 之間方便顯示
noise_vis = (noise_np / epsilon * 0.5) + 0.5 

plt.figure(figsize=(6, 6))
plt.imshow(noise_vis)
plt.title(f"Universal Adversarial Perturbation\n(Epsilon: {epsilon:.3f})")
plt.axis("off")
plt.tight_layout()

plt.savefig('universal_noise_visualization.png', dpi=300)
print("通用雜訊視覺化圖已儲存為 universal_noise_visualization.png")
