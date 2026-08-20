import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 透過白底圖表進行「反向工程」還原數據
# ==========================================
np.random.seed(42) # 固定隨機種子

# 還原攻擊前 (Mean=0.123, Std=0.025, N=1000)
loss_before = np.random.normal(loc=0.123, scale=0.025, size=1000)

# 還原攻擊後 (Mean=0.050, Std=0.008, N=1000)
loss_after = np.random.normal(loc=0.050, scale=0.008, size=1000)

# ==========================================
# 2. 設定深色高科技風格 (Cyber-Security Theme)
# ==========================================
plt.rcParams['axes.facecolor'] = '#0F172A'
plt.rcParams['figure.facecolor'] = '#0F172A'
plt.rcParams['text.color'] = '#E2E8F0'
plt.rcParams['axes.labelcolor'] = '#E2E8F0'
plt.rcParams['xtick.color'] = '#E2E8F0'
plt.rcParams['ytick.color'] = '#E2E8F0'

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# ==========================================
# 3. 繪製重疊直方圖 (🎯關鍵修正：完美對齊原圖的 Bins 數量)
# ==========================================
# 藍色攻擊前：切分為 14 條
ax.hist(loss_before, bins=14, alpha=0.6, color='#00FFFF', edgecolor='#0F172A', 
        linewidth=1.2, label='Before Attack (Original Fake)')

# 綠色攻擊後：切分為 12 條 (這樣最高的那根柱子就會剛好頂到 200 左右！)
ax.hist(loss_after, bins=12, alpha=0.8, color='#39FF14', edgecolor='#0F172A', 
        linewidth=1.2, label='After Attack (Adversarial Fake)')

# ==========================================
# 4. 加入精確的平均值輔助線
# ==========================================
ax.axvline(0.123, color='#00FFFF', linestyle='dashed', linewidth=2.5, 
           label='Mean Before (0.123)')
ax.axvline(0.050, color='#39FF14', linestyle='dashed', linewidth=2.5, 
           label='Mean After (0.050)')

# ==========================================
# 5. 排版細節與輸出
# ==========================================
ax.set_title('Distribution of Feature MSE Loss Before & After Attack', 
             fontsize=16, fontweight='bold', pad=20, color='#E2E8F0')
ax.set_xlabel('Feature MSE Loss (Distance to Real Centroid)', fontsize=14)
ax.set_ylabel('Frequency (Number of Images)', fontsize=14)

ax.grid(True, linestyle=':', color='#334155', linewidth=1, axis='y')

# 圖例設定
legend = ax.legend(fontsize=12, loc='upper right', frameon=True)
legend.get_frame().set_facecolor('#1E293B')
legend.get_frame().set_edgecolor('#334155')

plt.tight_layout()
plt.savefig('batch_attack_histogram_dark_final.png', bbox_inches='tight')
print("✅ 最終校準完成！Y 軸高達 200 的深色直方圖 (batch_attack_histogram_dark_final.png) 繪製完畢！")