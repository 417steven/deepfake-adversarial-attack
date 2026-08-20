import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 載入你的真實 Log 數據
# ==========================================
# 災難發散組 (Batch Size = 1, 沒有梯度累積)，只跑了 5 個 Epoch 就爛掉了
epochs_bad = np.array([1, 2, 3, 4, 5])
loss_bad = np.array([0.0508, 0.0541, 0.0594, 0.0680, 0.0684])

# 完美收斂組 (Accumulation Steps = 32, 980 Real)，取前 15 個 Epoch 作為對比
epochs_good = np.arange(1, 16)
loss_good = np.array([
    0.0868, 0.0541, 0.0464, 0.0424, 0.0404, 
    0.0389, 0.0378, 0.0369, 0.0362, 0.0355, 
    0.0348, 0.0346, 0.0341, 0.0335, 0.0336
])

# ==========================================
# 2. 設定深色高科技風格 (Cyber-Security Theme)
# ==========================================
# 背景色：深藍黑 (#0F172A) / 網格與文字：淺灰綠
plt.rcParams['axes.facecolor'] = '#0F172A'
plt.rcParams['figure.facecolor'] = '#0F172A'
plt.rcParams['text.color'] = '#E2E8F0'
plt.rcParams['axes.labelcolor'] = '#E2E8F0'
plt.rcParams['xtick.color'] = '#E2E8F0'
plt.rcParams['ytick.color'] = '#E2E8F0'

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# ==========================================
# 3. 繪製兩條動力學曲線
# ==========================================
# 壞曲線：使用警告的紅色，帶有發散感
ax.plot(epochs_bad, loss_bad, marker='X', linestyle='--', color='#FF3366', 
        linewidth=2.5, markersize=8, label='Naive Update (Batch=1) - Divergence')

# 好曲線：使用代表駭客科技的螢光綠，展現平滑收斂
ax.plot(epochs_good, loss_good, marker='o', linestyle='-', color='#39FF14', 
        linewidth=3, markersize=7, label='Gradient Accumulation (Steps=32) - Convergence')

# ==========================================
# 4. 加入專業的學術標註 (Annotations)
# ==========================================
# 標註發散原因
ax.annotate('Tug-of-War Effect\n(Extreme Gradient Conflict)',
            xy=(4, 0.0680), xytext=(5, 0.0800),
            arrowprops=dict(facecolor='#FF3366', shrink=0.05, width=2, headwidth=8),
            fontsize=12, color='#FF3366', fontweight='bold', ha='center')

# 標註收斂優勢
ax.annotate('Smooth Landing via\nAccumulated Consensus',
            xy=(12, 0.0346), xytext=(11, 0.0500),
            arrowprops=dict(facecolor='#39FF14', shrink=0.05, width=2, headwidth=8),
            fontsize=12, color='#39FF14', fontweight='bold', ha='center')

# ==========================================
# 5. 排版細節與輸出
# ==========================================
ax.set_title('Optimization Dynamics: Catastrophic Interference vs. Stable Convergence', 
             fontsize=16, fontweight='bold', pad=20, color='#39FF14')
ax.set_xlabel('Training Epochs', fontsize=14)
ax.set_ylabel('Average Feature MSE Loss', fontsize=14)

ax.set_xticks(np.arange(1, 16, 1))
ax.grid(True, linestyle=':', color='#334155', linewidth=1)

# 圖例設定，把外框設成透明以免擋到線
legend = ax.legend(fontsize=12, loc='upper right', frameon=True)
legend.get_frame().set_facecolor('#1E293B')
legend.get_frame().set_edgecolor('#39FF14')

plt.tight_layout()
plt.savefig('optimization_dynamics_dark.png', bbox_inches='tight')
print("✅ 深色高科技版「最佳化動力學圖」(optimization_dynamics_dark.png) 繪製完成！")