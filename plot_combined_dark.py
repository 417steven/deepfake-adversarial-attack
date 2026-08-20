import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 載入你的真實 Log 數據
# ==========================================
epochs = np.arange(1, 31)

# (程式碼內建了一組逼近你真實 Log 的平滑模擬數據，方便你直接看效果)
# 👉 實戰時，請務必把下面這三個 Array 換成你剛剛印出來的那 30 個真實 Loss 數值！
loss_8 = [0.1338, 0.1021, 0.0894, 0.0812, 0.0755, 0.0712, 0.0681, 0.0654, 0.0632, 0.0615, 0.0598, 0.0584, 0.0572, 0.0561, 0.0552, 0.0544, 0.0537, 0.0531, 0.0525, 0.0519, 0.0514, 0.0509, 0.0505, 0.0501, 0.0497, 0.0494, 0.0491, 0.0488, 0.0485, 0.0482]
loss_12 = [0.1338, 0.0954, 0.0812, 0.0721, 0.0658, 0.0612, 0.0578, 0.0551, 0.0529, 0.0511, 0.0496, 0.0483, 0.0472, 0.0462, 0.0454, 0.0446, 0.0439, 0.0433, 0.0427, 0.0422, 0.0417, 0.0413, 0.0409, 0.0405, 0.0402, 0.0399, 0.0396, 0.0393, 0.0391, 0.0388]
loss_16 = [0.1338, 0.0882, 0.0734, 0.0641, 0.0582, 0.0539, 0.0506, 0.0481, 0.0461, 0.0445, 0.0431, 0.0419, 0.0409, 0.0400, 0.0392, 0.0385, 0.0379, 0.0373, 0.0368, 0.0363, 0.0359, 0.0355, 0.0351, 0.0348, 0.0345, 0.0342, 0.0339, 0.0337, 0.0335, 0.0333]

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
# 3. 繪製三條消融實驗曲線 (高對比螢光色)
# ==========================================
# Epsilon = 8 (保守/絕對隱形)
ax.plot(epochs, loss_8, marker='o', linestyle='-', color='#00FFFF', 
        linewidth=2.5, markersize=6, label='ε = 8/255 (Invisible)')

# Epsilon = 12 (平衡)
ax.plot(epochs, loss_12, marker='s', linestyle='-', color='#FF9F1C', 
        linewidth=2.5, markersize=6, label='ε = 12/255 (Balanced)')

# Epsilon = 16 (最強火力)
ax.plot(epochs, loss_16, marker='^', linestyle='-', color='#39FF14', 
        linewidth=2.5, markersize=6, label='ε = 16/255 (Aggressive)')

# ==========================================
# 4. 排版細節與輸出
# ==========================================
ax.set_title('Universal Attack Loss: Epsilon Ablation Study', 
             fontsize=16, fontweight='bold', pad=20, color='#E2E8F0')
ax.set_xlabel('Training Epochs', fontsize=14)
ax.set_ylabel('Average Feature MSE Loss', fontsize=14)

# 設定 X 軸每 2 個 Epoch 顯示一個刻度，避免太擠
ax.set_xticks(np.arange(2, 32, 2))
ax.grid(True, linestyle=':', color='#334155', linewidth=1)

# 圖例設定 (深色透明框)
legend = ax.legend(fontsize=12, loc='upper right', frameon=True)
legend.get_frame().set_facecolor('#1E293B')
legend.get_frame().set_edgecolor('#334155')

plt.tight_layout()
plt.savefig('universal_loss_combined_dark.png', bbox_inches='tight')
print("✅ 深色高科技版「Epsilon 消融實驗曲線圖」(universal_loss_combined_dark.png) 繪製完成！")