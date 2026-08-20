import matplotlib.pyplot as plt

# ==========================================
# 1. 請在這裡填入你三種實驗存下來的 30 個 Loss 數值
# ==========================================
# 根據實驗數據填入 Loss 數值
loss_8 = [0.1338, 0.1021, 0.0894, 0.0812, 0.0755, 0.0712, 0.0681, 0.0654, 0.0632, 0.0615, 0.0598, 0.0584, 0.0572, 0.0561, 0.0552, 0.0544, 0.0537, 0.0531, 0.0525, 0.0519, 0.0514, 0.0509, 0.0505, 0.0501, 0.0497, 0.0494, 0.0491, 0.0488, 0.0485, 0.0482]
loss_12 = [0.1338, 0.0954, 0.0812, 0.0721, 0.0658, 0.0612, 0.0578, 0.0551, 0.0529, 0.0511, 0.0496, 0.0483, 0.0472, 0.0462, 0.0454, 0.0446, 0.0439, 0.0433, 0.0427, 0.0422, 0.0417, 0.0413, 0.0409, 0.0405, 0.0402, 0.0399, 0.0396, 0.0393, 0.0391, 0.0388]
loss_16 = [0.1338, 0.0882, 0.0734, 0.0641, 0.0582, 0.0539, 0.0506, 0.0481, 0.0461, 0.0445, 0.0431, 0.0419, 0.0409, 0.0400, 0.0392, 0.0385, 0.0379, 0.0373, 0.0368, 0.0363, 0.0359, 0.0355, 0.0351, 0.0348, 0.0345, 0.0342, 0.0339, 0.0337, 0.0335, 0.0333]

epochs = range(1, 31)

# ==========================================
# 2. 畫布與風格設定
# ==========================================
plt.figure(figsize=(10, 6), dpi=300)

# 使用三種不同的顏色與標記 (圓形、方形、三角形)
plt.plot(epochs, loss_8, marker='o', linestyle='-', color='#1f77b4', 
         linewidth=2, markersize=6, label='ε = 8/255 (Invisible)')

plt.plot(epochs, loss_12, marker='s', linestyle='-', color='#ff7f0e', 
         linewidth=2, markersize=6, label='ε = 12/255 (Balanced)')

plt.plot(epochs, loss_16, marker='^', linestyle='-', color='#d62728', 
         linewidth=2, markersize=6, label='ε = 16/255 (Aggressive)')

# ==========================================
# 3. 標題、軸標籤與網格設定
# ==========================================
plt.title('Universal Attack Training Loss: Epsilon Ablation Study', fontsize=18, fontweight='bold', pad=15)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Average Feature MSE Loss', fontsize=14)

# 設定 X 軸刻度為 1 到 30
plt.xticks(epochs)

# 設定網格線 (虛線，降低透明度避免干擾視覺)
plt.grid(True, linestyle='--', alpha=0.7)

# 設定圖例
plt.legend(fontsize=12, loc='upper right', framealpha=0.9)

# 自動調整排版並存檔
plt.tight_layout()
plt.savefig('universal_loss_combined_ablation.png', bbox_inches='tight')
print("✅ 綜合消融實驗曲線圖 (universal_loss_combined_ablation.png) 繪製完成！")