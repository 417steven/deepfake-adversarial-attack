import os
import cv2
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

# ==========================================
# 參數設定區 (可自行調整)
# ==========================================
INPUT_DIR = "./ffpp_videos"       # 包含 real 和 fake 影片的根目錄
OUTPUT_DIR = "./dataset_faces"    # 裁切後人臉圖片的儲存目錄
FRAMES_PER_VIDEO = 5              # 每部影片要抽取幾張臉 (建議 5-10 張即可)
IMAGE_SIZE = 224                  # 輸出圖片大小 (DINOv2 標準輸入尺寸)
MARGIN = 20                       # 人臉裁切邊緣的保留像素 (避免切得太緊)

# ==========================================
# 初始化設定
# ==========================================

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"正在使用運算裝置: {device}")

# 初始化 MTCNN 人臉偵測器
mtcnn = MTCNN(
    image_size=IMAGE_SIZE, 
    margin=MARGIN, 
    keep_all=False, # 每張圖只抓取最大的一張臉
    device=device
)

def process_video(video_path, save_dir, video_name):
    """
    處理單部影片：均勻抽幀 -> 抓取人臉 -> 儲存圖片
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法讀取影片: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return

    # 計算均勻抽幀的間隔
    interval = max(1, total_frames // FRAMES_PER_VIDEO)
    
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 只擷取特定間隔的幀
        if frame_count % interval == 0 and saved_count < FRAMES_PER_VIDEO:
            # OpenCV 預設讀取為 BGR，需轉換為 RGB 給 PIL/MTCNN 使用
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            
            # 定義這張人臉的儲存檔名 (例如: video01_003.jpg)
            save_path = os.path.join(save_dir, f"{video_name}_{saved_count:03d}.jpg")
            
            # MTCNN 魔法：偵測人臉、裁切、縮放並直接存檔
            # 若沒偵測到人臉，會回傳 None
            face = mtcnn(img_pil, save_path=save_path)
            
            if face is not None:
                saved_count += 1
                
        frame_count += 1
        if saved_count >= FRAMES_PER_VIDEO:
            break
            
    cap.release()

def main():
    # 建立輸出目錄結構
    for category in ['real', 'fake']:
        os.makedirs(os.path.join(OUTPUT_DIR, category), exist_ok=True)
        
        input_category_dir = os.path.join(INPUT_DIR, category)
        output_category_dir = os.path.join(OUTPUT_DIR, category)
        
        if not os.path.exists(input_category_dir):
            print(f"警告：找不到目錄 {input_category_dir}")
            continue
            
        # 取得所有 mp4 影片檔案
        video_files = [f for f in os.listdir(input_category_dir) if f.endswith('.mp4')]
        
        print(f"\n開始處理 {category} 類別，共 {len(video_files)} 部影片...")
        
        # 使用 tqdm 顯示進度條
        for video_file in tqdm(video_files):
            video_path = os.path.join(input_category_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            process_video(video_path, output_category_dir, video_name)
            
    print("\n✅ 資料集萃取完成！")
    print(f"請前往 {OUTPUT_DIR} 檢查你們的彈藥庫。")

if __name__ == "__main__":
    main()