import torch
import clip
import cv2
import mediapipe as mp
from PIL import Image
import yaml

# Load mô hình CLIP và tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
descriptions = config['descriptions']

# Tokenize các mô tả hành vi
text_inputs = clip.tokenize(descriptions).to(device)

# Khởi tạo Mediapipe cho skeleton tracking
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Hàm nhận diện hành vi
def detect_behavior(frame):
    # Preprocess frame
    image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
    
    # Dự đoán hành vi sử dụng CLIP
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text_inputs)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # Lấy hành vi có xác suất cao nhất
    predicted_behavior = descriptions[probs.argmax()]
    return predicted_behavior

# Sử dụng OpenCV để mở video
cap = cv2.VideoCapture(0)  # Dùng webcam hoặc video path

frame_skip = 5  # Số lượng khung hình bỏ qua giữa mỗi lần xử lý
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Lật ảnh để sửa lỗi mirror (lật ngược)
    frame = cv2.flip(frame, 1)  # Flip theo trục dọc
    
    # Resize frame để tăng tốc độ xử lý
    resized_frame = cv2.resize(frame, (224, 224))
    
    # Mediapipe xử lý bộ xương
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    # Vẽ skeleton lên hình
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Chỉ xử lý 1 khung hình sau mỗi frame_skip khung hình để giảm tải
    if frame_count % frame_skip == 0:
        # Nhận diện hành vi
        behavior = detect_behavior(resized_frame)
    
    frame_count += 1
    
    # Hiển thị hành vi lên video
    cv2.putText(frame, f"Behavior: {behavior}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Hiển thị video với skeleton và mô tả hành vi
    cv2.imshow('Real-time Behavior Detection with Skeleton', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
