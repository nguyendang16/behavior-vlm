import torch
import clip
import cv2
from PIL import Image

# Load pretrain model CLIP và tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Danh sách mô tả hành vi
descriptions = [
    "a person sitting",
    "a person walking",
    "a person running",
    "a person picking up something",
    "a person jumping",
    "a person waving",
    "a person moving",
    "a person standing"
]

# Tokenize các mô tả hành vi
text_inputs = clip.tokenize(descriptions).to(device)

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
    
    frame = cv2.flip(frame, 1) 
    resized_frame = cv2.resize(frame, (224, 224))
    
    # Chỉ xử lý 1 khung hình sau mỗi frame_skip khung hình để giảm tải
    if frame_count % frame_skip == 0:
        # Nhận diện hành vi
        behavior = detect_behavior(resized_frame)
    
    frame_count += 1
    
    # Hiển thị hành vi lên video
    cv2.putText(frame, f"Behavior: {behavior}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Hiển thị video
    cv2.imshow('Real-time Behavior Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
