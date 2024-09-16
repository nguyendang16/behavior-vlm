import torch
import clip
import cv2
from PIL import Image

# Load pretrain model CLIP và tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

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

# Tokenize behavior description
text_inputs = clip.tokenize(descriptions).to(device)

def detect_behavior(frame):
    # Preprocess frame
    image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
    
    # predict behavior using clip
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text_inputs)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # take the highest behavior's accuracy
    predicted_behavior = descriptions[probs.argmax()]
    return predicted_behavior

cap = cv2.VideoCapture(0)  

frame_skip = 5  # frame skipped to process 
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1) 
    resized_frame = cv2.resize(frame, (224, 224))
    
    if frame_count % frame_skip == 0:
        behavior = detect_behavior(resized_frame)
    
    frame_count += 1
    
    cv2.putText(frame, f"Behavior: {behavior}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Real-time Behavior Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
