import torch
import clip
import cv2
import mediapipe as mp
from PIL import Image
import yaml

# Load the CLIP model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Read config.yaml to get the behavior descriptions
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
descriptions = config['descriptions']

# Tokenize the behavior descriptions
text_inputs = clip.tokenize(descriptions).to(device)

# Initialize Mediapipe for skeleton tracking
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to detect behavior
def detect_behavior(frame):
    # Preprocess frame
    image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
    
    # Predict behavior using CLIP
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text_inputs)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # Get the behavior with the highest probability
    predicted_behavior = descriptions[probs.argmax()]
    return predicted_behavior

# Use OpenCV to open video
cap = cv2.VideoCapture(0)  # Use webcam or video path

frame_skip = 5  # Number of frames to skip between each processing
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the image to fix the mirror effect (inverted)
    frame = cv2.flip(frame, 1)  # Flip vertically
    
    # Resize frame to speed up processing
    resized_frame = cv2.resize(frame, (224, 224))
    
    # Mediapipe processes the skeleton
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    # Draw skeleton on the frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Only process one frame every frame_skip frames to reduce load
    if frame_count % frame_skip == 0:
        # Detect behavior
        behavior = detect_behavior(resized_frame)
    
    frame_count += 1
    
    # Display behavior on the video
    cv2.putText(frame, f"Behavior: {behavior}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Show video with skeleton and behavior description
    cv2.imshow('Real-time Behavior Detection with Skeleton', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
