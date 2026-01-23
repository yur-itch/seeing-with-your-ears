import cv2

video_path = "samples/videos/forest.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Frames:", frame_count)
print("FPS:", fps)

cap.release()
