import cv2
from detector import VehicleDetector
from trajectory import TrajectoryManager
from visualizer import Visualizer
from accident_detector import AccidentDetector

VIDEO_PATH = "data/input/video.mp4"
OUTPUT_PATH = "data/output/result_2.mp4"
MODEL_PATH = "models/yolo11s.pt"

FPS = 30

detector = VehicleDetector(MODEL_PATH)
trajectory = TrajectoryManager(FPS)
visualizer = Visualizer()
acc_detector = AccidentDetector()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Video gagal dibuka!")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (width, height))

post_crash_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    tracked = detector.detect_and_track(frame)
    histories = trajectory.update(tracked)
    
    accidents = acc_detector.detect(histories)

    frame = visualizer.draw_all(frame, tracked, histories, accidents)

    out.write(frame)
    cv2.imshow("PoC Accident Detection", frame)
    
    if accidents:
        post_crash_counter += 1
        
    if post_crash_counter > 30:
        print("Tabrakan terdeteksi! Memotong video dan selesai.")
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"PoC berhasil disimpan di: {OUTPUT_PATH}")