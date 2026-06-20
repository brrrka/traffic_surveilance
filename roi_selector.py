import cv2
import numpy as np

# Simpan titik ROI
roi_points = []

# Callback mouse
def draw_roi(event, x, y, flags, param):
    global roi_points

    if event == cv2.EVENT_LBUTTONDOWN:

        roi_points.append((x, y))

        print(f"Titik: ({x}, {y})")


# ======================
# Main Program
video_path = "data/input/video_cut.mp4"

cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

# Cek apakah video berhasil dibuka
if not ret:
    print("Video gagal dibuka")
    exit()

cv2.namedWindow("ROI Selector")
cv2.setMouseCallback("ROI Selector", draw_roi)

# Loop utama
while True:

    temp = frame.copy()

    # Gambar titik
    for point in roi_points:

        cv2.circle(temp, point, 5, (0,255,0), -1)

    # Hubungkan titik
    if len(roi_points) > 1:

        cv2.polylines(
            temp,
            [np.array(roi_points)],
            False,
            (255,0,0),
            2
        )

    cv2.imshow("ROI Selector", temp)

    key = cv2.waitKey(1)

    # Tekan ENTER untuk selesai
    if key == 13:

        break

    # Tekan R untuk reset
    elif key == ord('r'):

        roi_points = []

    # Tekan ESC untuk keluar
    elif key == 27:

        break

cap.release()
cv2.destroyAllWindows()

print("\nROI Points:")

print(roi_points)