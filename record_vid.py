import cv2
import os
import time

save_folder = "recordings"
os.makedirs(save_folder, exist_ok=True)
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

cv2.namedWindow("Video Recorder")
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cam.get(cv2.CAP_PROP_FPS)) or 30
while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame.")
        break
    cv2.imshow("Video Recorder", frame)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        print("Recording started...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(save_folder, f"recording_{timestamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
        start_time = time.time()
        while time.time() - start_time < 5:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame during recording.")
                break
            out.write(frame)
            cv2.imshow("Video Recorder", frame)
            if cv2.waitKey(1) % 256 == 27:
                print("Escape hit, stopping recording...")
                break
        out.release()
        print(f"Video saved as {video_filename}")

cam.release()
cv2.destroyAllWindows()
