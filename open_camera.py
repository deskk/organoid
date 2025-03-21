'''
if this script doesnt work, run
1. find the process ID (PID) using /dev/video0:
fuser /dev/video0
--> it will output something like /dev/video0:          2237m
--> PID is 2237 in this case

2. stop the process
sudo kill <PID>

'''

import cv2
import os

save_folder = "pictures"
os.makedirs(save_folder, exist_ok=True)

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = os.path.join(save_folder, f"opencv_frame_{img_counter}.png")
        cv2.imwrite(img_name, frame)
        print(f"{img_name} written!")
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
