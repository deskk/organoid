'''
only print image quality
'''
import cv2
import time

def measure_focus(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()
cv2.namedWindow("Focus Monitor")
last_print_time = time.time()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame.")
        break
    cv2.imshow("Focus Monitor", frame)
    current_time = time.time()
    if current_time - last_print_time >= 2:
        focus_score = measure_focus(frame)
        print(f"Focus Score (Laplacian Variance): {focus_score}")
        last_print_time = current_time
    k = cv2.waitKey(1)
    if k % 256 == 27:
        print("Escape hit, closing...")
        break
cam.release()
cv2.destroyAllWindows()
