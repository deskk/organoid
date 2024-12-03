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

'''
control camera Z coordinate to ensure image quality using Laplacian variance
'''

# import cv2
# import serial
# import time

# # Function to measure focus using Laplacian variance
# def measure_focus(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#     return laplacian_var

# # Function to move the Z-axis
# def move_z(serial_connection, z_value, feedrate=100):
#     command = f"G1 Z{z_value} F{feedrate}\n"
#     serial_connection.write(command.encode())
#     time.sleep(1)  # Wait for the move to complete

# # Main focus adjustment logic
# def auto_focus(cam, serial_connection, z_start, z_end, z_step):
#     best_focus = 0
#     best_z = z_start
#     current_z = z_start

#     # Move to the start position
#     move_z(serial_connection, z_start)

#     while current_z <= z_end:
#         ret, frame = cam.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

#         focus_score = measure_focus(frame)
#         print(f"Z: {current_z}, Focus Score: {focus_score}")

#         if focus_score > best_focus:
#             best_focus = focus_score
#             best_z = current_z

#         # Move to the next Z position
#         current_z += z_step
#         move_z(serial_connection, current_z)

#     # Return to the best focus position
#     move_z(serial_connection, best_z)
#     print(f"Best focus at Z: {best_z}, Score: {best_focus}")

# # Initialize the camera and serial connection
# cam = cv2.VideoCapture(0)
# serial_connection = serial.Serial('/dev/ttyUSB0', 115200)  # Adjust port and baudrate

# # Wait for serial connection to initialize
# time.sleep(2)

# # Run auto-focus
# auto_focus(cam, serial_connection, z_start=0, z_end=10, z_step=0.5)

# # Release resources
# cam.release()
# serial_connection.close()
# cv2.destroyAllWindows()
