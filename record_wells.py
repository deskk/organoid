import cv2
import os
import time
import argparse
import requests

MOONRAKER_HOST = ''
MOONRAKER_PORT = ''

save_folder = "organoid_growth"
os.makedirs(save_folder, exist_ok=True)

def get_position():
    """
    Retrieves the current position of the printer's end-effector.
    Returns:
        list: The current [X, Y, Z, E] positions.
    """
    url = f'http://{MOONRAKER_HOST}:{MOONRAKER_PORT}/printer/objects/query?motion_report'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        live_position = data['result']['status']['motion_report']['live_position']
        return live_position  # [X, Y, Z, E]
    except Exception as e:
        print('Error getting position:', e)
        return None

def go_to(position, description='', blocking=True):
    """
    Commands the printer to move to a specified position.
    Args:
        position (list or tuple): The target [X, Y, Z, E] positions.
        description (str): Description of the movement for logging purposes.
        blocking (bool): If True, the function blocks until the printer reaches the target position.
    Returns:
        bool: True if the command was successful, False otherwise.
    """
    x, y, z, e = position
    gcode_cmd = 'G1'
    if x is not None:
        gcode_cmd += f' X{x}'
    if y is not None:
        gcode_cmd += f' Y{y}'
    if z is not None:
        gcode_cmd += f' Z{z}'
    if e is not None:
        gcode_cmd += f' E{e}'
    gcode_cmd += ' F3000'
    url = f'http://{MOONRAKER_HOST}:{MOONRAKER_PORT}/printer/gcode/script'
    headers = {'Content-Type': 'application/json'}
    payload = {'script': gcode_cmd}
    try:
        if description:
            print(f'Action: {description}')
        else:
            print(f'Moving to position: X={x}, Y={y}, Z={z}, E={e}')
        print(f'Sending G-code: {gcode_cmd}')
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print('Error sending G-code command:', e)
        return False

    if blocking:
        tolerance = 0.1
        while True:
            current_position = get_position()
            if current_position is None:
                time.sleep(0.1)
                continue
            delta = [abs(cp - tp) for cp, tp in zip(current_position[:3], [x, y, z]) if tp is not None]
            if all(d <= tolerance for d in delta):
                break
            time.sleep(0.1)
    return True

def generate_well_plate_coordinates():
    """
    Generates a dictionary mapping well positions to their XY coordinates based on the provided corner coordinates.
    Returns:
        dict: A dictionary with well labels as keys and (X, Y) tuples as values.
    """
    coordinate_matrix = {}
    X_start = 48    # X coordinate of A1
    X_end = 148     # X coordinate of A12
    Y_start = 73.5  # Y coordinate of A1
    Y_end = 9.5     # Y coordinate of H1
    num_columns = 12
    num_rows = 8

    x_increment = (X_end - X_start) / (num_columns - 1)
    y_increment = (Y_start - Y_end) / (num_rows - 1)

    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    for row_index, row in enumerate(rows):
        for col in range(1, num_columns + 1):
            x = X_start + (col - 1) * x_increment
            y = Y_start - row_index * y_increment  # Decreasing Y coordinate
            well_label = f"{row}{col}"
            coordinate_matrix[well_label] = (round(x, 2), round(y, 2))

    return coordinate_matrix

def record_video(cam, well_label, duration=10):
    """
    Records video for a specified duration and saves it with a specific filename.
    Args:
        cam: The OpenCV VideoCapture object.
        well_label (str): The label of the well being recorded.
        duration (int): Duration of the video in seconds.
    """
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cam.get(cv2.CAP_PROP_FPS)) or 30
    timestamp = time.strftime("%y%m%d")
    video_filename = os.path.join(save_folder, f"{timestamp}_{well_label}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
    print(f"Recording video for well {well_label}...")
    start_time = time.time()
    while time.time() - start_time < duration:
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Well Plate Video Recording Script')
    parser.add_argument('--wells', type=str, required=True, help='Comma-separated list of wells to record, e.g., A1,A2,B3')
    args = parser.parse_args()
    wells = args.wells.split(',')
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera.")
        exit()

    cv2.namedWindow("Video Recorder")
    if get_position() is not None:
        print('Successfully connected to Moonraker.')
        coordinates = generate_well_plate_coordinates()

        for well in wells:
            if well not in coordinates:
                print(f"Well {well} is not valid. Skipping.")
                continue

            x, y = coordinates[well]
            go_to([None, None, 40, None], description='Move to Z height')
            go_to([x, y, None, None], description=f'Move to well {well} (smooth XY movement)')
            go_to([None, None, 40, None], description='Lower to imaging Z height (Z=40)')
            print("Stabilizing...")
            time.sleep(2)
            record_video(cam, well_label=well, duration=10)
            go_to([None, None, 40, None], description='Raise to safe Z height')
        cam.release()
        cv2.destroyAllWindows()
    else:
        print('Failed to connect to Moonraker. Please check your host and port settings.')
