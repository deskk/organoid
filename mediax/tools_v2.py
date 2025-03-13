'''
version 2
'''

import requests
import time
import cv2
import numpy as np
import os
import json
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ExifTags
import datetime
import sys

###############################################################################
#                         GLOBAL LOGGING UTILS
###############################################################################
LOGS = []

def log(message: str):
    """
    Appends a message to a global log list and also prints it to stdout.
    """
    print(message)
    LOGS.append(message)

def save_logs(log_dir: str, filename="session.log"):
    """
    Writes all logged messages to a text file in the specified directory.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, filename)
    with open(log_path, "w", encoding="utf-8") as f:
        for line in LOGS:
            f.write(line + "\n")
    print(f"[save_logs] Logs have been written to {log_path}")

###############################################################################
#                          LOAD CONFIG & SKIP WELLS
###############################################################################

def load_config(config_path):
    """
    Load configuration parameters from a JSON file.
    """
    with open(config_path, 'r') as f:
        return json.load(f)

def load_skip_wells(skip_wells_path):
    """
    Load well labels (e.g. 'A1', 'D7') to skip, from a JSON file.
    Returns a set of (row_index, col_index) pairs for quick lookup.
    """
    with open(skip_wells_path, 'r') as f:
        data = json.load(f)

    skip_wells_labels = data.get("skip_wells", [])
    return {well_label_to_indices(label) for label in skip_wells_labels}

def well_label_to_indices(label):
    """
    Convert a well label like 'A1', 'D7' into (row_index, col_index).
    Rows A..H => 0..7, columns 1..12 => 0..11.
    E.g., 'A1' => (0, 0), 'D7' => (3, 6).
    """
    row_letter = label[0].upper()  # e.g. 'A' -> 'H'
    col_part   = label[1:]        # e.g. '1' -> '12'

    row_index = ord(row_letter) - ord('A')  # 'A'=0, 'B'=1, ...
    col_index = int(col_part) - 1           # '1' -> 0, '12' -> 11
    return (row_index, col_index)

def row_col_to_well_label(row, col):
    """
    Inverse of well_label_to_indices: (row=3, col=6) => 'D7'
    row=0 => A, row=1 => B, etc. ; col=0 => 1, col=6 => 7, etc.
    """
    row_letter = chr(ord('A') + row)  # 0->A, 3->D
    col_number = col + 1             # 0->1, 6->7
    return f"{row_letter}{col_number}"

###############################################################################
#                          PRINTER UTILS
###############################################################################

def home(config):
    send_gcode(config, "G28")

def get_position(config):
    url = f'http://{config["MOONRAKER_HOST"]}:{config["MOONRAKER_PORT"]}/printer/objects/query?toolhead'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['result']['status']['toolhead']['position']  # [x, y, z, e]
    except Exception as e:
        log(f"Error getting position: {e}")
        return None

def send_gcode(config, gcode_cmd:str):
    url = f'http://{config["MOONRAKER_HOST"]}:{config["MOONRAKER_PORT"]}/printer/gcode/script'
    payload = {'script': gcode_cmd}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except Exception as e:
        log(f"Error sending G-code command '{gcode_cmd}': {e}")
        return False
    return True

def move_to(config, position, speed=None, description=''):
    """
    position: (x, y, z). If any is None, fill it with the current coordinate.
    """
    if speed is None:
        speed = config["MOVE_SPEED"]

    current = get_position(config)
    if current is None:
        log("Failed to get current position from printer.")
        return

    cx, cy, cz = current[:3]
    x, y, z = position

    # fill in None with current coords
    if x is None: x = cx
    if y is None: y = cy
    if z is None: z = cz

    # always absolute coords
    send_gcode(config, 'G90')

    # issue the move
    gcode_cmd = f"G1 X{x} Y{y} Z{z} F{speed}"
    if description:
        log(f'Action: {description} => {gcode_cmd}')
    else:
        log(f'Move => {gcode_cmd}')

    send_gcode(config, gcode_cmd)
    send_gcode(config, 'M400')  # Force flush motion

def control_syringe(config, volume_uL:float):
    """
    Negative => aspirate,
    Positive => dispense.
    """
    send_gcode(config, 'M83')  # Relative extrusion
    e_movement = volume_uL * config["SYRINGE_E_PER_UL"]
    move_cmd = f'G1 E{e_movement} F3000'
    send_gcode(config, move_cmd)
    send_gcode(config, 'M400')

###############################################################################
#                          CAMERA/IMAGE UTILS
###############################################################################

def auto_orient_image(image_path):
    '''
    Opens image with PIL, checks EXIF orientation, rotates accordingly,
    converts to OpenCV BGR format.
    '''
    pil_image = Image.open(image_path)
    try:
        exif = dict(pil_image._getexif().items()) if pil_image._getexif() else {}
        for orientation_tag in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation_tag] == 'Orientation':
                break
        orientation_value = exif.get(orientation_tag, None)

        if orientation_value == 3:
            pil_image = pil_image.rotate(180, expand=True)
        elif orientation_value == 6:
            pil_image = pil_image.rotate(270, expand=True)
        elif orientation_value == 8:
            pil_image = pil_image.rotate(90, expand=True)
    except Exception:
        pass

    bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    log("[auto_orient_image] converted PIL(RGB) -> OpenCV(BGR).")
    return bgr_image

def preprocess_image_for_inference(image_path):
    '''
    Resize image to 640x640 and convert to RGB format for YOLO input.
    '''
    bgr_image = auto_orient_image(image_path)
    bgr_resized = cv2.resize(bgr_image, (640, 640), interpolation=cv2.INTER_AREA)
    rgb_image = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)
    return rgb_image

###############################################################################
#                           ORGANOID DETECTOR
###############################################################################

class OrganoidDetector:
    def __init__(self, model_path, confidence_threshold):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, image_path, save_annotated=False, output_dir=None):
        '''
        Runs YOLO detection on image_path (640x640, RGB).
        Optionally saves an annotated version of the image in output_dir.
        '''
        image_for_inference = preprocess_image_for_inference(image_path)
        results = self.model.predict(
            source=image_for_inference,
            imgsz=640,
            conf=self.confidence_threshold,
            save=False
        )

        if save_annotated and len(results) > 0:
            annotated_rgb = results[0].plot()  # YOLO returns an RGB array
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            out_filename = f"{Path(image_path).stem}_bbox.png"
            out_filepath = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_filepath, annotated_bgr)
            log(f"[detect] Saved annotated image to {out_filepath} in BGR.")

        if len(results) > 0:
            boxes = results[0].boxes
            if boxes and len(boxes.data) > 0:
                conf = boxes.conf[0].item()
                return True, conf
        return False, None

###############################################################################
#                         DIRECTORY HELPER
###############################################################################

def make_new_session_directory(base_dir):
    """
    Creates a new timestamped session subdirectory in base_dir
    """
    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(base_dir, f"session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    log(f"[make_new_session_directory] Created new session directory: {session_dir}") 
    return session_dir

###############################################################################
#                 MAIN MEDIA EXCHANGE FUNCTION
###############################################################################

def run_media_exchange(config, skip_well_set, final_save_dir, camera_id, yolo_path):
    """
    Main logic for performing cell media exchange on a 96-well plate.

    :param config: Dictionary loaded from config.json
    :param skip_well_set: Set of (row_index, col_index) wells to skip
    :param final_save_dir: Output directory for images/logs
    :param camera_id: Which camera device ID to use (e.g. 0)
    :param yolo_path: Path to YOLO model weights
    """
    FRESH_MEDIA_POS = config["FRESH_MEDIA_POS"]
    WATER_POS       = config["WATER_POS"]
    WASTE_MEDIA_POS = config["WASTE_MEDIA_POS"]
    CAMERA_OFFSET   = config["CAMERA_OFFSET"]

    SAFE_Z  = config["SAFE_Z"]
    EXTRACT_Z = config["EXTRACT_Z"]
    EXTRACT_Z_DISH = config["EXTRACT_Z_DISH"]
    DISPENSE_OFFSET = config["DISPENSE_OFFSET"]
    EVAPORATION_OFFSET = config["EVAPORATION_OFFSET"]
    EXCHANGE_MEDIA_VOLUME = config["EXCHANGE_MEDIA_VOLUME"]
    CAMERA_Z = config["CAMERA_Z"]
    MIXING_WAIT_TIME = config["MIXING_WAIT_TIME"]
    CAMERA_STABILIZE_TIME = config["CAMERA_STABILIZE_TIME"]
    CONFIDENCE_THRESHOLD = config["CONFIDENCE_THRESHOLD"]

    # extra offset in the well to adjust for evaporation
    evaporation = EVAPORATION_OFFSET

    original_image_folder = os.path.join(final_save_dir, "original_images")
    detections_dir = os.path.join(final_save_dir, "detections")
    os.makedirs(original_image_folder, exist_ok=True)
    os.makedirs(detections_dir, exist_ok=True)

    # 1) home the printer
    home(config)

    # 2) initialize camera and organoid detector:
    cap = cv2.VideoCapture(camera_id)
    assert cap.isOpened(), "[run_media_exchange] camera failed to open"
    detector = OrganoidDetector(
        model_path=yolo_path,
        confidence_threshold=CONFIDENCE_THRESHOLD)

    # 3) generate well locations (8 rows x 12 cols)
    well_locations = []
    for row in range(8):
        for col in range(12):
            if (row, col) in skip_well_set:
                well_label = row_col_to_well_label(row, col)
                log(f"Skipping well: {well_label} (row={row}, col={col})")
                continue

            x = config["SYRINGE_ORIGIN_X"] + col * config["WELL_SPACING_X"]
            y = config["SYRINGE_ORIGIN_Y"] + row * config["WELL_SPACING_Y"]
            well_locations.append((row, col, x, y))

    # 4) perform media exchange on each well
    for (row, col, x, y) in well_locations:
        well_label = row_col_to_well_label(row, col)
        log(f"Processing well: {well_label} (row={row}, col={col})")

        # pull water for mixing offset
        move_to(config, [WATER_POS[0], WATER_POS[1], SAFE_Z])
        move_to(config, [WATER_POS[0], WATER_POS[1], EXTRACT_Z_DISH])
        control_syringe(config, -evaporation)  # aspirate from water
        move_to(config, [WATER_POS[0], WATER_POS[1], SAFE_Z])

        # go to the well
        move_to(config, [x, y, SAFE_Z])
        move_to(config, [x, y, EXTRACT_Z])
        control_syringe(config, evaporation + DISPENSE_OFFSET)  # dispense mixing volume
        move_to(config, [x, y, SAFE_Z])
        control_syringe(config, -DISPENSE_OFFSET)
        time.sleep(MIXING_WAIT_TIME)

        found = False
        attempts = 0
        while True:
            # aspirate old media
            move_to(config, [x, y, EXTRACT_Z])
            control_syringe(config, -EXCHANGE_MEDIA_VOLUME)
            move_to(config, [x, y, CAMERA_Z])

            # move camera over the well
            move_to(config, [x + CAMERA_OFFSET[0], y + CAMERA_OFFSET[1], CAMERA_Z])
            log("[run_media_exchange] Waiting for camera to stabilize...")
            time.sleep(CAMERA_STABILIZE_TIME)

            # flush camera buffer
            for _ in range(4):
                cap.grab()
                time.sleep(0.1)
            ret, well_image_bgr = cap.retrieve()
            if not ret:
                raise RuntimeError("[run_media_exchange] error retrieving camera frame")

            # use well label in filename (eg. 'A1_attempt0.png')
            image_filename = f"{well_label}_attempt{attempts}.png"
            image_path = os.path.join(original_image_folder, image_filename)
            cv2.imwrite(image_path, well_image_bgr)
            log(f"[run_media_exchange] saved raw image: {image_path}")

            found, conf = detector.detect(
                image_path=image_path,
                save_annotated=True,
                output_dir=detections_dir
            )

            if found or attempts == 5:
                log(f"[run_media_exchange] detection result for {well_label}: found={found}, conf={conf}")
                break
            else:
                attempts += 1
                log(f"[run_media_exchange] No object found on attempt {attempts-1}. Retrying capture...")
                # return the media to well before retry
                move_to(config, [x, y, SAFE_Z])
                move_to(config, [x, y, EXTRACT_Z])
                control_syringe(config, EXCHANGE_MEDIA_VOLUME + DISPENSE_OFFSET)
                move_to(config, [x, y, SAFE_Z])
                control_syringe(config, -DISPENSE_OFFSET)
                time.sleep(MIXING_WAIT_TIME)

        # dispose old media in the waste container
        move_to(config, [WASTE_MEDIA_POS[0], WASTE_MEDIA_POS[1], SAFE_Z])
        move_to(config, [WASTE_MEDIA_POS[0], WASTE_MEDIA_POS[1], EXTRACT_Z_DISH])
        control_syringe(config, EXCHANGE_MEDIA_VOLUME + DISPENSE_OFFSET)
        move_to(config, [WASTE_MEDIA_POS[0], WASTE_MEDIA_POS[1], SAFE_Z])
        control_syringe(config, -DISPENSE_OFFSET)

        # get fresh media
        move_to(config, [FRESH_MEDIA_POS[0], FRESH_MEDIA_POS[1], SAFE_Z])
        move_to(config, [FRESH_MEDIA_POS[0], FRESH_MEDIA_POS[1], EXTRACT_Z_DISH])
        control_syringe(config, -EXCHANGE_MEDIA_VOLUME)
        move_to(config, [FRESH_MEDIA_POS[0], FRESH_MEDIA_POS[1], SAFE_Z])

        # dispense fresh media into well
        move_to(config, [x, y, SAFE_Z])
        move_to(config, [x, y, EXTRACT_Z])
        control_syringe(config, EXCHANGE_MEDIA_VOLUME + DISPENSE_OFFSET)
        move_to(config, [x, y, SAFE_Z])
        control_syringe(config, -DISPENSE_OFFSET)

    log('Moving to final position...')
    move_to(config, [15, 15, 25])
    log('Completed run_media_exchange')

    cap.release()

###############################################################################
#                                MAIN
###############################################################################

def main():
    config_path = "mediax/config.json"
    skip_wells_path = "mediax/skip_wells.json"
    base_output_dir = "mediax_output"
    camera_id = 0
    yolo_path = "/home/pi/Documents/organoid/best.pt"
    config = load_config(config_path)
    skip_well_set = load_skip_wells(skip_wells_path)
    final_save_dir = make_new_session_directory(base_output_dir)
    try:
        run_media_exchange(
            config=config,
            skip_well_set=skip_well_set,
            final_save_dir=final_save_dir,
            camera_id=camera_id,
            yolo_path=yolo_path
        )
    except KeyboardInterrupt:
        log("User triggered Ctrl+C. Aborting run.")
    except Exception as e:
        log(f"Caught unexpected error: {e}")
    finally:
        save_logs(final_save_dir, "session.log")

if __name__ == "__main__":
    main()


