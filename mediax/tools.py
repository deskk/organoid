'''
version 1
'''

import requests
import time
import csv
import cv2
import numpy as np
import os
import argparse
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ExifTags
import datetime

MOONRAKER_HOST = '172.26.8.71'
MOONRAKER_PORT = 7125
MOVE_SPEED = 3000
SYRINGE_E_PER_UL = -9

###########################################################################################################
# on first well (A1), need to update/check before running the script
SYRINGE_ORIGIN_X = 44.5
SYRINGE_ORIGIN_Y = 201

FRESH_MEDIA_POS = (44.5, 105)
WATER_POS       = (100.5, 105)
WASTE_MEDIA_POS = (156.5, 105)
##############################################################################################################

CAMERA_OFFSET = (5, -137.6)

WELL_SPACING_X = 9.04
WELL_SPACING_Y = -9.0

SAFE_Z = 35
EXTRACT_Z = 1.5
EXTRACT_Z_DISH = -0.1

DISPENSE_OFFSET = 10  # Extra µL to dispense, then pull back
EVAPORATION_OFFSET = 20
EXCHANGE_MEDIA_VOLUME = 50

CAMERA_Z = 41

MIXING_WAIT_TIME = 5
CAMERA_STABILIZE_TIME = 3

def home():
    send_gcode("G28")

def get_position():
    url = f'http://{MOONRAKER_HOST}:{MOONRAKER_PORT}/printer/objects/query?toolhead'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['result']['status']['toolhead']['position']  # [x, y, z, e]
    except Exception as e:
        print('Error getting position:', e)
        return None


def send_gcode(gcode_cmd:str):
    url = f'http://{MOONRAKER_HOST}:{MOONRAKER_PORT}/printer/gcode/script'
    payload = {'script': gcode_cmd}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except Exception as e:
        print('Error sending G-code command:', e)
        return False
    return True


def move_to(position, speed=MOVE_SPEED, description=''):
    """
    - position: (x, y, z).  If any is None, fill it with the current coordinate.
    - Then do a G1 with explicit X, Y, Z.
    - Use M400 to ensure all moves are flushed.
    - blocking=True => poll until within tolerance of final position.
    """
    # TODO: Add assertion check to make sure we don't ever send unsafe commands
    current = get_position()
    if current is None:
        print("Failed to get current position from printer.")
        return

    cx, cy, cz = current[:3]
    x, y, z = position

    # Fill in None with current coords
    if x is None: x = cx
    if y is None: y = cy
    if z is None: z = cz

    # Always absolute coords
    send_gcode('G90')

    # Issue the move
    gcode_cmd = f"G1 X{x} Y{y} Z{z} F{speed}"
    if description:
        print(f'Action: {description} => {gcode_cmd}')
    else:
        print(f'Move => {gcode_cmd}')

    send_gcode(gcode_cmd)

    # Force flush motion
    send_gcode('M400')

    # if blocking:
    #     tolerance = 0.2
    #     target = (x, y, z)
    #     while True:
    #         current_pos = get_position()
    #         if not current_pos:
    #             time.sleep(0.1)
    #             continue
    #         cx, cy, cz = current_pos[:3]
    #         diffs = [abs(cx - x), abs(cy - y), abs(cz - z)]
    #         if all(d <= tolerance for d in diffs):
    #             break
    #         time.sleep(0.2)

def control_syringe(volume_uL:float):
    """
    Negative => aspirate,
    Positive => dispense.
    """
    send_gcode('M83')  # Relative extrusion
    e_movement = volume_uL * SYRINGE_E_PER_UL
    move_cmd = f'G1 E{e_movement} F3000'
    send_gcode(move_cmd)
    # optional flush
    send_gcode('M400')

def auto_orient_image(image_path):
    '''
    opens image with PIL
    checks EXIF orientation, rotates accordingly
    converts to openCV BGER format
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
    print("[auto_orient_image] converted PI(rgb) -> OpenCV(BGR).")
    return bgr_image

def preprocess_image_for_inference(image_path):
    '''
    resize image to 640x640
    converts to RGB format for yolo input
    '''
    bgr_image = auto_orient_image(image_path)
    bgr_resized = cv2.resize(bgr_image, (640, 640), interpolation=cv2.INTER_AREA)
    rgb_image = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)
    return rgb_image

class OrganoidDetector:
    def __init__(self, model_path, confidence_threshold=0.45): # level of detection confidence threshold the user configures
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, image_path, save_annotated=False, output_dir=None):
        '''
        runs yolo detection on image_path (640x640, RGB)
        optionally saves an annotated version of the image in output_dir
        '''
        image_for_inference = preprocess_image_for_inference(image_path)
        results = self.model.predict(
            source=image_for_inference,
            imgsz=640,
            conf=self.confidence_threshold,
            save=False
        )

        if save_annotated and len(results) > 0:
            # https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results.plot
            annotated_rgb = results[0].plot()  # this will give rbg frmat
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            out_filename = f"{Path(image_path).stem}_bbox.png"
            out_filepath = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_filepath, annotated_bgr)
            print(f"[detect] Saved annotated image to {out_filepath} in BGR.")

        if len(results) > 0:
            boxes = results[0].boxes
            if boxes and len(boxes.data) > 0:
                conf = boxes.conf[0].item()
                return True, conf
        return False, None

def make_or_get_directory(main_dir):
    """
    Creates a directory if it doesn't exist
    If it already exists, creates a timestamped subdirectory under it
    Returns the final directory path that was created or will be used
    """
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
        print(f"[make_or_get_directory] Created main directory: {main_dir}")
        return main_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sub_dir = os.path.join(main_dir, f"session_{timestamp}")
        os.makedirs(sub_dir)
        print(f"[make_or_get_directory] '{main_dir}' existed; created sub directory: {sub_dir}")
        return sub_dir


def run_media_exchange(camera_id:int,
                       yolo_path:str,
                       save_dir:str):
    # 1) create directory w/o overwriting
    final_save_dir = make_or_get_directory(save_dir)
    original_image_folder = os.path.join(final_save_dir, "original_images")
    detections_dir = os.path.join(final_save_dir, "detections")
    os.makedirs(original_image_folder, exist_ok=True)
    os.makedirs(detections_dir, exist_ok=True)

    # 2) home the printer
    home()

    # need modification/troubleshoot to get proper liquid volume
    evaporation = EVAPORATION_OFFSET

    # 3) initialize camera and organoid detector:
    cap = cv2.VideoCapture(camera_id)
    assert cap.isOpened(), "[run_media_exchange] camera failded to open"

    detector = OrganoidDetector(model_path=yolo_path)

    # 4) generate well locations
    well_locations = []
    for row in range(8):
        for col in range(12):
            x = SYRINGE_ORIGIN_X + col * WELL_SPACING_X
            y = SYRINGE_ORIGIN_Y + row * WELL_SPACING_Y
            well_locations.append((x,y))

    for (x, y) in well_locations:
        move_to([WATER_POS[0], WATER_POS[1], SAFE_Z]) # move to water
        move_to([WATER_POS[0], WATER_POS[1], EXTRACT_Z_DISH]) # dip into water
        control_syringe(-evaporation) # aspirate

        move_to([WATER_POS[0], WATER_POS[1], SAFE_Z]) # get out of water dish

        move_to([x, y, SAFE_Z]) # move to well
        move_to([x, y, EXTRACT_Z]) # dip into well

        control_syringe(evaporation + DISPENSE_OFFSET) # dispense
        move_to([x, y, SAFE_Z]) # get out of well.
        control_syringe(-DISPENSE_OFFSET)
        time.sleep(MIXING_WAIT_TIME)

        found = False
        attempts = 0
        while True:
            # extract old media
            move_to([x, y, EXTRACT_Z]) # dip into well
            control_syringe(-EXCHANGE_MEDIA_VOLUME) # aspirate half the old media
            move_to([x, y, CAMERA_Z]) # get out of well, up to camera height

            # check with camera:
            move_to([x + CAMERA_OFFSET[0], y + CAMERA_OFFSET[1], CAMERA_Z]) # move camera over well
            print("[run_media_exchange] Waiting for camera to stabilize...")
            time.sleep(CAMERA_STABILIZE_TIME) # wait for camera to stabilize

            # flush the camera buffer so we dont reuse an old frame
            for _ in range(4):
                cap.grab()
                time.sleep(0.1)
            ret, well_image_bgr = cap.retrieve()

            if not ret:
                raise RuntimeError("[run_media_exchange] error retrieving camera frame")
            print("[reun_media_exchange] captured frame is in bgr format from opencv")
            image_path = os.path.join(original_image_folder, f"{x}_{y}_attempt{attempts}.png") # prevents overwritting
            cv2.imwrite(image_path, well_image_bgr)
            print(f"[run_media_exchange] saved raw image: {image_path}")

            # detect
            found, conf = detector.detect(
                image_path=image_path,
                save_annotated=True,
                output_dir=detections_dir
            )
        
            if found or attempts == 5:
                print(f"[run_media_exchange] detection result for well at ({x},{y}): found={found},conf={conf}")
                break
            else:
                attempts += 1
                print("[run_media_exchange] did not find object, retrying capture")
                # return the media to well before retry
                move_to([x, y, SAFE_Z]) # move to well
                move_to([x, y, EXTRACT_Z]) # dip into well
                control_syringe(EXCHANGE_MEDIA_VOLUME + DISPENSE_OFFSET) # retrun media
                move_to([x, y, SAFE_Z]) # get out of the well
                control_syringe(-DISPENSE_OFFSET)

                time.sleep(MIXING_WAIT_TIME)
        
        # last place was above well, camera aligned, with old media in syringe
        # dispose old media:
        move_to([WASTE_MEDIA_POS[0], WASTE_MEDIA_POS[1], SAFE_Z]) 
        move_to([WASTE_MEDIA_POS[0], WASTE_MEDIA_POS[1], EXTRACT_Z_DISH]) 
        control_syringe(EXCHANGE_MEDIA_VOLUME + DISPENSE_OFFSET)
        move_to([WASTE_MEDIA_POS[0], WASTE_MEDIA_POS[1], SAFE_Z]) 
        control_syringe(-DISPENSE_OFFSET)

        # Get fresh media:
        move_to([FRESH_MEDIA_POS[0], FRESH_MEDIA_POS[1], SAFE_Z])
        move_to([FRESH_MEDIA_POS[0], FRESH_MEDIA_POS[1], EXTRACT_Z_DISH])
        control_syringe(-EXCHANGE_MEDIA_VOLUME) # aspirate fresh media
        move_to([FRESH_MEDIA_POS[0], FRESH_MEDIA_POS[1], SAFE_Z])

        # move back to well and add fresh media:
        move_to([x, y, SAFE_Z]) # move to well
        move_to([x, y, EXTRACT_Z]) # dip into well
        control_syringe(EXCHANGE_MEDIA_VOLUME + DISPENSE_OFFSET) # put in exchange media
        move_to([x, y, SAFE_Z]) # get out of well
        control_syringe(-DISPENSE_OFFSET)

        
    print('Run Moving')
    move_to([15, 15, 25])
    print('Ran Moving')
    # release camera at the end
    cap.release()


if __name__ == '__main__':
    run_media_exchange(
        camera_id=0,
        yolo_path="/home/pi/Documents/organoid/best.pt",
        save_dir="tmp/test1"
    )