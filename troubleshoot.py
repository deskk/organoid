'''
How This Script Helps Troubleshoot
Choice of Input

You can either supply an existing image file via --image_path path/to/image.png or let it read from the camera by default ("camera").
If using "camera", it captures exactly one frame from your default camera (ID=0).
Stepwise Saving

01_raw_bgr.png: Directly saves the unmodified BGR frame from OpenCV.
02_raw_rgb.png: BGR→RGB, but we have to convert back to BGR for saving, so the name indicates it’s logically “RGB content,” though physically saved in BGR color order.
03_oriented_bgr.png: If you want to check EXIF orientation corrections, we do it in PIL (RGB) and then go back to BGR.
04_resized_bgr.png: The 640×640 version of the oriented image, still in BGR.
05_annotated_rgb.png: YOLO’s .plot() returns BGR. We convert that BGR to RGB so it looks correct in typical image viewers.
Explicit Logging

You’ll see [DEBUG] statements explaining exactly what color format each image is supposed to represent and how it’s saved.
This helps you confirm at which point the image might appear incorrectly colored if the pipeline is messed up.
Check Each Saved Image

By opening each .png in a standard viewer, you can confirm if the color channels are correct (for example, if your image looks normal, or if it looks tinted or unusual).
If at any step it looks off, you can isolate which conversion step caused the problem.
Adjust YOLO

You can tweak confidence threshold (conf=0.3) or other YOLO arguments if you need.
Use

Example command line for a file-based test:

python troubleshoot_color_pipeline.py \
  --image_path test_image.png \
  --model_path /home/pi/Documents/organoid/best.pt \
  --save_dir debug_out
For a single camera frame capture test:
bash
Copy
Edit
python troubleshoot_color_pipeline.py \
  --image_path camera \
  --model_path /home/pi/Documents/organoid/best.pt \
  --save_dir debug_out
By reviewing each image in the debug_out folder, you can see exactly what YOLO “sees” and whether your color channels are correct at each step. This workflow should help you pinpoint and fix any mismatch in color channels that might be causing poor detection results.
'''


import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ExifTags
from pathlib import Path

def auto_orient_image_pil(pil_image):
    """
    Given a PIL image, check EXIF orientation and rotate if needed.
    Return a possibly rotated PIL image (still in PIL's mode, usually RGB).
    """
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
    except Exception as e:
        print("[auto_orient_image_pil] No orientation EXIF or error:", e)

    return pil_image

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    if args.image_path.lower() == "camera":
        print("[DEBUG] Attempting to read from camera (ID=0 by default).")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("[ERROR] Could not open camera (ID=0).")
        ret, frame_bgr = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("[ERROR] Could not read a frame from camera.")
        original_bgr = frame_bgr
        print("[DEBUG] Captured an image from camera in BGR format.")
    else:
        print(f"[DEBUG] Reading image from file: {args.image_path}")
        original_bgr = cv2.imread(args.image_path)
        if original_bgr is None:
            raise FileNotFoundError(f"[ERROR] Could not read file: {args.image_path}")
        print("[DEBUG] Read image from disk in BGR format (OpenCV default).")

    bgr_save_path = os.path.join(args.save_dir, "01_raw_bgr.png")
    cv2.imwrite(bgr_save_path, original_bgr)
    print(f"[DEBUG] Saved raw BGR image to: {bgr_save_path}")
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    rgb_save_path = os.path.join(args.save_dir, "02_raw_rgb.png")
    cv2.imwrite(rgb_save_path, cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR))
    print(f"[DEBUG] Saved raw RGB image to: {rgb_save_path} (internally reconverted to BGR for saving).")
    pil_rgb = Image.fromarray(original_rgb)
    oriented_pil = auto_orient_image_pil(pil_rgb)
    oriented_bgr = cv2.cvtColor(np.array(oriented_pil), cv2.COLOR_RGB2BGR)
    oriented_bgr_path = os.path.join(args.save_dir, "03_oriented_bgr.png")
    cv2.imwrite(oriented_bgr_path, oriented_bgr)
    print(f"[DEBUG] Auto-oriented via PIL. Saved oriented BGR image to: {oriented_bgr_path}")
    oriented_bgr_resized = cv2.resize(oriented_bgr, (640, 640), interpolation=cv2.INTER_AREA)
    resized_bgr_path = os.path.join(args.save_dir, "04_resized_bgr.png")
    cv2.imwrite(resized_bgr_path, oriented_bgr_resized)
    print(f"[DEBUG] Resized BGR image to 640x640. Saved to: {resized_bgr_path}")
    oriented_rgb_resized = cv2.cvtColor(oriented_bgr_resized, cv2.COLOR_BGR2RGB)

    print(f"[DEBUG] Loading YOLO model from: {args.model_path}")
    model = YOLO(args.model_path)

    results = model.predict(
        source=oriented_rgb_resized,   # Numpy array (RGB)
        imgsz=640,
        conf=0.3,
        save=False
    )
    if len(results) > 0 and len(results[0].boxes) > 0:
        print("[DEBUG] Detection found at least one object.")
    else:
        print("[DEBUG] No objects detected or detection results empty.")

    annotated_bgr = results[0].plot()
    annotated_rgb_path = os.path.join(args.save_dir, "05_annotated_rgb.png")
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(annotated_rgb_path, annotated_rgb)
    print(f"[DEBUG] Saved annotated result to: {annotated_rgb_path}")

    print("\n[INFO] Finished troubleshooting pipeline.")
    print("Check each saved image to see if color channels are correct.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Troubleshoot color channels and YOLO detection pipeline."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="camera",
        help="Path to an image file, or 'camera' to capture from default camera."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to your YOLO model, e.g. best.pt."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="debug_out",
        help="Directory to store intermediate images."
    )
    args = parser.parse_args()

    main(args)
