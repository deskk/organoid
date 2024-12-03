'''
Test control_syringe(volume_uL) Function
'''

import requests

MOONRAKER_HOST = ''
MOONRAKER_PORT = ''

SYRINGE_SPEED = 1000  # mm/min
SYRINGE_EXTRUDE_LENGTH_PER_UL = 1.0  # mm per uL, needs to be calibrated

def control_syringe(volume_uL):
    length_mm = volume_uL * SYRINGE_EXTRUDE_LENGTH_PER_UL
    gcode_cmd = f'G1 E{length_mm} F{SYRINGE_SPEED}'
    url = f'http://{MOONRAKER_HOST}:{MOONRAKER_PORT}/printer/gcode/script'
    headers = {'Content-Type': 'application/json'}
    payload = {'script': gcode_cmd}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return True
    except Exception as e:
        print('Error controlling syringe:', e)
        return False

if __name__ == '__main__':
    # aspirate 50 uL
    success = control_syringe(100)
    # # dispense 50 uL
    # success = control_syringe(-100)
    if success:
        print("Syringe moved to dispense X uL.")
    else:
        print("Failed to control the syringe.")
