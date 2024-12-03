'''
 Test move_to(position, speed, blocking) Function
'''

import requests

MOONRAKER_HOST = ''
MOONRAKER_PORT = ''

def move_to(position, speed=3000, blocking=False):
    x, y, z = position
    gcode_cmd = f'G1 X{x} Y{y} Z{z} F{speed}'
    url = f'http://{MOONRAKER_HOST}:{MOONRAKER_PORT}/printer/gcode/script'
    headers = {'Content-Type': 'application/json'}
    payload = {'script': gcode_cmd}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print('Error sending G-code command:', e)
        return False
    return True

if __name__ == '__main__':
    test_position = (50, 50, 50)
    success = move_to(test_position, speed=3000, blocking=True)
    if success:
        print(f"Moved to position {test_position}")
    else:
        print("Failed to move to the specified position.")
