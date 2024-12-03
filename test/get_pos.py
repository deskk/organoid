import requests

MOONRAKER_HOST = '172.20.10.2'  # Replace with your Moonraker host IP
MOONRAKER_PORT = 7125

def get_position():
    url = f'http://{MOONRAKER_HOST}:{MOONRAKER_PORT}/printer/objects/query?toolhead'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        position = data['result']['status']['toolhead']['position']
        return position  # [X, Y, Z, E]
    except Exception as e:
        print('Error getting position:', e)
        return None

if __name__ == '__main__':
    pos = get_position()
    print(f"Current position: {pos}")
