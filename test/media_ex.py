'''
Test perform_media_exchange(x_well, y_well, organoid_coords)
'''

def test_perform_media_exchange():
    x_well = 100
    y_well = 100
    organoid_coords = []
    success = perform_media_exchange(x_well, y_well, organoid_coords)
    if success:
        print("Media exchange function executed successfully.")
    else:
        print("Media exchange function failed.")

if __name__ == '__main__':
    test_perform_media_exchange()
