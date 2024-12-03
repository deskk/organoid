'''
Test perform_media_exchange(x_well, y_well, organoid_coords)
'''

def test_perform_media_exchange():
    # Define test well position
    x_well = 100  # Replace with a safe test coordinate
    y_well = 100  # Replace with a safe test coordinate
    organoid_coords = []  # Assume no organoids detected for this test

    # Call the function
    success = perform_media_exchange(x_well, y_well, organoid_coords)
    if success:
        print("Media exchange function executed successfully.")
    else:
        print("Media exchange function failed.")

if __name__ == '__main__':
    test_perform_media_exchange()
