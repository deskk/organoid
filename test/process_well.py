def test_process_well():
    # Generate well positions
    well_positions = generate_well_positions()
    # Select a test well
    well_id = 'A1'  # Ensure this well is safe to test
    if well_id in well_positions:
        success = process_well(well_id, well_positions)
        if success:
            print(f"Processed well {well_id} successfully.")
        else:
            print(f"Failed to process well {well_id}.")
    else:
        print(f"Well {well_id} not found in well positions.")

if __name__ == '__main__':
    test_process_well()
