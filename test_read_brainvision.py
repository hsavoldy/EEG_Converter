import mne
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def test_read_brainvision(vhdr_path):
    try:
        logging.debug(f"Attempting to read BrainVision file: {vhdr_path}")
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose='DEBUG')
        logging.debug("Successfully read BrainVision file.")
        print("EEG Data Info:")
        print(raw.info)
    except Exception as e:
        logging.error(f"Error reading BrainVision file: {e}")

if __name__ == "__main__":
    # Replace with your actual .vhdr file path
    vhdr_path = r"C:\Users\Giacomo\Documents\EEG_Converter\uploads\sub-1448_task-Rest_eeg.vhdr"
    
    # Ensure the path exists
    if not os.path.isfile(vhdr_path):
        logging.error(f"The .vhdr file does not exist: {vhdr_path}")
    else:
        test_read_brainvision(vhdr_path)
