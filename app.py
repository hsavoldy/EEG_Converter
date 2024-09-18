from flask import Flask, render_template, request, send_file, send_from_directory, url_for
import os
import mne
import uuid
import zipfile
import io
from werkzeug.utils import secure_filename
import logging
import base64
import matplotlib.pyplot as plt
import channel_standards
import numpy as np

app = Flask(__name__)

# Configure absolute paths based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2000 * 1024 * 1024  # Max file size: 2GB

ALLOWED_EXTENSIONS = {'edf', 'set', 'fif', 'vhdr', 'eeg', 'cnt', 'bdf', 'fdt'}


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, 'app.log')),
        logging.StreamHandler()
    ]
)
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fix_brainvision_header(vhdr_path):
    """
    Adjust the paths in the BrainVision header file to ensure correct file references.
    Ensures that both DataFile and MarkerFile entries exist and are correctly set.
    """
    logging.debug(f"Fixing BrainVision header: {vhdr_path}")
    try:
        # Read the .vhdr file
        with open(vhdr_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        base_name = os.path.splitext(os.path.basename(vhdr_path))[0]
        eeg_file = f'{base_name}.eeg'
        vmrk_file = f'{base_name}.vmrk'

        datafile_found = False
        markerfile_found = False

        # Process each line to update DataFile and MarkerFile entries
        for idx, line in enumerate(lines):
            line_lower = line.strip().lower()
            if line_lower.startswith('datafile='):
                lines[idx] = f'DataFile={eeg_file}\n'
                datafile_found = True
                logging.debug(f"Updated DataFile to: {eeg_file}")
            elif line_lower.startswith('markerfile='):
                # Set MarkerFile to point to .vmrk file
                lines[idx] = f'MarkerFile={vmrk_file}\n'
                markerfile_found = True
                logging.debug(f"Set MarkerFile to: {vmrk_file}")

        # Add DataFile entry if missing
        if not datafile_found:
            for idx, line in enumerate(lines):
                if line.strip().lower() == '[common infos]':
                    lines.insert(idx + 1, f'DataFile={eeg_file}\n')
                    datafile_found = True
                    logging.debug(f"Added missing DataFile entry: {eeg_file}")
                    break

        # Add MarkerFile entry if missing
        if not markerfile_found:
            for idx, line in enumerate(lines):
                if line.strip().lower() == '[common infos]':
                    lines.insert(idx + 2, f'MarkerFile={vmrk_file}\n')
                    markerfile_found = True
                    logging.debug(f"Added missing MarkerFile entry: {vmrk_file}")
                    break

        # Write back the modified .vhdr file
        with open(vhdr_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)

        # Create an empty .vmrk file if it doesn't exist
        vmrk_path = os.path.join(os.path.dirname(vhdr_path), vmrk_file)
        if not os.path.isfile(vmrk_path):
            with open(vmrk_path, 'w', encoding='utf-8') as vmrk:
                vmrk.write('')  # Write nothing to make it empty
            logging.debug(f"Created empty MarkerFile: {vmrk_path}")
        else:
            logging.debug(f"MarkerFile already exists: {vmrk_path}")

        # Log the final .vhdr content for debugging
        with open(vhdr_path, 'r', encoding='utf-8') as file:
            final_content = file.read()
            logging.debug(f"Final .vhdr content:\n{final_content}")

    except Exception as e:
        logging.error(f"Failed to fix BrainVision header: {e}")
        raise

def read_eeg_file(filepath):
    """
    Read EEG file using the appropriate MNE function based on file extension.
    """
    ext = filepath.split('.')[-1].lower()
    logging.debug(f"Reading EEG file with extension: .{ext}")
    try:
        if ext == 'vhdr':
            return mne.io.read_raw_brainvision(filepath, preload=True)
        elif ext == 'edf':
            return mne.io.read_raw_edf(filepath, preload=True)
        elif ext == 'set':
            return mne.io.read_raw_eeglab(filepath, preload=True)
        elif ext == 'fif':
            return mne.io.read_raw_fif(filepath, preload=True)
        elif ext == 'cnt':
            return mne.io.read_raw_cnt(filepath, preload=True)
        elif ext == 'bdf':
            return mne.io.read_raw_bdf(filepath, preload=True)
        elif ext == 'fdt':
            return
        else:
            raise ValueError(f'Unsupported file extension: .{ext}')
    except Exception as e:
        logging.error(f"Failed to read EEG file {filepath}: {e}")
        raise

def get_form_data():
    try:
        sampling_rate = float(request.form.get('sampling_rate', 0))
        filter_low = float(request.form.get('filter_low', 0))
        filter_high = float(request.form.get('filter_high', 0))
        notch_freq = float(request.form.get('notch_freq', 0))
        reference_choice = request.form.get('reference_choice')
        electrode_placement = request.form.get('electrode_placement')
        output_format = request.form.get('output_format', 'fif').lower()

        logging.debug(f"Form data - Sampling Rate: {sampling_rate}, Filter Low: {filter_low}, "
                      f"Filter High: {filter_high}, Notch Frequency: {notch_freq}, "
                      f"Electrode Placement: {electrode_placement}, "
                      f"Reference Choice: {reference_choice}, Output Format: {output_format}")
        return sampling_rate, filter_low, filter_high, notch_freq, electrode_placement, reference_choice, output_format
    except ValueError as ve:
        logging.error(f"Invalid form data: {ve}")
        raise

def save_uploaded_files(files):
    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if filename == '':
                logging.warning("Invalid file name detected.")
                raise ValueError('Invalid file name.')
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(upload_path)
                uploaded_files.append(filename)
                logging.debug(f"Saved uploaded file: {upload_path}")
            except Exception as e:
                logging.error(f"Failed to save file {filename}: {e}")
                raise
        else:
            logging.warning(f"File {file.filename} is not allowed.")
            raise ValueError(f'File {file.filename} is not allowed.')
    return uploaded_files

def group_files_by_dataset(uploaded_files):
    datasets = {}
    for filename in uploaded_files:
        base_name = os.path.splitext(filename)[0]
        datasets.setdefault(base_name, []).append(filename)
    logging.debug(f"Grouped datasets: {datasets}")
    return datasets

def validate_datasets(datasets):
    for base_name, file_group in datasets.items():
        extensions = {os.path.splitext(f)[1].lower() for f in file_group}
        if '.vhdr' in extensions:
            required_extensions = {'.vhdr', '.eeg'}
            missing = required_extensions - extensions
            if missing:
                logging.error(f"Dataset {base_name} is missing files: {missing}")
                raise ValueError(f'Dataset {base_name} is missing files: {", ".join(missing)}')

def process_dataset(base_name, file_group, sampling_rate, filter_low, filter_high, notch_freq, electrode_placement, reference_choice, output_format):
    logging.debug(f"Processing dataset: {base_name} with files: {file_group}")
    try:
        if any(f.endswith('.vhdr') for f in file_group):
            vhdr_file = next(f for f in file_group if f.endswith('.vhdr'))
            vhdr_path = os.path.join(app.config['UPLOAD_FOLDER'], vhdr_file)
            if not os.path.isfile(vhdr_path):
                logging.error(f".vhdr file not found: {vhdr_path}")
                raise FileNotFoundError(f'.vhdr file not found: {vhdr_path}')

            fix_brainvision_header(vhdr_path)
            logging.debug(f"Successfully fixed vhdr_path: {vhdr_path}")

            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose='DEBUG')
            logging.debug(f"Successfully read BrainVision file: {vhdr_path}")
        else:
            file_not_found = True
            file_index = 0
            while(file_index < len(file_group)):
                file_name = file_group[file_index]
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
                if not os.path.isfile(file_path):
                    logging.error(f"EEG file not found: {file_path}")
                    raise FileNotFoundError(f'EEG file not found: {file_path}')
                raw = read_eeg_file(file_path)
                if(raw is not None):
                    file_not_found = False
                    logging.debug(f"Successfully read EEG file: {file_path}")
                    break
                file_index += 1
            if(file_not_found):
                logging.error(f"No EEG files found in the uploaded dataset.")
                raise ValueError(f"No EEG files found in the uploaded dataset.")

        
        input_raw = raw  # Original raw before processing

        # Make a copy of the original raw for plotting
        output_raw = input_raw.copy()

        original_sfreq = raw.info['sfreq']
        logging.debug(f"Original sampling rate: {original_sfreq} Hz")
        if sampling_rate > 0 and sampling_rate != original_sfreq:
            output_raw.resample(sampling_rate)
            logging.debug(f"Resampled data to: {sampling_rate} Hz")

        if notch_freq > 0:
            output_raw.notch_filter(freqs=[notch_freq])
            logging.debug(f"Applied notch filter at: {notch_freq} Hz")

        if filter_low > 0 or filter_high > 0:
            output_raw.filter(l_freq=filter_low if filter_low > 0 else None,
                       h_freq=filter_high if filter_high > 0 else None)
            logging.debug(f"Applied filter - Low: {filter_low} Hz, High: {filter_high} Hz")


        # Set montage with error handling
        if electrode_placement:
            try:
                channel_names = output_raw.info['ch_names']
                num_electrodes = len(channel_names)
                montage = mne.channels.make_standard_montage(electrode_placement)
                output_raw.set_montage(montage, on_missing='warn')
                logging.debug(f"Set electrode montage to: {electrode_placement}")
            except Exception as e:
                logging.error(f"Failed to set montage '{electrode_placement}': {e}")
                raise ValueError(f"Failed to set montage '{electrode_placement}': {e}")

        if reference_choice:
            try:
                if reference_choice == 'average':
                    output_raw.set_eeg_reference('average')
                    logging.debug("Set EEG reference to average.")
                else:
                    output_raw.set_eeg_reference([reference_choice])
                    logging.debug(f"Set EEG reference to: {reference_choice}")
            except Exception as e:
                logging.error(f"Failed to set reference '{reference_choice}': {e}")
                raise ValueError(f"Failed to set reference '{reference_choice}': {e}")

        processed_filename = f'processed_{base_name}.{output_format}'
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

        if output_format == 'fif':
            output_raw.save(processed_path, overwrite=True)
            logging.debug(f"Saved processed FIF file: {processed_path}")
        elif output_format == 'edf':
            mne.export.export_raw(processed_path, output_raw, fmt='edf')
            logging.debug(f"Saved processed EDF file: {processed_path}")
        else:
            logging.error(f'Unsupported output format: {output_format}')
            raise ValueError(f'Unsupported output format: {output_format}')

        return processed_path, input_raw, output_raw
    except Exception as e:
        logging.error(f"Failed to process dataset: {e}")
        raise

   
def generate_plot(input_raw, output_raw):
    """
    Generates a plot with two subplots:
    - Top subplot: Input EEG data for the first channel.
    - Bottom subplot: Output EEG data for the same channel.
    Both plots share the same x-axis and y-axis scales.
    """
    first_channel = input_raw.ch_names[1]
    logging.debug(f"Selected channel for plotting: {first_channel}")

    # Extract data and times for input
    input_data, input_times = input_raw[first_channel, :]
    input_data = input_data[0]

    # Extract data and times for output
    output_data, output_times = output_raw[first_channel, :]
    output_data = output_data[0]

    # Determine y-axis limits based on both datasets
    combined_data = np.concatenate((input_data, output_data))
    y_min = combined_data.min()
    y_max = combined_data.max()

    # Create two subplots vertically
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
       
    # Plot input data
    axs[0].plot(input_times, input_data, label=f'Input {first_channel}', color='blue')
    axs[0].set_title(f'Input {first_channel}')
    axs[0].set_ylabel('Amplitude (µV)')
    axs[0].legend()
    axs[0].set_ylim(y_min, y_max)
    axs[0].grid(True)

    # Plot output data
    axs[1].plot(output_times, output_data, label=f'Output {first_channel}', color='green')
    axs[1].set_title(f'Output {first_channel}')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude (µV)')
    axs[1].legend()
    axs[1].set_ylim(y_min, y_max)
    axs[1].grid(True)

    plt.tight_layout()

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # Encode the image to base64
    plot_image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    logging.debug("Generated and encoded the plot image.")
    return plot_image_base64

def create_zip_file(processed_files):
    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for item in processed_files:
                arcname = os.path.basename(item)
                zf.write(item, arcname=arcname)
                logging.debug(f"Added {item} to ZIP as {arcname}")
        zip_buffer.seek(0)
        logging.debug("Created ZIP file of processed files.")
        return zip_buffer
    except Exception as e:
        logging.error(f"Failed to create ZIP file: {e}")
        raise

def cleanup_files():
    try:
        for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER']]:
            for f in os.listdir(folder):
                file_path = os.path.join(folder, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logging.debug(f"Removed file: {file_path}")
                elif os.path.isdir(file_path):
                    try:
                        os.rmdir(file_path)
                        logging.debug(f"Removed directory: {file_path}")
                    except OSError:
                        logging.warning(f"Could not remove directory (may not be empty): {file_path}")
        logging.debug("Cleaned up uploaded and processed directories.")
    except Exception as e:
        logging.error(f"Failed during cleanup: {e}")

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        logging.debug("Received POST request for file upload.")

        try:
            sampling_rate, filter_low, filter_high, notch_freq, electrode_placement, reference_choice, output_format = get_form_data()
        except ValueError as ve:
            return f'Invalid form data: {ve}', 400

        files = request.files.getlist('files')
        if not files:
            logging.warning("No files uploaded.")
            return 'No files uploaded', 400

        try:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
            logging.debug(f"Ensured upload and processed directories exist: {app.config['UPLOAD_FOLDER']}, {app.config['PROCESSED_FOLDER']}")
        except Exception as e:
            logging.error(f"Failed to create directories: {e}")
            return f'Failed to create directories: {e}', 500

        try:
            uploaded_files = save_uploaded_files(files)
        except ValueError as ve:
            return str(ve), 400
        except Exception as e:
            return f'Failed to save files: {e}', 500

        datasets = group_files_by_dataset(uploaded_files)

        try:
            validate_datasets(datasets)
        except ValueError as ve:
            return str(ve), 400

        processed_files = []
        plot_image_base64 = None

        for base_name, file_group in datasets.items():
            try:
                logging.debug(f"Processing dataset: {base_name} with files: {file_group}")
                processed_path, input_raw, output_raw = process_dataset(
                    base_name,
                    file_group,
                    sampling_rate,
                    filter_low,
                    filter_high,
                    notch_freq,
                    electrode_placement,
                    reference_choice,
                    output_format
                )
                if(processed_path is not None):
                    processed_files.append(processed_path)

                    if not plot_image_base64 and input_raw is not None:
                        plot_image_base64 = generate_plot(input_raw, output_raw)
            except Exception as e:
                return f'Error encountered: {e}', 500

        if not processed_files:
            logging.warning("No files were processed.")
            return 'No files were processed.', 400
        
        logging.debug(f"Processed files: {processed_files}")

        try:
            zip_buffer = create_zip_file(processed_files)
        except Exception as e:
            return f'Failed to create ZIP file: {e}', 500

        # Save the ZIP file to the server with a unique filename
        zip_filename = f'processed_files_{uuid.uuid4().hex}.zip'
        zip_path = os.path.join(app.config['PROCESSED_FOLDER'], zip_filename)
        try:
            with open(zip_path, 'wb') as f:
                f.write(zip_buffer.getvalue())
            logging.debug(f"Saved ZIP file to {zip_path}")
        except Exception as e:
            logging.error(f"Failed to save ZIP file: {e}")
            return f'Failed to save ZIP file: {e}', 500

        # Pass the zip filename to the template
        return render_template('result.html', plot_image=plot_image_base64, zip_filename=zip_filename)

    else:
        return render_template('upload.html')

@app.route('/download', methods=['GET'])
def download_zip():
    filename = request.args.get('filename')
    logging.debug(f"Received filename for download: {filename}")
    if not filename:
        logging.error("No filename provided for download.")
        return 'No filename provided.', 400

    zip_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(zip_path):
        logging.error(f"File not found: {zip_path}")
        return 'File not found.', 404

    try:
        response = send_from_directory(
            directory=app.config['PROCESSED_FOLDER'],
            path=filename,
            as_attachment=True,
            mimetype='application/zip',
            download_name='processed_files.zip'  # You can use filename if you prefer
        )
        logging.debug(f"Sending ZIP file {zip_path}")
        return response
    except Exception as e:
        logging.error(f"Failed to send ZIP file: {e}")
        return f'Failed to download ZIP file: {e}', 500
    finally:
        # Remove the ZIP file after sending it
        try:
            os.remove(zip_path)
            logging.debug(f"Deleted ZIP file {zip_path} after sending.")
        except Exception as e:
            logging.error(f"Failed to delete ZIP file: {e}")
    

if __name__ == '__main__':
    try:
        app.run(debug=True)
        cleanup_files()

    except Exception as e:
        logging.critical(f"Failed to start the Flask app: {e}")
