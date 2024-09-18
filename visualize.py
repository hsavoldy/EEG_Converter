import mne
import matplotlib.pyplot as plt

def plot_eeg_traces(eeg_file, channels, start_time=0, duration=10):
    """
    Plots specified EEG channels from an EEG file for comparison.

    Parameters:
    - eeg_file: str, path to the EEG file (.fif, .edf, etc.)
    - channels: list, list of channel names to plot (e.g., ['Fp1', 'Fp2', 'C3'])
    - start_time: float, start time in seconds for the plot (default 0)
    - duration: float, duration in seconds for the plot (default 10 seconds)
    """
    # Load the EEG data using MNE
    raw = mne.io.read_raw_fif(eeg_file, preload=True)

    # Apply any necessary pre-processing (e.g., filtering)
    # raw.filter(1, 40)  # Example: filter the data between 1-40 Hz (optional)

    # Pick the channels specified by the user
    picks = mne.pick_channels(raw.info['ch_names'], include=channels)

    # Create the time window to visualize
    start_sample = int(start_time * raw.info['sfreq'])  # convert start time to sample index
    end_sample = int((start_time + duration) * raw.info['sfreq'])  # end sample index

    # Extract the data from the raw object for the specified time window
    data, times = raw[picks, start_sample:end_sample]

    # Plot the data for each selected channel
    plt.figure(figsize=(12, 6))
    for idx, channel in enumerate(channels):
        plt.plot(times, data[idx, :] + idx * 100e-6, label=channel)  # Offset each channel for clarity

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.title('EEG Data')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# Example usage:
# plot_eeg_traces('sample_eeg_file.fif', channels=['Fp1', 'Fp2', 'C3', 'C4'], start_time=0, duration=10)
