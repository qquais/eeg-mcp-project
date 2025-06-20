from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations
from flask import Flask, jsonify, request, send_file
import numpy as np
import time
import os
import uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
import io
import mne

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

UPLOAD_DIR = './data'
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Helper function for reading EDF files using MNE
# This function reads an EDF file and returns the data and info.
def read_with_mne(filepath):
    raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
    data, times = raw[:, :]
    return data, raw.info


# Helper function to read Emulated EEG data using BrainFlow's SYNTHETIC_BOARD
# Emulated EEG using BrainFlow’s SYNTHETIC_BOARD
def read_with_brainflow(filepath):
    params = BrainFlowInputParams()
    params.file = filepath
    board_id = BoardIds.SYNTHETIC_BOARD.value

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    time.sleep(2)

    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sfreq = BoardShim.get_sampling_rate(board_id)

    return data, eeg_channels, sfreq


# Common helper to handle upload and parameters
def handle_upload_and_params(request):
    file = request.files.get('file')
    if not file or file.filename == '':
        raise ValueError("No file uploaded")

    engine = request.args.get('engine', 'mne').strip().lower()
    preview_channels = int(request.args.get('preview_channels', 3))
    preview_samples = int(request.args.get('preview_samples', 10))

    file_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_DIR, f'{file_id}.edf')
    file.save(filepath)

    return filepath, engine, preview_channels, preview_samples

# Helper function to compute and plot Power Spectral Density (PSD) using BrainFlow
# Now Configurable number of channels (max_channels)
def compute_band_power_plot_brainflow(data, eeg_channels, sampling_rate, max_channels=5):
    band_ranges = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }

    band_powers = {band: [] for band in band_ranges}
    freqs_global = None
    plt.figure(figsize=(10, 6))

    for i, ch in enumerate(eeg_channels[:max_channels]):
        psd, freqs = DataFilter.get_psd_welch(
            data[ch],
            nfft=256,
            overlap=128,
            sampling_rate=sampling_rate,
            window=WindowOperations.HANNING.value
        )
        if freqs_global is None:
            freqs_global = freqs
        plt.plot(freqs, psd, label=f'Ch-{i+1}')
        for band, (low, high) in band_ranges.items():
            idx = np.where((freqs >= low) & (freqs <= high))
            avg_power = np.mean(psd[idx])
            band_powers[band].append(avg_power)

    band_powers_avg = {band: float(np.mean(vals)) for band, vals in band_powers.items()}
    plt.title("Power Spectral Density (BrainFlow)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()
    return band_powers_avg, fig


# Helper function to compute and plot Power Spectral Density (PSD) using MNE
# Now Configurable number of channels (max_channels)
def compute_band_power_plot_mne(raw, max_channels=5):
    band_ranges = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }

    raw = raw.copy().pick_types(eeg=True).load_data()
    data, times = raw[:, :]
    sfreq = raw.info['sfreq']
    max_ch = min(max_channels, data.shape[0])
    band_powers = {band: [] for band in band_ranges}

    plt.figure(figsize=(10, 6))

    for i in range(max_ch):
        ch_data = data[i]
        freqs, psd = welch(ch_data, fs=sfreq, nperseg=256)
        plt.plot(freqs, psd, label=raw.ch_names[i])
        for band, (low, high) in band_ranges.items():
            idx = np.where((freqs >= low) & (freqs <= high))
            avg_power = np.mean(psd[idx])
            band_powers[band].append(avg_power)

    band_powers_avg = {band: float(np.mean(vals)) for band, vals in band_powers.items()}
    plt.title("Power Spectral Density (MNE)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (uV²/Hz)")
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()
    return band_powers_avg, fig


@app.route('/read-edf', methods=['POST'])
def read_edf():
    try:
        # Get engine and preview params
        filepath, engine, preview_channels, preview_samples = handle_upload_and_params(request)
        print(f"[DEBUG] Selected engine: '{engine}'")
        print(f"[DEBUG] Preview config: {preview_channels} channels × {preview_samples} samples")

        # MNE path
        if engine == 'mne':
            print("[DEBUG] Running MNE reader")
            data, info = read_with_mne(filepath)
            sfreq = info['sfreq']

            # Clip preview to available data
            max_ch = min(preview_channels, data.shape[0])
            max_sm = min(preview_samples, data.shape[1])

            preview = data[:max_ch, :max_sm].tolist()
            channels = info['ch_names'][:max_ch]

            response = {
                "status": "success",
                "engine": "mne",
                "shape": list(data.shape),
                "preview": preview,
                "sfreq": sfreq,
                "channels": channels
            }

        # BrainFlow path
        elif engine == 'brainflow':
            print("[DEBUG] Running BrainFlow reader")
            data, eeg_channels, sfreq = read_with_brainflow(filepath)

            max_ch = min(preview_channels, len(eeg_channels))
            max_sm = min(preview_samples, data.shape[1])

            preview = {
                f'channel_{i+1}': data[ch][:max_sm].tolist()
                for i, ch in enumerate(eeg_channels[:max_ch])
            }

            response = {
                "status": "success",
                "engine": "brainflow",
                "shape": [len(eeg_channels), data.shape[1]],
                "preview": preview,
                "sfreq": sfreq,
                "channels": [f'channel_{i+1}' for i in range(len(preview))]
            }

        else:
            return jsonify({"error": f"Invalid engine '{engine}'."}), 400

        os.remove(filepath)
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/visualize-edf', methods=['POST'])
def visualize_edf():
    try:
        # Reuse upload handler
        filepath, engine, preview_channels, preview_samples = handle_upload_and_params(request)

        img_io = io.BytesIO()

        # MNE Visualization path
        if engine == 'mne':
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            raw.pick_channels(raw.ch_names[:preview_channels])

            fig = raw.plot(n_channels=preview_channels,
                           duration=preview_samples / raw.info['sfreq'],
                           show=False, scalings='auto',
                           title="EEG Preview (MNE)")
            fig.savefig(img_io, format='png')
            plt.close(fig)
            

        # BrainFlow Visualization path
        elif engine == 'brainflow':
            data, eeg_channels, _ = read_with_brainflow(filepath)
            max_ch = min(preview_channels, len(eeg_channels))
            max_sm = min(preview_samples, data.shape[1])
            selected_data = [data[ch][:max_sm] for ch in eeg_channels[:max_ch]]

            plt.figure(figsize=(12, 6))
            for i, channel_data in enumerate(selected_data):
                plt.plot(channel_data + i * 100, label=f'Channel {i+1}')

            plt.title("EEG Preview (BrainFlow)")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude + Offset (uV)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(img_io, format='png')
            plt.close()

        else:
            return jsonify({"error": f"Invalid engine '{engine}'"}), 400

        img_io.seek(0)
        os.remove(filepath)
        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='eeg_visualization.png')


    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/psd-edf', methods=['POST'])
def psd_edf():
    try:
        # Get uploaded file and parameters
        filepath, engine, preview_channels, preview_samples = handle_upload_and_params(request)
        img_io = io.BytesIO()

        if engine == 'mne':
            # MNE path
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            raw.pick_channels(raw.ch_names[:preview_channels])

            # Compute PSD and band powers
            band_powers, fig = compute_band_power_plot_mne(raw, preview_channels)
            fig.savefig(img_io, format='png')
            plt.close(fig)

        elif engine == 'brainflow':
            # BrainFlow path
            data, eeg_channels, sampling_rate = read_with_brainflow(filepath)

            # Compute PSD and band powers
            band_powers, fig = compute_band_power_plot_brainflow(data, eeg_channels, sampling_rate, preview_channels)
            fig.savefig(img_io, format='png')
            plt.close(fig)

        else:
            return jsonify({"error": f"Invalid engine '{engine}'"}), 400

        img_io.seek(0)
        os.remove(filepath)

        # Send image + band powers in header
        response = send_file(
            img_io,
            mimetype='image/png',
            as_attachment=True,
            download_name='eeg_psd.png'
        )
        response.headers['X-Band-Powers'] = str(band_powers)
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/filter-edf', methods=['POST'])
def filter_edf():
    try:
        file = request.files['file']
        file_id = str(uuid.uuid4())
        filepath = os.path.join(UPLOAD_DIR, f'{file_id}.edf')
        file.save(filepath)

        params = BrainFlowInputParams()
        params.file = filepath
        board_id = BoardIds.SYNTHETIC_BOARD.value

        board = BoardShim(board_id, params)
        board.prepare_session()
        board.start_stream()
        time.sleep(2)

        data = board.get_board_data()

        board.stop_stream()
        board.release_session()

        eeg_channels = BoardShim.get_eeg_channels(board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_id)

        print(f"Applying Band-pass Filter: 0.5Hz - 40Hz on {len(eeg_channels)} channels")

        center_freq = (0.5 + 40.0) / 2  # 20.25 Hz
        band_width = 40.0 - 0.5         # 39.5 Hz

        for ch in eeg_channels:
            DataFilter.perform_bandpass(
                data[ch],         
                sampling_rate,    
                center_freq,      
                band_width,       
                4,                
                FilterTypes.BUTTERWORTH.value,  
                0                 
            )

        filtered_data = {f'channel_{i+1}': data[ch].tolist() for i, ch in enumerate(eeg_channels)}

        os.remove(filepath)
        return jsonify({'filtered_data': filtered_data})

    except Exception as e:
        print(f"Error in filter-edf: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/features-edf', methods=['POST'])
def features_edf():
    try:
        file = request.files['file']
        file_id = str(uuid.uuid4())
        filepath = os.path.join(UPLOAD_DIR, f'{file_id}.edf')
        file.save(filepath)

        params = BrainFlowInputParams()
        params.file = filepath
        board_id = BoardIds.SYNTHETIC_BOARD.value

        board = BoardShim(board_id, params)
        board.prepare_session()
        board.start_stream()
        time.sleep(2)

        data = board.get_board_data()
        board.stop_stream()
        board.release_session()

        eeg_channels = BoardShim.get_eeg_channels(board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_id)

        bands, _ = DataFilter.get_avg_band_powers(
            data,
            eeg_channels,
            sampling_rate,
            apply_filter=True
        )

        band_names = ["delta", "theta", "alpha", "beta", "gamma"]
        averaged_powers = dict(zip(band_names, bands))

        os.remove(filepath)
        return jsonify({"features": averaged_powers})

    except Exception as e:
        print(f"Error in features-edf: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/summary-edf', methods=['POST'])
def summary_edf():
    try:
        file = request.files['file']
        file_id = str(uuid.uuid4())
        filepath = os.path.join(UPLOAD_DIR, f'{file_id}.edf')
        file.save(filepath)

        params = BrainFlowInputParams()
        params.file = filepath
        board_id = BoardIds.SYNTHETIC_BOARD.value

        board = BoardShim(board_id, params)
        board.prepare_session()
        board.start_stream()
        time.sleep(2)

        data = board.get_board_data()

        board.stop_stream()
        board.release_session()

        eeg_channels = BoardShim.get_eeg_channels(board_id)
        summary = {}

        for i, ch in enumerate(eeg_channels):
            signal = data[ch]
            summary[f'channel_{i+1}'] = {
                'mean': np.mean(signal),
                'std': np.std(signal),
                'min': float(np.min(signal)),
                'max': float(np.max(signal))
            }

        os.remove(filepath)
        return jsonify({'summary': summary})

    except Exception as e:
        print(f"Error in summary-edf: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
