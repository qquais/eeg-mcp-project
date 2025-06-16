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


@app.route('/read-edf', methods=['POST'])
def read_edf():
    try:
        # ✅ Get engine from URL query parameter (temporary fix as we don't have a UI yet)
        engine = request.args.get('engine', 'mne').strip().lower()
        print(f"[DEBUG] Selected engine: '{engine}'")

        # Get the uploaded file
        file = request.files.get('file')
        if not file or file.filename == '':
            return jsonify({"error": "No file uploaded"}), 400

        # Save file temporarily
        file_id = str(uuid.uuid4())
        filepath = os.path.join(UPLOAD_DIR, f'{file_id}.edf')
        file.save(filepath)

        # MNE path
        if engine == 'mne':
            print("[DEBUG] Running MNE reader")
            data, info = read_with_mne(filepath)
            preview = data[:3, :10].tolist()
            sfreq = info['sfreq']
            channels = info['ch_names'][:3]

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
            preview = {
                f'channel_{i+1}': data[ch][:10].tolist()
                for i, ch in enumerate(eeg_channels[:3])
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
            print(f"[DEBUG] Invalid engine received: '{engine}'")
            return jsonify({"error": f"Invalid engine '{engine}'. Use 'mne' or 'brainflow'."}), 400

        os.remove(filepath)
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/visualize-edf', methods=['POST'])
def visualize_edf():
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

        plt.figure(figsize=(14, 7))
        for i, ch in enumerate(eeg_channels):
            plt.plot(data[ch], label=f'Channel {i+1}')

        plt.title('EEG Signals')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude (uV)')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='png')
        img_io.seek(0)
        plt.close()

        os.remove(filepath)
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        print(f"Error in visualize-edf: {e}")
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
