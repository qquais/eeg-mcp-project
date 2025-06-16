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


def read_edf_file(filepath):
    raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
    data, times = raw[:, :]
    return data, raw.info


# Actual reading the EEG files using MNE.
@app.route('/read-edf', methods=['POST'])
def read_edf():
    try:
        file = request.files.get('file')
        if file is None or file.filename == '':
            return jsonify({"error": "No file uploaded"}), 400

        file_id = str(uuid.uuid4())
        filepath = os.path.join(UPLOAD_DIR, f'{file_id}.edf')
        file.save(filepath)

        data, info = read_edf_file(filepath)
        os.remove(filepath)

        preview = data[:3, :10].tolist()  # preview: first 3 channels × 10 samples

        return jsonify({
            "status": "success",
            "shape": list(data.shape),
            "preview": preview,
            "sfreq": info['sfreq'],
            "channels": info['ch_names'][:3]  # preview channel names
        })

    except Exception as e:
        import traceback
        traceback.print_exc()  # Print full error for debugging
        return jsonify({"status": "error", "message": str(e)}), 500



# This was just EDF handling which is technically emulated EEG input, not actual file parsing.
# @app.route('/read-edf', methods=['POST'])
# def read_edf():
#     try:
#         file = request.files['file']
#         file_id = str(uuid.uuid4())
#         filepath = os.path.join(UPLOAD_DIR, f'{file_id}.edf')
#         file.save(filepath)

#         params = BrainFlowInputParams()
#         params.file = filepath
#         board_id = BoardIds.SYNTHETIC_BOARD.value

#         board = BoardShim(board_id, params)
#         board.prepare_session()
#         board.start_stream()
#         time.sleep(2)

#         data = board.get_board_data()

#         board.stop_stream()
#         board.release_session()

#         eeg_channels = BoardShim.get_eeg_channels(board_id)
#         eeg_data = {f'channel_{i+1}': data[ch].tolist() for i, ch in enumerate(eeg_channels)}

#         os.remove(filepath)
#         return jsonify({'channels': eeg_data})

#     except Exception as e:
#         print(f"Error in read-edf: {e}")
#         return jsonify({'error': str(e)}), 500

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
