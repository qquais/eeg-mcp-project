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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

UPLOAD_DIR = './data'
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route('/read-edf', methods=['POST'])
def read_edf():
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
        eeg_data = {f'channel_{i+1}': data[ch].tolist() for i, ch in enumerate(eeg_channels)}

        os.remove(filepath)
        return jsonify({'channels': eeg_data})

    except Exception as e:
        print(f"Error in read-edf: {e}")
        return jsonify({'error': str(e)}), 500

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

        print(f"Sampling Rate: {sampling_rate}")
        print(f"Data Shape: {data.shape}")
        print(f"EEG Channels: {eeg_channels}")

        band_powers = {}
        for i, ch in enumerate(eeg_channels):
            print(f"Processing channel {ch}: data[ch].shape = {data[ch].shape}")

            bands, rel_bands = DataFilter.get_avg_band_powers(
                data[ch],
                sampling_rate,
                True,    # ✅ Correct parameter for apply_filter
                0.5,
                40.0
            )

            band_powers[f'channel_{i+1}'] = {
                'delta': bands[0],
                'theta': bands[1],
                'alpha': bands[2],
                'beta': bands[3],
                'gamma': bands[4]
            }

        os.remove(filepath)
        return jsonify({'features': band_powers})

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
