import argparse
import torch
import numpy as np
import os
import sys
import time
import json
import mne
import scipy.io
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from train_gpt import make_model

PROTOTYPE_FILE = 'p300_prototypes_unified.json'
REQUIRED_CHANNELS_NEUROGPT = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T2',
    'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2'
]
TARGET_SAMPLING_RATE = 250
BANDPASS_FREQS = [0.5, 100.0]
NOTCH_FREQ = 60.0
FILTER_TRANS_BANDWIDTH_SHORT_EPOCH = 0.25
REQUIRED_SAMPLES = 3650
P300_CHANNELS = ['Fz', 'Cz', 'P3', 'P4', 'Pz', 'O1', 'O2', 'Oz']
EPOCH_WINDOW_P300 = [-0.1, 0.5]

MODEL_CONFIG = {
    'pretrained_model': 'pretrained_model/pytorch_model.bin',
    'chunk_len': 500,
    'num_chunks': 8,
    'chunk_ovlp': 50,
    'n_chans': 22,
    'n_filters_time': 40,
    'filter_time_length': 25,
    'pool_time_length': 75,
    'stride_avg_pool': 15,
    'embedding_dim': 1024,
    'num_hidden_layers': 4,
    'num_attention_heads': -1,
    'intermediate_dim_factor': 4,
    'hidden_activation': 'gelu_new',
    'dropout': 0.1,
    'num_hidden_layers_embedding_model': 1,
    'num_hidden_layers_unembedding_model': 1,
    'n_positions': 512,
    'num_decoding_classes': 4,
    'ft_only_encoder': False,
    'training_style': 'decoding',
    'architecture': 'GPT',
    'freeze_embedder': False,
    'freeze_unembedder': False,
    'freeze_decoder': False,
    'freeze_decoder_without_pooler_heads': False,
    'freeze_encoder': False,
    'use_encoder': True # Explicitly use the encoder
}

# ========================================================================================
# --- INFERENCE ---
# ========================================================================================

def setup_model_and_device(config):
    """Loads the model and sets up the device."""
    print("--- Setting up model and device ---")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = make_model(config)
    model.from_pretrained(config['pretrained_model'])
    model.to(device)
    model.eval()
    
    encoder = model.encoder
    if encoder is None:
        raise RuntimeError("Model was created without an encoder. Check the 'use_encoder' config.")
    
    print("--- Model loaded successfully ---")
    return encoder, device

def preprocess_eeg_for_model(eeg_data, config):
    """Splits the raw EEG signal into overlapping chunks for the model."""
    if eeg_data.shape[0] != config['n_chans']:
        raise ValueError(f"Expected {config['n_chans']} channels, but got {eeg_data.shape[0]}")

    total_len = eeg_data.shape[1]
    chunk_len = config['chunk_len']
    num_chunks = config['num_chunks']
    ovlp = config['chunk_ovlp']
    
    chunk_stride = chunk_len - ovlp
    required_len = (num_chunks - 1) * chunk_stride + chunk_len
    
    if total_len < required_len:
        raise ValueError(f"EEG data is too short. Needs at least {required_len} samples, but got {total_len}.")

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_stride
        end = start + chunk_len
        chunk = eeg_data[:, start:end]
        chunks.append(chunk)
    
    chunk_tensor = torch.from_numpy(np.array(chunks, dtype=np.float32))
    return chunk_tensor.unsqueeze(0)

def get_embedding(eeg_data, encoder, device, config):
    """Runs a single preprocessed EEG epoch through the model to get an embedding."""
    input_tensor = preprocess_eeg_for_model(eeg_data, config)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        chunk_embeddings = encoder(input_tensor)
        pooled_chunk_embeddings = torch.mean(chunk_embeddings, dim=1)
        final_embedding = torch.mean(pooled_chunk_embeddings, dim=0)
    
    return final_embedding.cpu().numpy()

# ========================================================================================
# --- DATA PREPARATION & PROCESSING ---
# ========================================================================================

def process_single_epoch(trial_raw, config):
    """Takes a cropped MNE raw object for one epoch and fully preprocesses it."""
    # a. Pick channels (already done before this function)
    # b. Resample
    if trial_raw.info['sfreq'] != TARGET_SAMPLING_RATE:
        trial_raw.resample(TARGET_SAMPLING_RATE, verbose=False)

    # c. Apply filters
    trial_raw.filter(l_freq=BANDPASS_FREQS[0], h_freq=BANDPASS_FREQS[1], l_trans_bandwidth=FILTER_TRANS_BANDWIDTH_SHORT_EPOCH, h_trans_bandwidth=FILTER_TRANS_BANDWIDTH_SHORT_EPOCH, verbose=False)
    trial_raw.notch_filter(freqs=NOTCH_FREQ, trans_bandwidth=FILTER_TRANS_BANDWIDTH_SHORT_EPOCH, verbose=False)
    trial_raw.set_eeg_reference('average', projection=True, verbose=False)

    # d. Get data as NumPy array
    eeg_data = trial_raw.get_data()

    # e. Pad channels to 22
    if eeg_data.shape[0] < config['n_chans']:
        pad_width = config['n_chans'] - eeg_data.shape[0]
        padding = np.zeros((pad_width, eeg_data.shape[1]))
        eeg_data = np.vstack((eeg_data, padding))

    # f. Normalize (Z-score)
    mean = eeg_data.mean(axis=1, keepdims=True)
    std = eeg_data.std(axis=1, keepdims=True)
    std[std == 0] = 1
    eeg_data_normalized = (eeg_data - mean) / std

    # g. Pad samples to required length
    if eeg_data_normalized.shape[1] < REQUIRED_SAMPLES:
        padded_data = np.zeros((config['n_chans'], REQUIRED_SAMPLES))
        padded_data[:, :eeg_data_normalized.shape[1]] = eeg_data_normalized
    else:
        padded_data = eeg_data_normalized[:, :REQUIRED_SAMPLES]
        
    return padded_data.astype(np.float32)


def process_mat_file(file_path, limit_per_class, encoder, device, config):
    """
    Loads a .mat file, extracts all relevant epochs, preprocesses them,
    and generates embeddings using the pre-loaded model.
    """
    print(f"\n--- Processing file: {os.path.basename(file_path)} ---")
    try:
        mat_data = scipy.io.loadmat(file_path)
        eeg_data_mat = mat_data['data'][0,0]['X'].T
        flash_events_mat = mat_data['data'][0,0]['flash']
        sfreq = 250
        ch_names = P300_CHANNELS
        ch_types = ['eeg'] * len(ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(eeg_data_mat, info, verbose=False)

        events_onsets = flash_events_mat[:, 0].flatten() / sfreq
        events_durations = np.zeros_like(events_onsets)
        events_descriptions = ['target' if hit == 1 else 'non_target' for hit in flash_events_mat[:, 3]]
        annotations = mne.Annotations(onset=events_onsets, duration=events_durations, description=events_descriptions)
        raw.set_annotations(annotations)
    except Exception as e:
        print(f"Error loading or processing MATLAB file {file_path}: {e}", file=sys.stderr)
        return [], []

    event_dict = {'target': 1, 'non_target': 2}
    events, event_id_map = mne.events_from_annotations(raw, event_id=event_dict, verbose=False)
    
    trials_to_process = []
    for label, event_code in event_id_map.items():
        onsets_samples = events[events[:, 2] == event_code][:, 0]
        start_times = onsets_samples / raw.info['sfreq']
        limited_times = start_times[:limit_per_class]
        for t in limited_times:
            trials_to_process.append({'label': label, 'start': t})
        print(f"Found {len(limited_times)} trials for '{label}' (limit was {limit_per_class}).")

    target_embeddings = []
    non_target_embeddings = []

    # Process all epochs from this file
    for trial in tqdm(trials_to_process, desc=f"Generating embeddings for {os.path.basename(file_path)}"):
        trial_raw = raw.copy()
        
        # Crop to the specific epoch window
        tmin = trial['start'] + EPOCH_WINDOW_P300[0]
        tmax = trial['start'] + EPOCH_WINDOW_P300[1]
        trial_raw.crop(tmin=tmin, tmax=tmax, include_tmax=False)
        trial_raw.pick_channels(P300_CHANNELS)
        
        # Preprocess the epoch data
        processed_eeg = process_single_epoch(trial_raw, config)
        
        # Get embedding from the model
        embedding = get_embedding(processed_eeg, encoder, device, config)
        
        if trial['label'] == 'target':
            target_embeddings.append(embedding)
        else:
            non_target_embeddings.append(embedding)
            
    return target_embeddings, non_target_embeddings

# ========================================================================================
# --- CLASSIFIER LOGIC (TRAIN & TEST) ---
# ========================================================================================

def train(args, encoder, device, config):
    """Train the classifier by generating and saving prototype embeddings."""
    print("\n=== STARTING TRAINING PHASE ===")
    train_subjects = [f'P300S{i:02d}.mat' for i in range(1, 7)]
    train_files = [os.path.join('P300_MAT', subject) for subject in train_subjects]
    print(f"Training with subjects: {', '.join(train_subjects)}")

    all_target_embeddings = []
    all_non_target_embeddings = []

    for file_path in train_files:
        target_embeds, non_target_embeds = process_mat_file(
            file_path, args.limit_per_class, encoder, device, config
        )
        all_target_embeddings.extend(target_embeds)
        all_non_target_embeddings.extend(non_target_embeds)

    if not all_target_embeddings or not all_non_target_embeddings:
        print("Error: No embeddings were collected. Cannot create prototypes.", file=sys.stderr)
        sys.exit(1)

    prototype_target = np.mean(all_target_embeddings, axis=0)
    prototype_non_target = np.mean(all_non_target_embeddings, axis=0)

    prototypes = {
        'target': prototype_target.tolist(),
        'non_target': prototype_non_target.tolist()
    }

    with open(PROTOTYPE_FILE, 'w') as f:
        json.dump(prototypes, f, indent=4)

    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Processed {len(all_target_embeddings)} target and {len(all_non_target_embeddings)} non-target events.")
    print(f"Prototypes saved to '{PROTOTYPE_FILE}'.")

def test(args, encoder, device, config):
    """Test the classifier using the saved prototypes."""
    print("\n=== STARTING TESTING PHASE ===")
    try:
        with open(PROTOTYPE_FILE, 'r') as f:
            prototypes_list = json.load(f)
        prototype_target = np.array(prototypes_list['target'])
        prototype_non_target = np.array(prototypes_list['non_target'])
        print(f"Successfully loaded prototypes from '{PROTOTYPE_FILE}'.")
    except FileNotFoundError:
        print(f"Error: Prototype file '{PROTOTYPE_FILE}' not found. Please run 'train' mode first.", file=sys.stderr)
        sys.exit(1)

    test_subjects = [f'P300S{i:02d}.mat' for i in range(7, 9)]
    test_files = [os.path.join('P300_MAT', subject) for subject in test_subjects]
    print(f"Testing with subjects: {', '.join(test_subjects)}")

    total_correct = 0
    total_tested = 0

    for file_path in test_files:
        target_embeds, non_target_embeds = process_mat_file(
            file_path, args.limit_per_class, encoder, device, config
        )

        for embed in target_embeds:
            dist_to_target = np.linalg.norm(embed - prototype_target)
            dist_to_non_target = np.linalg.norm(embed - prototype_non_target)
            if dist_to_target < dist_to_non_target:
                total_correct += 1
            total_tested += 1

        for embed in non_target_embeds:
            dist_to_target = np.linalg.norm(embed - prototype_target)
            dist_to_non_target = np.linalg.norm(embed - prototype_non_target)
            if dist_to_non_target < dist_to_target:
                total_correct += 1
            total_tested += 1

    if total_tested == 0:
        print("Warning: No test events were processed.", file=sys.stderr)
        return

    accuracy = (total_correct / total_tested) * 100
    print("\n=== TESTING COMPLETE ===")
    print(f"Correctly classified {total_correct} out of {total_tested} test events.")
    print(f"Accuracy: {accuracy:.2f}%")

# ========================================================================================
# --- MAIN ---
# ========================================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified P300 Classifier using Prototype Embeddings.")
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Mode of operation')

    parser_train = subparsers.add_parser('train', help='Generate and save prototype embeddings.')
    parser_train.add_argument('--limit_per_class', type=int, default=100, help="Max trials per class per file.")
    parser_train.set_defaults(func=train)

    parser_test = subparsers.add_parser('test', help='Test the classifier on new data.')
    parser_test.add_argument('--limit_per_class', type=int, default=100, help="Max trials per class per file.")
    parser_test.set_defaults(func=test)

    args = parser.parse_args()

    # --- Load model once, then run the selected mode ---
    encoder, device = setup_model_and_device(MODEL_CONFIG)
    args.func(args, encoder, device, MODEL_CONFIG)

if __name__ == '__main__':
    main()
