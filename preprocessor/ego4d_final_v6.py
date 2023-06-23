import os

import librosa
import numpy as np
from scipy.io import wavfile
from utils.auto_tqdm import tqdm
import pandas as pd
from text import _clean_text


def prepare_align(config):
    in_dir = config['path']['corpus_path']
    out_dir = config['path']['raw_path']
    sampling_rate = config['preprocessing']['audio']['sampling_rate']
    max_wav_value = config['preprocessing']['audio']['max_wav_value']
    cleaners = config['preprocessing']['text']['text_cleaners']
    # pretend ego4d is single-player for now
    speaker = 'Ego4D_final_v6'
    transcript_train_path = config['path']['transcript_train_path']
    transcript_val_path = config['path']['transcript_val_path']

    train_df = pd.read_csv(transcript_train_path)
    val_df = pd.read_csv(transcript_val_path)
    print('here')

    for idx, row in tqdm(train_df.iterrows()):
        uid = row['utterance_id']
        text = row['transcription']
        text = _clean_text(text, cleaners)
        wav_path = os.path.join(in_dir, "train", f"{uid}.wav")
        if os.path.exists(wav_path):
            os.makedirs(os.path.join(out_dir, 'train', speaker), exist_ok=True)
            wav, _ = librosa.load(wav_path, sr=None)
            # print(wav_path)
            # print(wav.shape)
            # print(_)
            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                os.path.join(out_dir, 'train', speaker, f"{uid}.wav"),
                sampling_rate,
                wav.astype(np.int16),
            )
            with open(os.path.join(out_dir, 'train', speaker, f"{uid}.lab"), "w",) as f1:
                f1.write(text)

    for idx, row in tqdm(val_df.iterrows()):
        uid = row['utterance_id']
        text = row['transcription']
        text = _clean_text(text, cleaners)
        wav_path = os.path.join(in_dir, "val", f"{uid}.wav")
        if os.path.exists(wav_path):
            os.makedirs(os.path.join(out_dir, 'val', speaker), exist_ok=True)
            wav, _ = librosa.load(wav_path, sr=None)
            # print(wav_path)
            # print(wav.shape)
            # print(_)
            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                os.path.join(out_dir, 'val', speaker, f"{uid}.wav"),
                sampling_rate,
                wav.astype(np.int16),
            )
            with open(os.path.join(out_dir, 'val', speaker, f"{uid}.lab"), "w",) as f1:
                f1.write(text)
    print('prepare_align() completed.')
