import os
import random
import json
import glob

import tgt
import librosa
import numpy as np
import pandas as pd
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from utils.auto_tqdm import tqdm

import audio as Audio
import torch


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        try:
            self.splits = config["preprocessing"]["splits"]
        except Exception as e:
            print(e)
        try:
            self.is_using_video_info = config["preprocessing"]["video_info"]["using_video_info"]
        except Exception as e:
            print(e)
            self.is_using_video_info = False
            print('Not using video info.')

        if self.is_using_video_info:
            self.video_dir = config["path"]["video_embedding_path"]
            video_count = len(glob.glob(f"{self.video_dir}/*.pt"))
            print(f"Video count: {video_count}")
        self.drop_audio_shorter_than = config["preprocessing"]["audio"]["drop_audio_shorter_than"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )


    def build_video_trainval_seqs(self):
        print("build_video_trainval_seqs()")
        df_train_path = self.config['path']['transcript_train_path']
        df_val_path = self.config['path']['transcript_val_path']
        df_paths = {
            'train': df_train_path,
            'val': df_val_path,
        }
        for split_name in self.splits:
            os.makedirs(
                (os.path.join(self.out_dir, "video_embedding", split_name)), exist_ok=True)
            df = pd.read_csv(df_paths[split_name])
            # print(df.shape)
            # print(df.columns)
            for idx, row in tqdm(df.iterrows()):
                uid = row['utterance_id']
                vid = row['video_id']
                start_time = row['video_start_time']
                end_time = row['video_end_time']
                try:
                    video_embeddings = torch.load(os.path.join(self.video_dir, f"{vid}.pt")).cpu()

                except Exception as e:
                    print(e)
                    continue
                # print(f"video_embeddings.shape: {video_embeddings.shape}")
                sample_idxs = self._get_sample_idxs(video_embeddings.shape[0], start_time, end_time)
                # print(start_time, end_time)
                # print(sample_idxs)

                sample_seq = video_embeddings[sample_idxs]
                sample_seq = sample_seq.numpy()
                
                # print(sample_seq.shape)
                np.save(os.path.join(self.out_dir, "video_embedding",
                split_name, f"{uid}.npy"), sample_seq)
                


                # break

            
    def _get_sample_idxs(self, video_seq_len, start_time, end_time):
        VIDEO_FPS = self.config["preprocessing"]["video_info"]["video_fps"]
        # FEATURE_WINDOW = self.config["preprocessing"]["video_info"]["feature_window_size"]
        FEATURE_STRIDE = self.config["preprocessing"]["video_info"]["feature_stride"]

        start_idx = start_time*VIDEO_FPS//FEATURE_STRIDE
        end_idx = end_time*VIDEO_FPS//FEATURE_STRIDE
        history_stride = self.config["preprocessing"]["video_info"]["history_stride"]
        history_len = self.config["preprocessing"]["video_info"]["n_history_vecs"]
        content_idxs = torch.arange(start_idx, end_idx + 1, dtype=torch.int64)
        history_idxs = self._create_history_with_stride(start_idx, (-1)*history_stride, history_len)
        history_idxs[history_idxs < 0] = 0
        video_seq_len
        history_idxs[history_idxs >= video_seq_len] = video_seq_len - 1


        return torch.cat([history_idxs, content_idxs])

    def _create_history_with_stride(self, start, stride, seq_len):
        # exclude the starting point
        start += seq_len * stride
        tensor = torch.arange(start, start-seq_len*stride, -stride, dtype=torch.int64)
        return tensor



    def build_from_trainval_path(self):
        if self.is_using_video_info:
            print("Preprocessing video embeddings...")
            self.build_video_trainval_seqs()
        # print('Stop here.')
        # return
        for split_name in self.splits:
            os.makedirs(
                (os.path.join(self.out_dir, "mel", split_name)), exist_ok=True)
            os.makedirs(
                (os.path.join(self.out_dir, "pitch", split_name)), exist_ok=True)
            os.makedirs(
                (os.path.join(self.out_dir, "energy", split_name)), exist_ok=True)
            os.makedirs(
                (os.path.join(self.out_dir, "duration", split_name)), exist_ok=True)

            print(f"Processing {split_name} Data ...")
            out = list()
            n_frames = 0
            pitch_scaler = StandardScaler()
            energy_scaler = StandardScaler()

            # Compute pitch, energy, duration, and mel-spectrogram
            speakers = {}
            for i, speaker in enumerate(tqdm(os.listdir(os.path.join(self.in_dir, split_name)))):
                # {'LJSpeech': 0}
                speakers[speaker] = i
                for wav_name in tqdm(os.listdir(os.path.join(self.in_dir, split_name, speaker))):
                    if ".wav" not in wav_name:
                        continue
                    basename = wav_name.split(".")[0]
                    tg_path = os.path.join(
                        self.out_dir, "TextGrid", split_name, speaker, "{}.TextGrid".format(
                            basename)
                    )
                    if os.path.exists(tg_path):
                        ret = self.process_trainval_utterance(
                            split_name, speaker, basename)
                        if ret is None:
                            continue
                        else:
                            info, pitch, energy, n = ret
                        out.append(info)
                    if len(pitch) > 0:
                        pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                    if len(energy) > 0:
                        energy_scaler.partial_fit(energy.reshape((-1, 1)))

                    n_frames += n

            print("Computing statistic quantities ...")
            # Perform normalization if necessary
            if self.pitch_normalization:
                pitch_mean = pitch_scaler.mean_[0]
                pitch_std = pitch_scaler.scale_[0]
            else:
                # A numerical trick to avoid normalization...
                pitch_mean = 0
                pitch_std = 1
            if self.energy_normalization:
                energy_mean = energy_scaler.mean_[0]
                energy_std = energy_scaler.scale_[0]
            else:
                energy_mean = 0
                energy_std = 1

            pitch_min, pitch_max = self.normalize(
                os.path.join(self.out_dir, "pitch",
                             split_name), pitch_mean, pitch_std
            )
            energy_min, energy_max = self.normalize(
                os.path.join(self.out_dir, "energy",
                             split_name), energy_mean, energy_std
            )
            # Save files
            # 这里没有创建为train和val创建单独的json，
            # 而是混起来存的
            with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
                f.write(json.dumps(speakers))

            with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
                stats = {
                    "pitch": [
                        float(pitch_min),
                        float(pitch_max),
                        float(pitch_mean),
                        float(pitch_std),
                    ],
                    "energy": [
                        float(energy_min),
                        float(energy_max),
                        float(energy_mean),
                        float(energy_std),
                    ],
                }
                f.write(json.dumps(stats))

            print(
                "Total time: {} hours".format(
                    n_frames * self.hop_length / self.sampling_rate / 3600
                )
            )
            with open(os.path.join(self.out_dir, f"{split_name}.txt"), "w", encoding="utf-8") as f:
                for m in out:
                    f.write(m + "\n")

        # return out

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            # {'LJSpeech': 0}
            speakers[speaker] = i
            for wav_name in tqdm(os.listdir(os.path.join(self.in_dir, speaker))):
                if ".wav" not in wav_name:
                    continue
                print(wav_name)
                basename = wav_name.split(".")[0]
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(
                        basename)
                )
                print(tg_path)
                if os.path.exists(tg_path):

                    ret = self.process_utterance(speaker, basename)
                    if ret is None:
                        continue
                    else:
                        info, pitch, energy, n = ret
                    out.append(info)

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))

                n_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size:]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker,
                                "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker,
                                 "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path, self.sampling_rate)
        wav = wav[
            int(self.sampling_rate * start): int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64),
                             pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        ### new
        ### remove the all-0 pitch filter
        ### cancelled
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos: pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos: pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def process_trainval_utterance(self, split_name, speaker, basename):
        wav_path = os.path.join(self.in_dir, split_name,
                                speaker, "{}.wav".format(basename))
        text_path = os.path.join(
            self.in_dir, split_name, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", split_name, speaker, "{}.TextGrid".format(
                basename)
        )

        log_path = os.path.join(self.out_dir, "preprocessing_log.txt")
        log_file = open(log_path, "a+")

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            # print(f'Skipped because `alignment start>=end`: {tg_path}')
            log_file.write(
                f'Skipped because `alignment start>=end`: {tg_path}\n')
            return None
        # Read and trim wav files
        wav, _ = librosa.load(wav_path, self.sampling_rate)
        # print('sample_rate here: ', _)
        wav = wav[
            int(self.sampling_rate * start): int(self.sampling_rate * end)
        ].astype(np.float32)
        # filtered out audio with speech duration
        # shorter than 0.1s
        # print(np.count_nonzero(np.isnan(wav)))
        if wav.shape[0] <= self.drop_audio_shorter_than * self.sampling_rate:
            log_file.write(
                f'Skipped because `wav[start:end] is too short`: {tg_path}\n')
            # print(f'Skipped because `wav[start:end].shape == 0`: {tg_path}\n')
            return None
        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        # try:
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        # except:
        #     print('wav.shape: ', wav.shape)
        pitch = pw.stonemask(wav.astype(np.float64),
                             pitch, t, self.sampling_rate)
        pitch = pitch[: sum(duration)]
        ### 
        if len(pitch) == 0:
            # print(f'Skipped because `len(pitch) == 0`: {wav_path}')
            return None
        if np.sum(pitch != 0) <= 1:
            # print(f'Skipped because `np.sum(pitch != 0) <= 1`: {wav_path}')
            log_file.write(
                f'Skipped because `np.sum(pitch != 0) <= 1`: {wav_path}\n')
            return None
        log_file.close()

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if i > len(pitch) - 1:
                    break
                if d > 0:
                    ###
                    temp = pitch[pos: pos + d]
                    if len(temp) == 0:
                        pitch[i] = 0
                    else:
                        pitch[i] = np.mean(temp)
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if i > len(energy) - 1:
                    break
                if d > 0:
                    ###
                    temp = energy[pos: pos + d]
                    if len(temp) == 0:
                        energy[i] = 0
                    else:
                        energy[i] = np.mean(temp)
  
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration",
                split_name, dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch",
                split_name, pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy",
                split_name, energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", split_name, mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
