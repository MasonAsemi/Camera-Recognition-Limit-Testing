"""
Voice identity verification: is the voice currently on the microphone yours?

Pipeline: MFCC ("Mel-Frequency Cepstral Coefficients") acoustic features --
the standard hand-engineered front end used throughout speech and speaker
processing -- pooled (mean + std over time) into a fixed-length per-utterance
fingerprint, then compared to enrolled fingerprints with per-dimension
z-normalized Euclidean distance.

This is deliberately the classical statistical approach, not a trained
neural speaker-embedding model (the voice equivalent of dlib's ResNet in
face_identity.py). A neural embedding would need a pretrained model
download and is overkill for "basic scripts"; MFCC pooling is a real,
well-established signal-processing technique and needs nothing but local
computation, so it fits the "everything stays on this machine" requirement
directly.

Audio capture (microphone open/close, detecting when a phrase starts and
stops) is handled by the `speech_recognition` package's Recognizer/Microphone
classes. Only its local capture/VAD pieces are used here -- none of its
`recognize_*` methods are called, so no audio or transcript ever leaves
the machine.

Everything here -- capture, feature extraction, comparison, storage -- runs
locally. No network calls, no cloud speech APIs.
"""

import json
import os
import time

import numpy as np

PROFILE_FORMAT = "mfcc_meanstd_v1"

N_MFCC = 20  # cepstral coefficients kept; mean+std pooling doubles this to embedding size


class VoiceIdentity:
    def __init__(self, profile_path):
        self.profile_path = profile_path
        self.profile = self._load_profile()

    def _load_profile(self):
        if not os.path.exists(self.profile_path):
            return None
        with open(self.profile_path) as f:
            data = json.load(f)
        if data.get("format") != PROFILE_FORMAT:
            print(f"Profile at {self.profile_path} is from an incompatible format. "
                  "Run enrollment again to create a new one.")
            return None
        data["embeddings"] = [np.array(e) for e in data["embeddings"]]
        data["scaler_mean"] = np.array(data["scaler_mean"])
        data["scaler_std"] = np.array(data["scaler_std"])
        return data

    @staticmethod
    def fit_scaler(raw_embeddings):
        """Per-dimension mean/std across enrollment samples, plus each sample normalized
        against it. Shared by save_profile and the enrollment script's threshold suggestion
        so both operate on the exact same normalized vectors."""
        stacked = np.stack(raw_embeddings)
        scaler_mean = stacked.mean(axis=0)
        scaler_std = stacked.std(axis=0)
        scaler_std[scaler_std < 1e-6] = 1e-6  # guard divide-by-zero for near-constant dims
        normalized = [(e - scaler_mean) / scaler_std for e in raw_embeddings]
        return scaler_mean, scaler_std, normalized

    def save_profile(self, raw_embeddings, threshold):
        """raw_embeddings: list of un-normalized MFCC mean+std vectors from enrollment samples."""
        scaler_mean, scaler_std, normalized = self.fit_scaler(raw_embeddings)

        profile = {
            "format": PROFILE_FORMAT,
            "embeddings": [e.tolist() for e in normalized],
            "scaler_mean": scaler_mean.tolist(),
            "scaler_std": scaler_std.tolist(),
            "threshold": threshold,
            "sample_count": len(raw_embeddings),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.profile_path, "w") as f:
            json.dump(profile, f)

        profile["embeddings"] = normalized
        profile["scaler_mean"] = scaler_mean
        profile["scaler_std"] = scaler_std
        self.profile = profile
        return profile

    @staticmethod
    def raw_embedding_from_audio(audio_data, sample_rate=None):
        """audio_data: raw PCM16 mono bytes (e.g. sr.AudioData.get_raw_data()).
        Returns an un-normalized MFCC mean+std feature vector (2*N_MFCC dims)."""
        import librosa

        sr_rate = sample_rate
        y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        if y.size == 0:
            raise ValueError("empty audio buffer")
        mfcc = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=N_MFCC)
        return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

    def normalize(self, raw_embedding):
        return (raw_embedding - self.profile["scaler_mean"]) / self.profile["scaler_std"]

    def distance_to_profile(self, raw_embedding):
        """Euclidean distance (in normalized feature space) to the closest enrolled
        sample. Lower = more likely the same voice."""
        if self.profile is None:
            return None
        normalized = self.normalize(raw_embedding)
        return min(float(np.linalg.norm(normalized - e)) for e in self.profile["embeddings"])

    @property
    def threshold(self):
        return self.profile["threshold"] if self.profile else None


def suggest_threshold(normalized_embeddings):
    """Heuristic cutoff from the enrollment samples' own spread: mean + 2*std of
    all pairwise distances between enrolled samples. Not a validated universal
    constant (there's no equivalent of dlib's LFW calibration here) -- it's
    calibrated per-user from your own enrollment recordings. Tune with
    --threshold if it's too strict/loose in practice."""
    n = len(normalized_embeddings)
    pairwise = []
    for i in range(n):
        for j in range(i + 1, n):
            pairwise.append(float(np.linalg.norm(normalized_embeddings[i] - normalized_embeddings[j])))
    if not pairwise:
        return 3.0  # single-sample enrollment: no pairwise spread to learn from, use a fallback
    mean = float(np.mean(pairwise))
    std = float(np.std(pairwise))
    return mean + 2 * std
