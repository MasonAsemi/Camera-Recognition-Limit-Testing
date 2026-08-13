"""
Identity verification layered on top of chair occupancy detection.

Uses dlib's face recognition pipeline: a 5-point landmark predictor for
alignment, then a ResNet trained with triplet loss specifically to pull
same-person face embeddings together and push different-person embeddings
apart -- 128 learned dimensions, the actual technology behind most
production face recognition.

This module previously computed hand-picked geometric ratios (eye width,
nose length, etc.) from MediaPipe's face mesh. In testing that approach
mismatched a different person as a match: coarse proportions like "mouth
width / eye span" turn out to be similar across a lot of faces. That's the
same limitation that pushed the field toward learned embeddings in the
first place, so this module was rewritten to use one.

Model files (not in git, see chair_presence.py docstring for provenance):
  models/shape_predictor_5_face_landmarks.dat
  models/dlib_face_recognition_resnet_model_v1.dat
Both from https://github.com/davisking/dlib-models (dlib's official model repo).
"""

import json
import os
import time

import cv2
import dlib
import numpy as np

SHAPE_PREDICTOR_PATH_DEFAULT = os.path.join("models", "shape_predictor_5_face_landmarks.dat")
FACE_REC_MODEL_PATH_DEFAULT = os.path.join("models", "dlib_face_recognition_resnet_model_v1.dat")

# dlib/face_recognition's long-established default: embeddings from the same
# person are typically <0.6 apart (Euclidean, 128-d), calibrated against LFW.
DEFAULT_THRESHOLD = 0.6

PROFILE_FORMAT = "dlib_embedding_v1"


class FaceIdentity:
    def __init__(self, profile_path, shape_predictor_path=SHAPE_PREDICTOR_PATH_DEFAULT,
                 face_rec_model_path=FACE_REC_MODEL_PATH_DEFAULT):
        for p in (shape_predictor_path, face_rec_model_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required model file not found: {p}")
        self.profile_path = profile_path
        self._shape_predictor = dlib.shape_predictor(shape_predictor_path)
        self._face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
        self.profile = self._load_profile()

    def close(self):
        pass  # dlib objects need no explicit teardown

    def _load_profile(self):
        if not os.path.exists(self.profile_path):
            return None
        with open(self.profile_path) as f:
            data = json.load(f)
        if data.get("format") != PROFILE_FORMAT:
            print(f"Profile at {self.profile_path} is from the old geometric-ratio approach "
                  "(retired -- it didn't discriminate between people well enough). "
                  "Run --enroll-face again to create a new one.")
            return None
        data["embeddings"] = [np.array(e) for e in data["embeddings"]]
        return data

    def save_profile(self, embeddings):
        profile = {
            "format": PROFILE_FORMAT,
            "embeddings": [e.tolist() for e in embeddings],
            "sample_count": len(embeddings),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.profile_path, "w") as f:
            json.dump(profile, f)
        profile["embeddings"] = embeddings
        self.profile = profile
        return profile

    def compute_embedding(self, frame_bgr, box):
        """box is (x, y, w, h), e.g. from a Haar detection. Returns a 128-d np.array."""
        x, y, w, h = box
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        shape = self._shape_predictor(rgb, rect)
        descriptor = self._face_rec_model.compute_face_descriptor(rgb, shape)
        return np.array(descriptor)

    def distance_to_profile(self, embedding):
        """Euclidean distance to the closest enrolled sample. Lower = more likely a match."""
        if self.profile is None:
            return None
        return min(float(np.linalg.norm(embedding - e)) for e in self.profile["embeddings"])


def run_enrollment(cap, profile_path, shape_predictor_path=SHAPE_PREDICTOR_PATH_DEFAULT,
                    face_rec_model_path=FACE_REC_MODEL_PATH_DEFAULT, min_samples=8):
    identity = FaceIdentity(profile_path, shape_predictor_path, face_rec_model_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    embeddings = []
    window = "Enroll face - 'c' capture sample, 's' save profile, 'q' cancel"
    cv2.namedWindow(window)
    print(f"Enrollment: look at the camera. Press 'c' to capture a sample (try a few "
          f"different angles/expressions), 's' to save once you have at least {min_samples}, "
          "'q' to cancel.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            box = max(faces, key=lambda f: f[2] * f[3]) if len(faces) else None

            display = frame.copy()
            found_txt = "face detected" if box is not None else "no face"
            color = (0, 255, 0) if box is not None else (0, 0, 255)
            if box is not None:
                x, y, w, h = box
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display, f"samples: {len(embeddings)}  ({found_txt})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                if box is not None:
                    embeddings.append(identity.compute_embedding(frame, box))
                    print(f"captured sample {len(embeddings)}")
                else:
                    print("no face detected, try again")
            elif key == ord("s"):
                if len(embeddings) >= min_samples:
                    identity.save_profile(embeddings)
                    print(f"Saved profile ({len(embeddings)} samples) to {profile_path}")
                    return True
                print(f"need at least {min_samples} samples, have {len(embeddings)}")
            elif key in (ord("q"), 27):
                print("Enrollment cancelled.")
                return False
    finally:
        cv2.destroyWindow(window)
