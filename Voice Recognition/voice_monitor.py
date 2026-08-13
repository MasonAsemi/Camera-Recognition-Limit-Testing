"""
Voice identity monitor: listens on the microphone and continuously reports
how closely the current speaker's voice matches your enrolled voice profile.

Everything runs locally -- microphone capture, feature extraction, and
comparison all happen on this machine. No audio or transcript is ever sent
anywhere; none of speech_recognition's cloud `recognize_*` methods are used,
only its local Microphone/Recognizer capture and phrase-boundary detection.
See voice_identity.py for the feature/comparison pipeline.

Usage:
    python voice_monitor.py --enroll        # record your voice, save profile.json
    python voice_monitor.py                 # run the live monitor (GUI window)
    python voice_monitor.py --headless       # run the live monitor (console only)

Each recognized phrase produces one "distance" reading (lower = more like the
enrolled voice). The display shows that reading plus a rolling mean and
rolling standard deviation over the last --window readings, so a single odd
phrase (cough, background noise) doesn't flip the verdict -- the same
debounce idea chair_presence.py uses for occupancy, applied to a continuous
similarity score instead of a binary detection.
"""

import argparse
import csv
import os
import queue
import statistics
import threading
import time
from collections import deque

import speech_recognition as sr

import voice_identity

PROFILE_PATH_DEFAULT = "voice_profile.json"
LOG_PATH_DEFAULT = "voice_log.csv"


def record_sample(recognizer, mic, prompt):
    print(prompt)
    input("  Press Enter, then speak a full sentence (stops automatically on silence)...")
    with mic as source:
        audio = recognizer.listen(source, timeout=10, phrase_time_limit=8)
    print("  captured.")
    return audio


def run_enrollment(profile_path, min_samples=6):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Calibrating for ambient noise (stay quiet for a second)...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.5)

    print(f"\nEnrollment: you'll record {min_samples} samples of your voice.\n"
          "Speak naturally, a normal sentence or two each time, in your normal tone.\n")

    raw_embeddings = []
    for i in range(min_samples):
        while True:
            audio = record_sample(recognizer, mic, f"\nSample {i + 1}/{min_samples}:")
            try:
                raw = voice_identity.VoiceIdentity.raw_embedding_from_audio(
                    audio.get_raw_data(), sample_rate=audio.sample_rate)
            except ValueError:
                print("  didn't catch enough audio, try again.")
                continue
            raw_embeddings.append(raw)
            break

    _, _, normalized = voice_identity.VoiceIdentity.fit_scaler(raw_embeddings)
    threshold = voice_identity.suggest_threshold(normalized)

    identity = voice_identity.VoiceIdentity(profile_path)
    identity.save_profile(raw_embeddings, threshold)
    print(f"\nSaved voice profile ({len(raw_embeddings)} samples) to {profile_path}")
    print(f"Suggested match threshold: {threshold:.3f} (override at runtime with --threshold)")


def listener_thread(identity, out_queue, stop_event, phrase_time_limit):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        print("Calibrating for ambient noise (stay quiet for a second)...")
        recognizer.adjust_for_ambient_noise(source, duration=1.5)
        print("Listening. Speak normally; Ctrl+C (or close the window) to stop.")

        while not stop_event.is_set():
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=phrase_time_limit)
            except sr.WaitTimeoutError:
                continue
            try:
                raw = identity.raw_embedding_from_audio(audio.get_raw_data(), sample_rate=audio.sample_rate)
                distance = identity.distance_to_profile(raw)
            except ValueError:
                continue
            out_queue.put(distance)


def run_monitor(args):
    identity = voice_identity.VoiceIdentity(args.profile)
    if identity.profile is None:
        print(f"No voice profile at {args.profile}. Run with --enroll first.")
        return

    threshold = args.threshold if args.threshold is not None else identity.threshold
    print(f"Loaded voice profile ({identity.profile['sample_count']} samples). "
          f"Match threshold: {threshold:.3f}")

    log_new = not os.path.exists(args.log)
    log_file = open(args.log, "a", newline="")
    logger = csv.writer(log_file)
    if log_new:
        logger.writerow(["timestamp", "distance", "rolling_mean", "rolling_stdev", "decision"])

    readings = deque(maxlen=args.window)
    result_queue = queue.Queue()
    stop_event = threading.Event()
    thread = threading.Thread(
        target=listener_thread, args=(identity, result_queue, stop_event, args.phrase_time_limit),
        daemon=True)
    thread.start()

    def handle_reading(distance):
        readings.append(distance)
        mean = statistics.fmean(readings)
        stdev = statistics.pstdev(readings) if len(readings) > 1 else 0.0
        ready = len(readings) >= args.window
        if not ready:
            decision = "EVALUATING"
        else:
            decision = "MATCH" if mean <= threshold else "NO MATCH"
        logger.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), f"{distance:.3f}",
                          f"{mean:.3f}", f"{stdev:.3f}", decision])
        log_file.flush()
        return distance, mean, stdev, decision, len(readings)

    try:
        if args.headless:
            while True:
                distance = result_queue.get()
                d, mean, stdev, decision, count = handle_reading(distance)
                print(f"[{time.strftime('%H:%M:%S')}] distance={d:.3f}  "
                      f"rolling_mean={mean:.3f}  rolling_stdev={stdev:.3f}  -> {decision}")
        else:
            run_gui(result_queue, handle_reading, threshold, stop_event)
    except KeyboardInterrupt:
        print("Stopping (Ctrl+C).")
    finally:
        stop_event.set()
        log_file.close()


def run_gui(result_queue, handle_reading, threshold, stop_event):
    import tkinter as tk

    root = tk.Tk()
    root.title("Voice Identity Monitor")
    root.configure(bg="#1e1e1e")
    root.geometry("480x320")

    status_var = tk.StringVar(value="LISTENING...")
    status_label = tk.Label(root, textvariable=status_var, font=("Segoe UI", 28, "bold"),
                             fg="#cccccc", bg="#1e1e1e")
    status_label.pack(pady=(24, 12))

    reading_font = ("Consolas", 14)
    lines = {}
    for key, text in [
        ("distance", "Distance (this phrase):"),
        ("mean", "Rolling mean:"),
        ("stdev", "Rolling std dev:"),
        ("threshold", "Match threshold:"),
        ("count", "Readings in window:"),
    ]:
        row = tk.Frame(root, bg="#1e1e1e")
        row.pack(fill="x", padx=30, pady=3)
        tk.Label(row, text=text, font=reading_font, fg="#999999", bg="#1e1e1e", anchor="w",
                 width=22).pack(side="left")
        val = tk.StringVar(value="--")
        tk.Label(row, textvariable=val, font=reading_font, fg="#ffffff", bg="#1e1e1e",
                 anchor="w").pack(side="left")
        lines[key] = val

    lines["threshold"].set(f"{threshold:.3f}")

    canvas = tk.Canvas(root, width=420, height=24, bg="#333333", highlightthickness=0)
    canvas.pack(pady=(14, 0))
    bar = canvas.create_rectangle(0, 0, 0, 24, fill="#888888", width=0)

    def poll():
        try:
            while True:
                distance = result_queue.get_nowait()
                d, mean, stdev, decision, count = handle_reading(distance)
                lines["distance"].set(f"{d:.3f}")
                lines["mean"].set(f"{mean:.3f}")
                lines["stdev"].set(f"{stdev:.3f}")
                lines["count"].set(str(count))

                if decision == "MATCH":
                    status_var.set("MATCH")
                    status_label.configure(fg="#3ddc84")
                    bar_color = "#3ddc84"
                elif decision == "NO MATCH":
                    status_var.set("NOT A MATCH")
                    status_label.configure(fg="#ff5c5c")
                    bar_color = "#ff5c5c"
                else:
                    status_var.set("EVALUATING...")
                    status_label.configure(fg="#ffcc66")
                    bar_color = "#ffcc66"

                # bar fill scaled to 2x threshold = full width
                frac = max(0.0, min(1.0, mean / (2 * threshold))) if threshold else 0.0
                canvas.coords(bar, 0, 0, 420 * frac, 24)
                canvas.itemconfig(bar, fill=bar_color)
        except queue.Empty:
            pass
        root.after(150, poll)

    def on_close():
        stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(150, poll)
    root.mainloop()


def parse_args():
    p = argparse.ArgumentParser(description="Local voice identity monitor.")
    p.add_argument("--profile", default=PROFILE_PATH_DEFAULT, help="path to voice profile JSON")
    p.add_argument("--log", default=LOG_PATH_DEFAULT, help="path to CSV reading log")
    p.add_argument("--enroll", action="store_true", help="record voice samples and save a profile, then exit")
    p.add_argument("--samples", type=int, default=6, help="number of enrollment samples to record (default 6)")
    p.add_argument("--threshold", type=float, default=None,
                    help="override the profile's saved match threshold")
    p.add_argument("--window", type=int, default=4,
                    help="number of recent phrase readings averaged for the rolling mean/stdev "
                         "and MATCH/NO MATCH decision (default 4)")
    p.add_argument("--phrase-time-limit", dest="phrase_time_limit", type=float, default=8.0,
                    help="max seconds captured per phrase (default 8)")
    p.add_argument("--headless", action="store_true", help="console output only, no GUI window")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.enroll:
        run_enrollment(args.profile, min_samples=args.samples)
    else:
        run_monitor(args)
