"""
Chair occupancy detection: watches a webcam feed and reports whether someone
is sitting in a user-calibrated zone (e.g. your desk chair).

Pipeline: Haar cascade face detection (bundled with OpenCV, no model download)
+ a manually calibrated region-of-interest (ROI) + a debounce/hysteresis state
machine so brief flickers (you leaning out of frame, a missed detection frame)
don't cause false EMPTY/OCCUPIED toggles.

Usage:
    python chair_presence.py --calibrate     # draw the chair zone, then run
    python chair_presence.py                 # run using saved config.json
    python chair_presence.py --camera 1      # use a different camera index

Tunable parameters (see --help): detection sensitivity (scale-factor,
min-neighbors, min-size) and debounce timing (enter-frames, exit-frames).

By default (Windows only), the monitor is powered off when the zone goes
EMPTY and woken back up when it goes OCCUPIED. Disable with --no-monitor-control.

Optional identity layer: run --enroll-face once to record a dlib face
embedding of you, then the main loop will tag whoever is in the zone as ME
or UNKNOWN. By default, once a profile is enrolled, the monitor only wakes
for a recognized match -- an unrecognized person sitting in the zone is
logged (ID: UNKNOWN) but the screen stays off. Pass --no-require-identity
to go back to waking for any presence. See face_identity.py.
"""

import argparse
import csv
import ctypes
import json
import logging
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import cv2

import face_identity

WM_SYSCOMMAND = 0x0112
SC_MONITORPOWER = 0xF170
HWND_BROADCAST = 0xFFFF
VK_SHIFT = 0x10
KEYEVENTF_KEYUP = 0x0002
MOUSEEVENTF_MOVE = 0x0001

log = logging.getLogger("chair_presence")


def _target_hwnd():
    """Prefer our own console window over HWND_BROADCAST.

    SC_MONITORPOWER is a system-wide power command: any top-level window's
    default WndProc will trigger it, so we don't need to broadcast. Broadcasting
    sends the message to *every* top-level window on the system and, if any of
    them is slow to pump messages (a stuck dialog, a busy background app),
    SendMessage blocks the caller until it responds -- which looks exactly like
    the whole script freezing at the moment the monitor turns off.
    """
    hwnd = ctypes.windll.kernel32.GetConsoleWindow()
    return hwnd if hwnd else HWND_BROADCAST


def turn_monitor_off():
    """Windows-only: request the display go into the off power state (DPMS off)."""
    hwnd = _target_hwnd()
    t0 = time.perf_counter()
    log.debug("turn_monitor_off: PostMessageW(hwnd=%s) start", hwnd)
    # PostMessageW queues the message and returns immediately (unlike
    # SendMessageW, which blocks until the target window handles it).
    ok = ctypes.windll.user32.PostMessageW(hwnd, WM_SYSCOMMAND, SC_MONITORPOWER, 2)
    log.debug("turn_monitor_off: returned ok=%s in %.1fms", ok, (time.perf_counter() - t0) * 1000)


def wake_monitor():
    """Windows-only: inject harmless input to trigger the OS's normal wake path.

    SC_MONITORPOWER's "on" state (-1) is unreliable across display drivers;
    a synthetic keypress + mouse nudge wakes the display the same way real
    input does, and is honored much more consistently. keybd_event/mouse_event
    queue input asynchronously, so this does not block.
    """
    t0 = time.perf_counter()
    log.debug("wake_monitor: start")
    ctypes.windll.user32.keybd_event(VK_SHIFT, 0, 0, 0)
    ctypes.windll.user32.keybd_event(VK_SHIFT, 0, KEYEVENTF_KEYUP, 0)
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, 1, 0, 0, 0)
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, -1, 0, 0, 0)
    log.debug("wake_monitor: returned in %.1fms", (time.perf_counter() - t0) * 1000)

CONFIG_PATH_DEFAULT = "config.json"
LOG_PATH_DEFAULT = "presence_log.csv"
DEBUG_LOG_PATH_DEFAULT = "debug.log"
FACE_PROFILE_PATH_DEFAULT = "face_profile.json"
IDENTITY_LOG_PATH_DEFAULT = "identity_log.csv"

DEFAULT_CONFIG = {
    "roi": None,  # [x, y, w, h] in pixels, set via --calibrate
    "scale_factor": 1.1,
    "min_neighbors": 5,
    "min_size": 60,
    "enter_frames": 8,   # consecutive positive frames to declare OCCUPIED
    "exit_frames": 15,   # consecutive negative frames to declare EMPTY
}


def load_config(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            cfg = DEFAULT_CONFIG.copy()
            cfg.update(json.load(f))
            return cfg
    return DEFAULT_CONFIG.copy()


def save_config(path, cfg):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def open_camera(index, width=None, height=None):
    # CAP_DSHOW opens faster and more reliably on Windows than the default backend.
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {index}. Is it connected and not in use "
            "by another application?"
        )
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}"
          + (f" (requested {width}x{height}, camera may not support it exactly)"
             if width or height else " (not constrained -- camera's own default)"))
    return cap


def calibrate_roi(cap, config_path, cfg):
    """Let the user drag a rectangle over a live-ish preview to mark the chair zone."""
    state = {"drawing": False, "start": None, "rect": None}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"] = (x, y)
            state["rect"] = None
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            x0, y0 = state["start"]
            state["rect"] = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            if state["start"] is not None:
                x0, y0 = state["start"]
                state["rect"] = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))

    window = "Calibrate chair zone - drag a box, 's' save, 'r' reset, 'q' cancel"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    print("Calibration: drag a rectangle over the chair/desk area you want to watch.")
    print("Press 's' to save, 'r' to redo, 'q'/ESC to cancel.")

    saved = False
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        display = frame.copy()
        if state["rect"]:
            x, y, w, h = state["rect"]
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and state["rect"] and state["rect"][2] > 5 and state["rect"][3] > 5:
            cfg["roi"] = list(state["rect"])
            save_config(config_path, cfg)
            print(f"Saved ROI {cfg['roi']} to {config_path}")
            saved = True
            break
        elif key == ord("r"):
            state["rect"] = None
        elif key in (ord("q"), 27):
            break

    cv2.destroyWindow(window)
    return saved


def rect_overlap_ratio(box, roi):
    """Fraction of the detected face box's area that falls inside the ROI."""
    bx, by, bw, bh = box
    rx, ry, rw, rh = roi
    ix1, iy1 = max(bx, rx), max(by, ry)
    ix2, iy2 = min(bx + bw, rx + rw), min(by + bh, ry + rh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    box_area = max(1, bw * bh)
    return inter / box_area


_worker_identity = None  # set once per worker process by _init_identity_worker


def _init_identity_worker(profile_path, shape_predictor_path, face_rec_model_path):
    """ProcessPoolExecutor initializer: loads dlib's models once in the worker process.

    dlib's compute_face_descriptor holds Python's GIL for its entire ~350ms
    call (confirmed by direct measurement), so a background *thread* gives no
    real concurrency -- the whole interpreter, including the main loop's
    cv2.imshow, still stalls for that long. A separate process has its own
    GIL, so this is a process, not a thread. dlib's compiled model objects
    aren't picklable, so each worker loads its own copy here rather than
    receiving one from the parent.
    """
    global _worker_identity
    _worker_identity = face_identity.FaceIdentity(profile_path, shape_predictor_path, face_rec_model_path)


def _identity_distance_job(frame, box):
    """Runs in the worker process spawned by _init_identity_worker."""
    embedding = _worker_identity.compute_embedding(frame, box)
    return _worker_identity.distance_to_profile(embedding)


def run(args):
    if args.monitor_control and os.name != "nt":
        print("Monitor control is Windows-only; disabling it for this run.")
        args.monitor_control = False

    file_handler = logging.FileHandler(args.debug_log)
    file_handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s", "%H:%M:%S"))
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.setLevel(logging.WARNING)
    log.setLevel(logging.DEBUG)
    log.addHandler(file_handler)
    log.addHandler(console_handler)
    log.info("=== run start ===")

    cfg = load_config(args.config)

    # CLI flags override whatever is in config.json for this run.
    for key in ("scale_factor", "min_neighbors", "min_size", "enter_frames", "exit_frames"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    cap = open_camera(args.camera, args.width, args.height)

    if args.enroll_face:
        face_identity.run_enrollment(cap, args.face_profile, args.shape_predictor_model, args.face_rec_model)
        cap.release()
        return

    if args.calibrate or cfg.get("roi") is None:
        if not calibrate_roi(cap, args.config, cfg):
            print("No ROI saved, exiting.")
            cap.release()
            return

    roi = cfg["roi"]
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    identity = None
    if os.path.exists(args.face_profile):
        try:
            candidate = face_identity.FaceIdentity(
                args.face_profile, args.shape_predictor_model, args.face_rec_model)
        except FileNotFoundError as e:
            print(f"Identity recognition disabled: {e}")
        else:
            if candidate.profile is not None:
                identity = candidate
                print(f"Loaded face profile from {args.face_profile} "
                      f"({identity.profile['sample_count']} samples).")
            # else: FaceIdentity already printed why it's being skipped (e.g. stale format)
    else:
        print(f"No face profile at {args.face_profile}; identity recognition disabled. "
              "Run --enroll-face to create one.")

    if args.require_identity and identity is None:
        print("No face profile is loaded, so identity-gated wake can't be enforced; "
              "falling back to waking for any presence. Run --enroll-face to enable it.")
        args.require_identity = False

    identity_scores = deque(maxlen=args.identity_window)
    is_me = None  # None = unknown/not yet evaluated, True/False once smoothed
    avg_distance = None  # persists across frames; only updated when a background result lands
    identity_future = None  # in-flight background embedding job, or None if idle
    identity_executor = None
    if identity is not None:
        identity_executor = ProcessPoolExecutor(
            max_workers=1, initializer=_init_identity_worker,
            initargs=(args.face_profile, args.shape_predictor_model, args.face_rec_model))

    print("Effective parameters:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    log_new = not os.path.exists(args.log)
    log_file = open(args.log, "a", newline="")
    logger = csv.writer(log_file)
    if log_new:
        logger.writerow(["timestamp", "event", "prev_state_duration_sec"])

    identity_log_file = None
    identity_logger = None
    if identity is not None:
        id_log_new = not os.path.exists(args.identity_log)
        identity_log_file = open(args.identity_log, "a", newline="")
        identity_logger = csv.writer(identity_log_file)
        if id_log_new:
            identity_logger.writerow(["timestamp", "event", "avg_deviation"])

    occupied = False
    in_streak = 0
    out_streak = 0
    last_transition = time.time()
    prev_frame_time = time.time()
    frame_num = 0
    faces = ()  # carried over between detection frames when --detect-interval > 1

    SLOW_STEP_SEC = 0.25  # log a warning if any single step takes longer than this

    def timed(label, t_prev):
        t_now = time.perf_counter()
        dt = t_now - t_prev
        if dt > SLOW_STEP_SEC:
            log.warning("slow step '%s': %.0fms", label, dt * 1000)
        else:
            log.debug("step '%s': %.1fms", label, dt * 1000)
        return t_now

    print("Running. Press 'q' to quit.")
    try:
        while True:
            t = time.perf_counter()
            frame_num += 1

            ok, frame = cap.read()
            if not ok:
                print("Camera read failed, stopping.")
                break
            t = timed("camera.read", t)

            if frame_num % args.detect_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=cfg["scale_factor"],
                    minNeighbors=cfg["min_neighbors"],
                    minSize=(cfg["min_size"], cfg["min_size"]),
                )
                t = timed("detect", t)

            zone_faces = [f for f in faces if rect_overlap_ratio(tuple(f), roi) > 0.5]
            face_in_zone = len(zone_faces) > 0
            identity_box = max(zone_faces, key=lambda f: f[2] * f[3]) if zone_faces else None

            if face_in_zone:
                in_streak += 1
                out_streak = 0
            else:
                out_streak += 1
                in_streak = 0

            if identity is not None:
                if identity_box is not None:
                    # The dlib embedding is a ~350ms call that holds Python's GIL the whole
                    # time (confirmed by measurement), so a background thread would still
                    # stall the main loop's cv2.imshow for that long -- this runs in a
                    # separate process instead, which has its own GIL. The main loop only
                    # ever checks whether a previously submitted job has finished.
                    if identity_future is None and frame_num % args.identity_interval == 0:
                        identity_future = identity_executor.submit(
                            _identity_distance_job, frame.copy(), tuple(identity_box))
                    elif identity_future is not None and identity_future.done():
                        try:
                            dist = identity_future.result()
                        except Exception:
                            log.exception("identity computation failed")
                            dist = None
                        identity_future = None
                        if dist is not None:
                            identity_scores.append(dist)
                        if identity_scores:
                            avg_distance = sum(identity_scores) / len(identity_scores)
                        t = timed("identity.harvest", t)
                        if len(identity_scores) >= args.identity_window:
                            new_is_me = avg_distance <= args.identity_threshold
                            if new_is_me != is_me:
                                event = "ME" if new_is_me else "UNKNOWN"
                                identity_logger.writerow(
                                    [time.strftime("%Y-%m-%d %H:%M:%S"), event, f"{avg_distance:.3f}"])
                                identity_log_file.flush()
                                print(f"[{time.strftime('%H:%M:%S')}] Identity: {event} "
                                      f"(avg distance {avg_distance:.3f})")
                                # Identity smoothing (identity_window samples) can resolve after
                                # occupancy already triggered (enter_frames frames), so the initial
                                # wake_monitor() call may have been skipped under --require-identity.
                                # Catch up here once a match is confirmed.
                                if new_is_me and occupied and args.monitor_control and args.require_identity:
                                    wake_monitor()
                            is_me = new_is_me
                else:
                    identity_scores.clear()
                    is_me = None
                    avg_distance = None
                    identity_future = None  # drop reference; the thread (if any) finishes on its own

            if not occupied and in_streak >= cfg["enter_frames"]:
                occupied = True
                now = time.time()
                duration = now - last_transition
                logger.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), "OCCUPIED", f"{duration:.1f}"])
                log_file.flush()
                print(f"[{time.strftime('%H:%M:%S')}] OCCUPIED (was empty for {duration:.1f}s)")
                last_transition = now
                if args.monitor_control and (not args.require_identity or is_me):
                    wake_monitor()
                t = timed("monitor.wake", t)
            elif occupied and out_streak >= cfg["exit_frames"]:
                occupied = False
                now = time.time()
                duration = now - last_transition
                logger.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), "EMPTY", f"{duration:.1f}"])
                log_file.flush()
                print(f"[{time.strftime('%H:%M:%S')}] EMPTY (was occupied for {duration:.1f}s)")
                last_transition = now
                if args.monitor_control:
                    turn_monitor_off()
                t = timed("monitor.off", t)

            if not args.headless:
                # --- overlay ---
                rx, ry, rw, rh = roi
                zone_color = (0, 200, 0) if occupied else (0, 0, 255)
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), zone_color, 2)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 1)

                now_time = time.time()
                fps = 1.0 / max(1e-6, now_time - prev_frame_time)
                prev_frame_time = now_time

                status = "OCCUPIED" if occupied else "EMPTY"
                cv2.putText(frame, f"Status: {status}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, zone_color, 2)
                cv2.putText(frame, f"FPS: {fps:.1f}  in:{in_streak} out:{out_streak}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if identity is not None:
                    if avg_distance is None:
                        id_text, id_color = "ID: --", (200, 200, 200)
                    elif is_me is None:
                        id_text, id_color = f"ID: evaluating ({avg_distance:.3f})", (0, 200, 255)
                    elif is_me:
                        id_text, id_color = f"ID: ME ({avg_distance:.3f})", (0, 200, 0)
                    else:
                        id_text, id_color = f"ID: UNKNOWN ({avg_distance:.3f})", (0, 0, 255)
                    cv2.putText(frame, id_text, (10, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, id_color, 2)

                t = timed("overlay", t)

                cv2.imshow("Chair presence", frame)
                t = timed("imshow", t)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                timed("waitKey", t)
    except KeyboardInterrupt:
        print("Stopping (Ctrl+C).")
    finally:
        log_file.close()
        if identity_log_file is not None:
            identity_log_file.close()
        if identity_executor is not None:
            identity_executor.shutdown(wait=False, cancel_futures=True)
        if identity is not None:
            identity.close()
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Detect presence in a calibrated chair zone via webcam.")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--config", default=CONFIG_PATH_DEFAULT, help="path to config JSON")
    p.add_argument("--log", default=LOG_PATH_DEFAULT, help="path to CSV event log")
    p.add_argument("--debug-log", dest="debug_log", default=DEBUG_LOG_PATH_DEFAULT,
                    help="path to per-step timing/debug log, used to diagnose freezes (default debug.log)")
    p.add_argument("--calibrate", action="store_true", help="(re)draw the chair zone before running")
    p.add_argument("--no-monitor-control", dest="monitor_control", action="store_false",
                    help="disable turning the monitor off/on with occupancy (Windows only)")
    p.add_argument("--scale-factor", dest="scale_factor", type=float, default=None,
                    help="Haar cascade scaleFactor, lower = more sensitive/slower (default 1.1)")
    p.add_argument("--min-neighbors", dest="min_neighbors", type=int, default=None,
                    help="Haar cascade minNeighbors, lower = more sensitive/more false positives (default 5)")
    p.add_argument("--min-size", dest="min_size", type=int, default=None,
                    help="minimum face size in pixels, tune for distance from camera (default 60)")
    p.add_argument("--enter-frames", dest="enter_frames", type=int, default=None,
                    help="consecutive positive frames before declaring OCCUPIED (default 8)")
    p.add_argument("--exit-frames", dest="exit_frames", type=int, default=None,
                    help="consecutive negative frames before declaring EMPTY (default 15)")

    p.add_argument("--width", type=int, default=640,
                    help="requested camera capture width in pixels (default 640; smaller = less "
                         "CPU per frame for both detection and identity). Pass 0 to leave the "
                         "camera at its own default resolution.")
    p.add_argument("--height", type=int, default=480,
                    help="requested camera capture height in pixels (default 480). Pass 0 to leave "
                         "the camera at its own default resolution.")
    p.add_argument("--detect-interval", dest="detect_interval", type=int, default=2,
                    help="run Haar cascade face detection every Nth frame, reusing the last result "
                         "on skipped frames (default 2; occupancy debounce already tolerates this)")
    p.add_argument("--identity-interval", dest="identity_interval", type=int, default=20,
                    help="run the dlib face embedding (by far the most expensive step -- observed "
                         "300-370ms per call) every Nth frame while a face is in the zone, instead "
                         "of every frame (default 20). Raise this further if CPU/power use is still "
                         "too high; lower it for faster ME/UNKNOWN confirmation.")
    p.add_argument("--headless", action="store_true",
                    help="skip the preview window entirely (no overlay drawing, no imshow) for "
                         "lower overhead once you're done tuning; stop with Ctrl+C instead of 'q'")

    p.add_argument("--enroll-face", dest="enroll_face", action="store_true",
                    help="capture face embedding samples and save a profile, then exit")
    p.add_argument("--face-profile", dest="face_profile", default=FACE_PROFILE_PATH_DEFAULT,
                    help="path to the enrolled face profile JSON (default face_profile.json)")
    p.add_argument("--shape-predictor-model", dest="shape_predictor_model",
                    default=face_identity.SHAPE_PREDICTOR_PATH_DEFAULT,
                    help="path to dlib's shape_predictor_5_face_landmarks.dat")
    p.add_argument("--face-rec-model", dest="face_rec_model",
                    default=face_identity.FACE_REC_MODEL_PATH_DEFAULT,
                    help="path to dlib's dlib_face_recognition_resnet_model_v1.dat")
    p.add_argument("--identity-log", dest="identity_log", default=IDENTITY_LOG_PATH_DEFAULT,
                    help="path to CSV identity event log (default identity_log.csv)")
    p.add_argument("--identity-threshold", dest="identity_threshold", type=float,
                    default=face_identity.DEFAULT_THRESHOLD,
                    help="max Euclidean distance (128-d dlib embedding) to count as a match; "
                         f"lower = stricter (default {face_identity.DEFAULT_THRESHOLD}, "
                         "dlib's standard same-person cutoff)")
    p.add_argument("--identity-window", dest="identity_window", type=int, default=3,
                    help="consecutive identity samples averaged before committing to ME/UNKNOWN "
                         "(default 3; each sample is identity-interval frames apart, so total delay "
                         "before first commit is roughly identity-window x identity-interval frames)")
    p.add_argument("--no-require-identity", dest="require_identity", action="store_false",
                    help="wake the monitor for any presence in the zone, not just a recognized "
                         "match (default: require a match; falls back to any-presence "
                         "automatically if no profile is enrolled)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
