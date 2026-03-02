"""
pipipticon — macOS Dog Deterrent App
Run: .venv/bin/python app.py
Deps: pip install opencv-python pillow ultralytics
"""

import threading
import time
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import cv2
from PIL import Image, ImageTk

# COCO class index for dog
DOG_CLASS_ID = 16

CANVAS_W = 640
CANVAS_H = 480
TICK_MS = 30  # ~33 fps

# Distinct zone colours (R, G, B) cycled for each zone
ZONE_COLORS = [
    (0, 220, 0),
    (0, 180, 255),
    (255, 200, 0),
    (180, 0, 255),
    (0, 220, 220),
]
ALERT_COLOR = (255, 50, 50)


class DogDeterrentApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("pipipticon — Dog Deterrent")
        self.root.resizable(False, False)

        # --- State ---
        self.running = False
        self.audio_path: str | None = None

        # ROIs: list of (x1, y1, x2, y2) in canvas coords
        self._rois: list[tuple[int, int, int, int]] = []
        # In-progress drag
        self._roi_start: tuple[int, int] | None = None
        self._roi_rect_id = None  # canvas item for drag ghost

        # Which ROI indices are currently alerting
        self._alerting_rois: set[int] = set()

        # Detection mode: "dog" | "motion"
        self._mode = tk.StringVar(value="dog")

        # Sensitivity / cooldown
        self._confidence = tk.DoubleVar(value=0.45)
        self._cooldown = tk.IntVar(value=5)

        # Background subtractor — applied to full frame
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=False
        )

        # YOLO state
        self._yolo_model = None
        self._yolo_thread: threading.Thread | None = None
        self._yolo_lock = threading.Lock()
        self._yolo_detected_rois: set[int] = set()

        # Audio state
        self._audio_lock = threading.Lock()
        self._audio_playing = False
        self._last_alert_time = 0.0

        # Camera
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam.")

        self._current_frame = None
        self._alert_on = False

        self._build_ui()
        self._tick()

    # ------------------------------------------------------------------ #
    #  UI
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        left = tk.Frame(self.root)
        left.pack(side=tk.LEFT, fill=tk.BOTH)

        self.canvas = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H,
                                bg="black", cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._roi_press)
        self.canvas.bind("<B1-Motion>", self._roi_drag)
        self.canvas.bind("<ButtonRelease-1>", self._roi_release)
        self.canvas.bind("<ButtonPress-3>", self._roi_remove)  # right-click removes

        self._status_var = tk.StringVar(
            value="Ready — draw watch zones (right-click to remove), load audio, then Start."
        )
        tk.Label(left, textvariable=self._status_var, anchor="w",
                 relief=tk.SUNKEN, font=("Helvetica", 11)).pack(fill=tk.X)

        right = tk.Frame(self.root, padx=12, pady=12)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        # ---- Audio ----
        tk.Label(right, text="AUDIO", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 4))
        self._audio_label = tk.Label(right, text="No file loaded", fg="gray",
                                     font=("Helvetica", 10), wraplength=160, justify="left")
        self._audio_label.pack(anchor="w")
        btn_frame = tk.Frame(right)
        btn_frame.pack(anchor="w", pady=4)
        tk.Button(btn_frame, text="Load file…", command=self._load_audio).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(btn_frame, text="Test", command=self._test_audio).pack(side=tk.LEFT)

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, pady=10)

        # ---- Detection mode ----
        tk.Label(right, text="DETECTION", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 4))
        tk.Radiobutton(right, text="Dog only (YOLO)", variable=self._mode, value="dog").pack(anchor="w")
        tk.Radiobutton(right, text="Any motion", variable=self._mode, value="motion").pack(anchor="w")

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, pady=10)

        # ---- Sensitivity ----
        tk.Label(right, text="SENSITIVITY", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 4))
        tk.Label(right, text="Confidence (YOLO)").pack(anchor="w")
        tk.Scale(right, variable=self._confidence, from_=0.1, to=0.9,
                 resolution=0.05, orient=tk.HORIZONTAL, length=160).pack(anchor="w")
        tk.Label(right, text="Cooldown (seconds)").pack(anchor="w", pady=(6, 0))
        tk.Scale(right, variable=self._cooldown, from_=1, to=30,
                 resolution=1, orient=tk.HORIZONTAL, length=160).pack(anchor="w")

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, pady=10)

        # ---- Zone controls ----
        tk.Label(right, text="ZONES", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 4))
        self._zone_count_label = tk.Label(right, text="0 zones defined", fg="gray",
                                          font=("Helvetica", 10))
        self._zone_count_label.pack(anchor="w")
        tk.Label(right, text="Right-click a zone to remove it.", fg="gray",
                 font=("Helvetica", 9), wraplength=160, justify="left").pack(anchor="w", pady=(2, 4))
        tk.Button(right, text="Clear all zones", command=self._clear_all_rois).pack(anchor="w")

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, pady=10)

        # ---- Start/Stop ----
        bottom = tk.Frame(right)
        bottom.pack(anchor="w", pady=(0, 4))
        self._start_btn = tk.Button(bottom, text="▶  Start", width=10,
                                    font=("Helvetica", 12, "bold"),
                                    bg="#2ecc71", activebackground="#27ae60",
                                    command=self._toggle)
        self._start_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._alert_canvas = tk.Canvas(bottom, width=24, height=24,
                                       bg=self.root.cget("bg"), highlightthickness=0)
        self._alert_canvas.pack(side=tk.LEFT)
        self._alert_dot = self._alert_canvas.create_oval(2, 2, 22, 22, fill="gray", outline="")

    # ------------------------------------------------------------------ #
    #  Main tick loop
    # ------------------------------------------------------------------ #
    def _tick(self):
        ret, frame = self._cap.read()
        if ret:
            frame = cv2.resize(frame, (CANVAS_W, CANVAS_H))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._current_frame = frame_rgb

            if self.running:
                self._process(frame_rgb.copy())

            self._draw_frame(frame_rgb)

        self.root.after(TICK_MS, self._tick)

    def _draw_frame(self, frame_rgb):
        from PIL import ImageDraw
        img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img)

        # Draw each committed ROI
        for i, (x1, y1, x2, y2) in enumerate(self._rois):
            color = ALERT_COLOR if i in self._alerting_rois else ZONE_COLORS[i % len(ZONE_COLORS)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 4, y1 + 4), f"Zone {i + 1}", fill=color)

        # LIVE dot
        draw.ellipse([CANVAS_W - 20, 8, CANVAS_W - 8, 20], fill=(255, 50, 50))
        draw.text((CANVAS_W - 44, 9), "LIVE", fill=(255, 50, 50))

        photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

        dot_color = "#e74c3c" if self._alert_on else "gray"
        self._alert_canvas.itemconfig(self._alert_dot, fill=dot_color)

    # ------------------------------------------------------------------ #
    #  Detection
    # ------------------------------------------------------------------ #
    def _process(self, frame_rgb):
        if not self._rois:
            self._status("No watch zones defined — draw some on the video.")
            return

        # Apply MOG2 to full frame once
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        fg_mask = self._bg_sub.apply(frame_bgr)

        mode = self._mode.get()

        if mode == "motion":
            alerting = set()
            for i, roi in enumerate(self._rois):
                if self._roi_has_motion(fg_mask, roi):
                    alerting.add(i)
            self._alerting_rois = alerting
            if alerting:
                self._alert_on = True
                self._trigger_alert()
            else:
                self._alert_on = False

        else:  # YOLO dog mode
            # Use motion as cheap pre-filter; if any ROI has motion, run YOLO on full frame
            any_motion = any(self._roi_has_motion(fg_mask, roi) for roi in self._rois)
            if any_motion:
                self._maybe_run_yolo(frame_rgb)

            with self._yolo_lock:
                detected = self._yolo_detected_rois.copy()

            self._alerting_rois = detected
            if detected:
                self._alert_on = True
                self._trigger_alert()
            else:
                self._alert_on = False

    def _roi_has_motion(self, fg_mask, roi) -> bool:
        x1, y1, x2, y2 = roi
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(CANVAS_W, x2), min(CANVAS_H, y2)
        if x2c <= x1c or y2c <= y1c:
            return False
        region = fg_mask[y1c:y2c, x1c:x2c]
        total = (x2c - x1c) * (y2c - y1c)
        return (cv2.countNonZero(region) / total) > 0.02

    def _maybe_run_yolo(self, frame_rgb):
        with self._yolo_lock:
            if self._yolo_thread is not None and self._yolo_thread.is_alive():
                return
        t = threading.Thread(target=self._run_yolo_async,
                             args=(frame_rgb.copy(), list(self._rois)), daemon=True)
        with self._yolo_lock:
            self._yolo_thread = t
        t.start()

    def _run_yolo_async(self, frame_rgb, rois):
        """Run YOLO on the full frame; map dog boxes back to ROI indices."""
        try:
            if self._yolo_model is None:
                self._status("Loading YOLOv8n model (first run)…")
                from ultralytics import YOLO
                self._yolo_model = YOLO("yolov8n.pt")

            img_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            results = self._yolo_model(img_bgr, verbose=False)
            detected: set[int] = set()
            conf_thresh = self._confidence.get()

            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) != DOG_CLASS_ID:
                        continue
                    if float(box.conf[0]) < conf_thresh:
                        continue
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    for i, (rx1, ry1, rx2, ry2) in enumerate(rois):
                        # Check bounding-box overlap with this ROI
                        if bx2 > rx1 and bx1 < rx2 and by2 > ry1 and by1 < ry2:
                            detected.add(i)

            with self._yolo_lock:
                self._yolo_detected_rois = detected
        except Exception as e:
            self._status(f"YOLO error: {e}")
            with self._yolo_lock:
                self._yolo_detected_rois = set()

    def _trigger_alert(self):
        now = time.time()
        cooldown = self._cooldown.get()
        if now - self._last_alert_time >= cooldown:
            self._last_alert_time = now
            n = len(self._alerting_rois)
            zones = ", ".join(f"Zone {i + 1}" for i in sorted(self._alerting_rois))
            noun = "Dog" if self._mode.get() == "dog" else "Motion"
            self._status(f"{noun} detected in {zones}! Playing deterrent…")
            self._fire_sound()
        else:
            remaining = int(cooldown - (now - self._last_alert_time))
            self._status(f"Detected — cooldown ({remaining}s remaining)…")

    # ------------------------------------------------------------------ #
    #  Audio
    # ------------------------------------------------------------------ #
    def _fire_sound(self):
        if not self.audio_path:
            self._status("No audio file loaded — load one in the Audio panel.")
            return
        with self._audio_lock:
            if self._audio_playing:
                return
            self._audio_playing = True

        def _play():
            try:
                subprocess.run(["afplay", self.audio_path], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                self._status(f"Audio error: {e}")
            finally:
                with self._audio_lock:
                    self._audio_playing = False

        threading.Thread(target=_play, daemon=True).start()

    def _load_audio(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", "*.wav *.mp3 *.m4a *.aiff *.aif"), ("All files", "*.*")]
        )
        if path:
            self.audio_path = path
            self._audio_label.config(text=Path(path).name, fg="black")
            self._status(f"Loaded: {Path(path).name}")

    def _test_audio(self):
        if not self.audio_path:
            messagebox.showinfo("No audio", "Load an audio file first.")
            return
        self._fire_sound()

    # ------------------------------------------------------------------ #
    #  ROI drawing
    # ------------------------------------------------------------------ #
    def _roi_press(self, event):
        self._roi_start = (event.x, event.y)

    def _roi_drag(self, event):
        if self._roi_start is None:
            return
        x0, y0 = self._roi_start
        if self._roi_rect_id:
            self.canvas.delete(self._roi_rect_id)
        self._roi_rect_id = self.canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline="#ffffff", width=2, dash=(6, 3)
        )

    def _roi_release(self, event):
        if self._roi_start is None:
            return
        x0, y0 = self._roi_start
        x1, y1 = event.x, event.y
        self._roi_start = None
        if self._roi_rect_id:
            self.canvas.delete(self._roi_rect_id)
            self._roi_rect_id = None

        # Ignore tiny accidental clicks
        if abs(x1 - x0) < 10 or abs(y1 - y0) < 10:
            return

        roi = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self._rois.append(roi)
        self._update_zone_label()
        n = len(self._rois)
        self._status(f"Zone {n} added ({abs(x1-x0)}×{abs(y1-y0)} px). "
                     f"{n} zone{'s' if n != 1 else ''} total.")

    def _roi_remove(self, event):
        """Right-click: remove the zone under the cursor."""
        for i, (x1, y1, x2, y2) in enumerate(self._rois):
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self._rois.pop(i)
                self._alerting_rois.discard(i)
                # Re-index alerting rois after removal
                self._alerting_rois = {j if j < i else j - 1 for j in self._alerting_rois if j != i}
                self._update_zone_label()
                n = len(self._rois)
                self._status(f"Zone {i + 1} removed. {n} zone{'s' if n != 1 else ''} remaining.")
                return

    def _clear_all_rois(self):
        self._rois.clear()
        self._alerting_rois.clear()
        self._update_zone_label()
        self._status("All zones cleared.")

    def _update_zone_label(self):
        n = len(self._rois)
        if n == 0:
            self._zone_count_label.config(text="0 zones defined", fg="gray")
        else:
            self._zone_count_label.config(text=f"{n} zone{'s' if n != 1 else ''} defined", fg="black")

    # ------------------------------------------------------------------ #
    #  Start / Stop
    # ------------------------------------------------------------------ #
    def _toggle(self):
        if self.running:
            self.running = False
            self._alert_on = False
            self._alerting_rois.clear()
            self._start_btn.config(text="▶  Start", bg="#2ecc71", activebackground="#27ae60")
            self._status("Monitoring stopped.")
        else:
            if not self._rois:
                messagebox.showwarning("No zones", "Draw at least one watch zone on the video first.")
                return
            if not self.audio_path:
                if not messagebox.askyesno("No audio",
                                           "No audio file loaded. Continue without deterrent sound?"):
                    return
            self.running = True
            self._last_alert_time = 0.0
            with self._yolo_lock:
                self._yolo_detected_rois = set()
            self._start_btn.config(text="◼  Stop", bg="#e74c3c", activebackground="#c0392b")
            n = len(self._rois)
            mode_label = "YOLO dog" if self._mode.get() == "dog" else "motion"
            self._status(f"Monitoring {n} zone{'s' if n != 1 else ''} — mode: {mode_label}.")

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _status(self, msg: str):
        self.root.after(0, lambda: self._status_var.set(msg))

    def on_close(self):
        self.running = False
        self._cap.release()
        self.root.destroy()


# ------------------------------------------------------------------ #
#  Entry point
# ------------------------------------------------------------------ #
def main():
    root = tk.Tk()
    app = DogDeterrentApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
