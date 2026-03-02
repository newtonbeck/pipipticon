"""
pipipticon — macOS Dog Deterrent App
Run: python app.py
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


class DogDeterrentApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("pipipticon — Dog Deterrent")
        self.root.resizable(False, False)

        # --- State ---
        self.running = False
        self.audio_path: str | None = None

        # ROI: stored as (x1, y1, x2, y2) in canvas coords
        self._roi: tuple[int, int, int, int] | None = None
        self._roi_start: tuple[int, int] | None = None
        self._roi_rect_id = None  # canvas rectangle item id

        # Detection mode: "dog" | "motion"
        self._mode = tk.StringVar(value="dog")

        # Sensitivity / cooldown
        self._confidence = tk.DoubleVar(value=0.45)
        self._cooldown = tk.IntVar(value=5)

        # Background subtractor (motion mode)
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=False
        )

        # YOLO state
        self._yolo_model = None
        self._yolo_thread: threading.Thread | None = None
        self._yolo_lock = threading.Lock()
        self._yolo_detected = False

        # Audio state
        self._audio_lock = threading.Lock()
        self._audio_playing = False
        self._last_alert_time = 0.0

        # Camera
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam.")

        # Current frame (numpy) for display
        self._current_frame = None

        # Detection indicator state
        self._alert_on = False

        self._build_ui()
        self._tick()

    # ------------------------------------------------------------------ #
    #  UI
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        # Left: video canvas
        left = tk.Frame(self.root)
        left.pack(side=tk.LEFT, fill=tk.BOTH)

        self.canvas = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H, bg="black", cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._roi_press)
        self.canvas.bind("<B1-Motion>", self._roi_drag)
        self.canvas.bind("<ButtonRelease-1>", self._roi_release)

        # Status bar
        self._status_var = tk.StringVar(value="Ready — draw a watch zone, load audio, then Start.")
        status_bar = tk.Label(left, textvariable=self._status_var, anchor="w",
                              relief=tk.SUNKEN, font=("Helvetica", 11))
        status_bar.pack(fill=tk.X)

        # Right: controls
        right = tk.Frame(self.root, padx=12, pady=12)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        # ---- Audio section ----
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
        tk.Label(right, text="ZONE", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 4))
        tk.Button(right, text="Clear zone", command=self._clear_roi).pack(anchor="w")

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, pady=10)

        # ---- Start/Stop + alert indicator ----
        bottom = tk.Frame(right)
        bottom.pack(anchor="w", pady=(0, 4))

        self._start_btn = tk.Button(bottom, text="▶  Start", width=10,
                                    font=("Helvetica", 12, "bold"),
                                    bg="#2ecc71", activebackground="#27ae60",
                                    command=self._toggle)
        self._start_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._alert_canvas = tk.Canvas(bottom, width=24, height=24, bg=self.root.cget("bg"),
                                       highlightthickness=0)
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
        img = Image.fromarray(frame_rgb)

        # Draw ROI rectangle
        if self._roi:
            x1, y1, x2, y2 = self._roi
            color = (255, 0, 0) if self._alert_on else (0, 220, 0)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw "LIVE" dot top-right
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.ellipse([CANVAS_W - 20, 8, CANVAS_W - 8, 20], fill=(255, 50, 50))
        draw.text((CANVAS_W - 44, 9), "LIVE", fill=(255, 50, 50))

        photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo  # keep reference

        # Alert dot
        dot_color = "#e74c3c" if self._alert_on else "gray"
        self._alert_canvas.itemconfig(self._alert_dot, fill=dot_color)

    # ------------------------------------------------------------------ #
    #  Detection
    # ------------------------------------------------------------------ #
    def _process(self, frame_rgb):
        if self._roi is None:
            self._status("No watch zone defined — draw one on the video.")
            return

        x1, y1, x2, y2 = self._roi
        # Clamp to frame bounds
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(CANVAS_W, x2), min(CANVAS_H, y2)
        if x2c <= x1c or y2c <= y1c:
            return

        roi_img = frame_rgb[y1c:y2c, x1c:x2c]

        mode = self._mode.get()

        if mode == "motion":
            motion = self._detect_motion(roi_img)
            if motion:
                self._trigger_alert()
            else:
                self._alert_on = False
        else:
            # Dog (YOLO) mode: use motion as cheap pre-filter
            motion = self._detect_motion(roi_img)
            if motion:
                self._maybe_run_yolo(roi_img)

            with self._yolo_lock:
                detected = self._yolo_detected

            if detected:
                self._trigger_alert()
                self._alert_on = True
            else:
                self._alert_on = False

    def _detect_motion(self, roi_bgr) -> bool:
        """Returns True if significant motion detected in ROI."""
        # bg_sub expects BGR — convert
        gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_RGB2BGR)
        mask = self._bg_sub.apply(gray_roi)
        motion_pixels = cv2.countNonZero(mask)
        total_pixels = roi_bgr.shape[0] * roi_bgr.shape[1]
        if total_pixels == 0:
            return False
        return (motion_pixels / total_pixels) > 0.02  # >2% changed

    def _maybe_run_yolo(self, roi_rgb):
        """Spawn YOLO thread if not already running."""
        with self._yolo_lock:
            if self._yolo_thread is not None and self._yolo_thread.is_alive():
                return  # already running

        t = threading.Thread(target=self._run_yolo_async, args=(roi_rgb.copy(),), daemon=True)
        with self._yolo_lock:
            self._yolo_thread = t
        t.start()

    def _run_yolo_async(self, roi_rgb):
        """Background thread: runs YOLO on roi_rgb, sets _yolo_detected."""
        try:
            if self._yolo_model is None:
                self._status("Loading YOLOv8n model (first run)…")
                from ultralytics import YOLO
                self._yolo_model = YOLO("yolov8n.pt")

            img_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
            results = self._yolo_model(img_bgr, verbose=False)
            detected = False
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls == DOG_CLASS_ID and conf >= self._confidence.get():
                        detected = True
                        break
                if detected:
                    break

            with self._yolo_lock:
                self._yolo_detected = detected
        except Exception as e:
            self._status(f"YOLO error: {e}")
            with self._yolo_lock:
                self._yolo_detected = False

    def _trigger_alert(self):
        now = time.time()
        cooldown = self._cooldown.get()
        self._alert_on = True
        if now - self._last_alert_time >= cooldown:
            self._last_alert_time = now
            if self._mode.get() == "motion":
                self._status("Motion detected in zone! Playing deterrent…")
            else:
                self._status("Dog detected in zone! Playing deterrent…")
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

        t = threading.Thread(target=_play, daemon=True)
        t.start()

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
        if self._roi_rect_id:
            self.canvas.delete(self._roi_rect_id)
            self._roi_rect_id = None
        self._roi = None

    def _roi_drag(self, event):
        if self._roi_start is None:
            return
        x0, y0 = self._roi_start
        if self._roi_rect_id:
            self.canvas.delete(self._roi_rect_id)
        self._roi_rect_id = self.canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline="#00ff00", width=2, dash=(6, 3)
        )

    def _roi_release(self, event):
        if self._roi_start is None:
            return
        x0, y0 = self._roi_start
        x1, y1 = event.x, event.y
        # Normalise so top-left < bottom-right
        self._roi = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self._roi_start = None
        if self._roi_rect_id:
            self.canvas.delete(self._roi_rect_id)
            self._roi_rect_id = None
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        self._status(f"Watch zone set ({w}×{h} px) — click Start to begin monitoring.")

    def _clear_roi(self):
        self._roi = None
        if self._roi_rect_id:
            self.canvas.delete(self._roi_rect_id)
            self._roi_rect_id = None
        self._status("Watch zone cleared.")

    # ------------------------------------------------------------------ #
    #  Start / Stop
    # ------------------------------------------------------------------ #
    def _toggle(self):
        if self.running:
            self.running = False
            self._alert_on = False
            self._start_btn.config(text="▶  Start", bg="#2ecc71", activebackground="#27ae60")
            self._status("Monitoring stopped.")
        else:
            if self._roi is None:
                messagebox.showwarning("No zone", "Draw a watch zone on the video first.")
                return
            if not self.audio_path:
                ok = messagebox.askyesno(
                    "No audio",
                    "No audio file loaded. Continue without deterrent sound?",
                )
                if not ok:
                    return
            self.running = True
            self._last_alert_time = 0.0
            with self._yolo_lock:
                self._yolo_detected = False
            self._start_btn.config(text="◼  Stop", bg="#e74c3c", activebackground="#c0392b")
            mode_label = "YOLO dog" if self._mode.get() == "dog" else "motion"
            self._status(f"Monitoring — mode: {mode_label}. Watching for activity…")

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _status(self, msg: str):
        # Safe to call from any thread via tkinter's thread-safe variable
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
