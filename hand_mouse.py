"""
Hand Gesture Virtual Mouse v3.0
================================
Uses MediaPipe Tasks API (0.10.30+)

GESTURES:
  Index finger up          -> Move cursor (always active)
  Thumb + Index pinch      -> Left click (while still moving)
  Thumb + Middle pinch     -> Right click
  Index + Middle up        -> Scroll mode (move hand up/down)
  Fist (all curled)        -> Drag & drop
  Open palm (5 fingers)    -> Pause (safe zone)
  Thumb only               -> Double click

Press Q to quit | Press P to pause/resume | Press D for debug overlay
"""

import cv2
import math
import time
import os
import sys
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pyautogui

# ============================================================
# Auto-download model
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 5_000_000:
    print("Downloading hand_landmarker.task ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done!")
    except Exception as e:
        print(f"Download failed: {e}\nPlease download manually:\n  {MODEL_URL}\nPlace at:\n  {MODEL_PATH}")
        sys.exit(1)

# ============================================================
# Config
# ============================================================
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()

CAM_INDEX = 0
CAM_W, CAM_H = 1280, 720
MARGIN = 100

# Pinch = normalized distance between fingertips (0.0 to 1.0 scale)
PINCH_THRESH = 0.06
CLICK_CD = 0.4
DBLCLICK_CD = 0.6
RCLICK_CD = 0.4

SMOOTHING = 0.25          # lower = smoother but more lag
DEAD_ZONE = 3             # pixels on screen
SCROLL_SENS = 3.0
FIST_CURL_THRESH = 0.05   # how close tip must be to MCP to count as curled

# Colors (BGR)
COL = {
    "green": (0, 255, 0), "red": (0, 0, 255), "blue": (255, 0, 0),
    "cyan": (255, 255, 0), "yellow": (0, 255, 255), "magenta": (255, 0, 255),
    "orange": (0, 165, 255), "white": (255, 255, 255), "gray": (100, 100, 100),
    "dark": (20, 20, 20), "purple": (180, 0, 255),
}

# ============================================================
# MediaPipe init
# ============================================================
base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
lm_options = vision.HandLandmarkerOptions(
    base_options=base_opts,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(lm_options)

# ============================================================
# Finger detection
# ============================================================

def norm_dist(lm, i, j):
    """Euclidean distance between two landmarks (normalized coords)."""
    return math.hypot(lm[i].x - lm[j].x, lm[i].y - lm[j].y)

def get_fingers(lm):
    """
    Finger state detection.
    Returns dict of finger_name -> bool (True = extended).

    For index/middle/ring/pinky: tip.y < pip.y means extended (simple & reliable).
    For thumb: tip should be far from index MCP base (works regardless of hand).
    """
    # --- Thumb ---
    # Thumb is extended if tip (4) is far from index finger base (5)
    # compared to thumb IP (3) to index base (5)
    thumb_tip_dist = norm_dist(lm, 4, 5)
    thumb_ip_dist = norm_dist(lm, 3, 5)
    thumb_up = thumb_tip_dist > thumb_ip_dist

    # --- Other fingers: simple tip above PIP joint ---
    return {
        "thumb":  thumb_up,
        "index":  lm[8].y < lm[6].y,    # tip above PIP
        "middle": lm[12].y < lm[10].y,
        "ring":   lm[16].y < lm[14].y,
        "pinky":  lm[20].y < lm[18].y,
    }

def is_fist(lm, fingers):
    """Fist = all 4 main fingers curled (tips below their PIP joints)."""
    if fingers["index"] or fingers["middle"] or fingers["ring"] or fingers["pinky"]:
        return False
    # Extra: all tips should be below (y >) their MCP joints
    for tip, mcp in [(8,5), (12,9), (16,13), (20,17)]:
        if lm[tip].y < lm[mcp].y:
            return False
    return True

# ============================================================
# Gesture detection
# ============================================================

def detect_gesture(fingers, fist, pinch_l, pinch_r):
    """
    Gesture priority:
    1. Open palm (all 5 up)     -> PAUSE
    2. Fist (all curled)        -> DRAG  
    3. Index+Middle up          -> SCROLL
    4. Index up (+ optional thumb) -> MOVE (pinch overlays click)
    5. Thumb only               -> DBLCLICK
    6. Otherwise                -> IDLE
    """
    up_count = sum(fingers.values())
    
    if up_count >= 4 and fingers["index"] and fingers["middle"] and fingers["ring"]:
        return "PAUSE"
    
    if fist:
        return "DRAG"
    
    if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        return "SCROLL"

    if fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        return "MOVE"  # pinch check done separately

    if fingers["thumb"] and up_count == 1:
        return "DBLCLICK"

    return "IDLE"

# ============================================================
# Drawing
# ============================================================

HAND_CONNS = [
    (0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(9,10),(10,11),(11,12),
    (13,14),(14,15),(15,16),(17,18),(18,19),(19,20),
    (0,5),(5,9),(9,13),(13,17),(0,17),
]

def draw_skeleton(img, lm, w, h, color=(0, 220, 0)):
    pts = [(int(l.x * w), int(l.y * h)) for l in lm]
    for a, b in HAND_CONNS:
        if a < len(pts) and b < len(pts):
            cv2.line(img, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(img, p, 3, (0, 255, 0), -1)

def draw_panel(img, x, y, w_p, h_p, alpha=0.65):
    ov = img.copy()
    cv2.rectangle(ov, (x, y), (x + w_p, y + h_p), COL["dark"], -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

def draw_hud(img, gesture, fps, fingers, pl, pr, dragging, debug):
    h, w = img.shape[:2]
    gcol = {"MOVE": COL["green"], "LEFT CLICK": COL["red"], "RIGHT CLICK": COL["yellow"],
            "DBLCLICK": COL["orange"], "SCROLL": COL["cyan"], "DRAG": COL["magenta"],
            "PAUSE": COL["white"], "IDLE": COL["gray"], "NO HAND": COL["gray"]}.get(gesture, COL["white"])

    # Left panel
    draw_panel(img, 8, 8, 310, 155)
    cv2.rectangle(img, (8, 8), (318, 163), gcol, 2)
    cv2.putText(img, gesture, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, gcol, 2)

    # Finger dots
    for i, (n, k) in enumerate(zip(["THM","IDX","MID","RNG","PNK"],
                                     ["thumb","index","middle","ring","pinky"])):
        fx = 22 + i * 58
        up = fingers.get(k, False)
        cv2.circle(img, (fx + 14, 68), 6, COL["green"] if up else (60, 60, 60), -1)
        cv2.putText(img, n, (fx, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL["white"], 1)

    # Pinch bars
    def pinch_bar(label, val, thresh, y_pos, col_active):
        pct = max(0, min(1, 1 - val / (thresh * 3)))
        bar_w = int(pct * 140)
        active = val < thresh
        c = col_active if active else (80, 80, 80)
        cv2.putText(img, label, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
        cv2.rectangle(img, (90, y_pos - 10), (90 + bar_w, y_pos), c, -1)
        cv2.rectangle(img, (90, y_pos - 10), (230, y_pos), c, 1)
        cv2.putText(img, f"{val:.3f}", (240, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)

    pinch_bar("L-Pinch", pl, PINCH_THRESH, 115, COL["red"])
    pinch_bar("R-Pinch", pr, PINCH_THRESH, 140, COL["yellow"])

    if dragging:
        cv2.putText(img, "DRAG ACTIVE", (20, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL["magenta"], 2)

    # Bottom bar
    cv2.rectangle(img, (0, h - 35), (w, h), COL["dark"], -1)
    cv2.putText(img, f"FPS: {int(fps)}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL["green"], 1)
    cv2.putText(img, f"{SCREEN_W}x{SCREEN_H}", (110, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL["gray"], 1)
    cv2.putText(img, "[Q]uit [P]ause [D]ebug", (w - 250, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

    # Right guide panel
    if not debug:
        gx = w - 245
        guide = [("Index up","Move",COL["green"]),("Thm+Idx pinch","L-Click",COL["red"]),
                 ("Thm+Mid pinch","R-Click",COL["yellow"]),("Thumb only","Dbl-Click",COL["orange"]),
                 ("Idx+Mid up","Scroll",COL["cyan"]),("Fist","Drag",COL["magenta"]),
                 ("Open palm","Pause",COL["white"])]
        gh = len(guide) * 20 + 25
        draw_panel(img, gx - 8, 8, 248, gh)
        cv2.putText(img, "GESTURES", (gx, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL["white"], 1)
        for i, (gt, act, c) in enumerate(guide):
            gy = 46 + i * 20
            cv2.circle(img, (gx + 4, gy - 3), 3, c, -1)
            cv2.putText(img, f"{gt}: {act}", (gx + 14, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (190,190,190), 1)

def draw_zone(img, x1, y1, x2, y2, color):
    cl = 25
    for c, he, ve in [((x1,y1),(x1+cl,y1),(x1,y1+cl)),((x2,y1),(x2-cl,y1),(x2,y1+cl)),
                       ((x1,y2),(x1+cl,y2),(x1,y2-cl)),((x2,y2),(x2-cl,y2),(x2,y2-cl))]:
        cv2.line(img, c, he, color, 2)
        cv2.line(img, c, ve, color, 2)

# ============================================================
# Camera
# ============================================================
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("ERROR: Cannot open webcam"); sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

# ============================================================
# State
# ============================================================
sx = sy = None                           # smoothed screen coords
prev_lpinch = prev_rpinch = False
last_lclick_t = last_rclick_t = last_dbl_t = 0.0
prev_scroll_y = None
prev_time = time.time()
drag_on = False
paused = False
debug_mode = False
frame_ts = 0

print(__doc__)
print(f"Screen: {SCREEN_W}x{SCREEN_H} | Camera: {CAM_W}x{CAM_H}")
print("Show your hand to the webcam!\n")

# ============================================================
# Main loop
# ============================================================
while True:
    ok, frame = cap.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]

    # Active zone
    zx1, zy1 = MARGIN, MARGIN
    zx2, zy2 = fw - MARGIN, fh - MARGIN

    gesture = "NO HAND"
    fingers = {}
    pl = pr = 999.0

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    frame_ts += 33
    result = detector.detect_for_video(mp_img, frame_ts)

    if result.hand_landmarks and not paused:
        lm = result.hand_landmarks[0]
        draw_skeleton(frame, lm, fw, fh)

        # Pixel coords for visualization
        ix, iy = int(lm[8].x * fw), int(lm[8].y * fh)
        tx, ty = int(lm[4].x * fw), int(lm[4].y * fh)
        mx, my = int(lm[12].x * fw), int(lm[12].y * fh)

        # ── Finger states ──
        fingers = get_fingers(lm)
        fist = is_fist(lm, fingers)

        # ── Pinch distances (normalized, not pixel) ──
        pl = norm_dist(lm, 4, 8)    # thumb-index
        pr = norm_dist(lm, 4, 12)   # thumb-middle
        lpinch = pl < PINCH_THRESH
        rpinch = pr < PINCH_THRESH

        # ── Gesture ──
        gesture = detect_gesture(fingers, fist, lpinch, rpinch)
        now = time.time()

        # ────────────────────────────────────
        # CURSOR MOVEMENT (always when index is up, regardless of pinch)
        # This is the KEY fix: cursor moves even during click gestures
        # ────────────────────────────────────
        if fingers["index"] and gesture in ("MOVE", "SCROLL", "IDLE"):
            cx = max(zx1, min(zx2, ix))
            cy = max(zy1, min(zy2, iy))
            nx = (cx - zx1) / (zx2 - zx1)
            ny = (cy - zy1) / (zy2 - zy1)
            tgt_x, tgt_y = int(nx * SCREEN_W), int(ny * SCREEN_H)
            tgt_x = max(0, min(SCREEN_W - 1, tgt_x))
            tgt_y = max(0, min(SCREEN_H - 1, tgt_y))

            if sx is None:
                sx, sy = float(tgt_x), float(tgt_y)

            dx, dy = tgt_x - sx, tgt_y - sy
            if abs(dx) < DEAD_ZONE: dx = 0
            if abs(dy) < DEAD_ZONE: dy = 0
            sx += SMOOTHING * dx
            sy += SMOOTHING * dy

            if gesture != "SCROLL":
                pyautogui.moveTo(int(sx), int(sy), duration=0)

            if drag_on:
                pyautogui.mouseUp()
                drag_on = False

        # ── LEFT CLICK (pinch overlay — doesn't block movement) ──
        if lpinch and not prev_lpinch and (now - last_lclick_t) > CLICK_CD:
            if gesture in ("MOVE", "IDLE"):
                pyautogui.click()
                last_lclick_t = now
                gesture = "LEFT CLICK"

        # ── RIGHT CLICK ──
        if rpinch and not prev_rpinch and (now - last_rclick_t) > RCLICK_CD:
            if fingers["index"] or fingers["middle"]:
                pyautogui.rightClick()
                last_rclick_t = now
                gesture = "RIGHT CLICK"

        # ── DOUBLE CLICK ──
        if gesture == "DBLCLICK" and (now - last_dbl_t) > DBLCLICK_CD:
            pyautogui.doubleClick()
            last_dbl_t = now

        # ── SCROLL ──
        if gesture == "SCROLL":
            # Use middle finger tip for scroll tracking (more stable)
            mid_y = int(lm[12].y * fh)
            if prev_scroll_y is None:
                prev_scroll_y = mid_y
            else:
                dy = mid_y - prev_scroll_y
                if abs(dy) > 3:
                    scroll_amt = int(-dy * SCROLL_SENS / 8)
                    if scroll_amt != 0:
                        pyautogui.scroll(scroll_amt)
                prev_scroll_y = mid_y
        else:
            prev_scroll_y = None

        # ── DRAG ──
        if gesture == "DRAG":
            wrist_x = int(lm[9].x * fw)  # use middle MCP, more stable than wrist
            wrist_y = int(lm[9].y * fh)
            cx = max(zx1, min(zx2, wrist_x))
            cy = max(zy1, min(zy2, wrist_y))
            nx = (cx - zx1) / (zx2 - zx1)
            ny = (cy - zy1) / (zy2 - zy1)
            tgt_x = max(0, min(SCREEN_W - 1, int(nx * SCREEN_W)))
            tgt_y = max(0, min(SCREEN_H - 1, int(ny * SCREEN_H)))
            if sx is None:
                sx, sy = float(tgt_x), float(tgt_y)
            sx += SMOOTHING * (tgt_x - sx)
            sy += SMOOTHING * (tgt_y - sy)
            if not drag_on:
                pyautogui.mouseDown()
                drag_on = True
            pyautogui.moveTo(int(sx), int(sy), duration=0)

        # ── PAUSE ──
        if gesture == "PAUSE" and drag_on:
            pyautogui.mouseUp()
            drag_on = False

        prev_lpinch = lpinch
        prev_rpinch = rpinch

        # Draw fingertip markers
        cv2.circle(frame, (ix, iy), 8, COL["blue"], -1)
        cv2.circle(frame, (tx, ty), 8, COL["red"], -1)
        cv2.circle(frame, (mx, my), 8, COL["yellow"], -1)

        if lpinch:
            cv2.line(frame, (ix, iy), (tx, ty), COL["red"], 3)
        if rpinch:
            cv2.line(frame, (mx, my), (tx, ty), COL["yellow"], 3)

        # Debug overlay
        if debug_mode:
            draw_panel(frame, 8, 175, 310, 130)
            cv2.putText(frame, f"Fist: {fist}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL["magenta"] if fist else COL["gray"], 1)
            cv2.putText(frame, f"L-pinch: {pl:.4f} (thr {PINCH_THRESH})", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL["red"] if lpinch else COL["gray"], 1)
            cv2.putText(frame, f"R-pinch: {pr:.4f} (thr {PINCH_THRESH})", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL["yellow"] if rpinch else COL["gray"], 1)
            cv2.putText(frame, f"Cursor: ({int(sx or 0)}, {int(sy or 0)})", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL["green"], 1)
            cv2.putText(frame, f"Drag: {drag_on}", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL["magenta"] if drag_on else COL["gray"], 1)
    else:
        if drag_on:
            pyautogui.mouseUp()
            drag_on = False
        prev_lpinch = prev_rpinch = False
        prev_scroll_y = None
        if paused:
            gesture = "PAUSE"

    # Zone corners
    gcol = {"MOVE": COL["green"], "LEFT CLICK": COL["red"], "RIGHT CLICK": COL["yellow"],
            "DBLCLICK": COL["orange"], "SCROLL": COL["cyan"], "DRAG": COL["magenta"],
            "PAUSE": COL["white"]}.get(gesture, COL["gray"])
    draw_zone(frame, zx1, zy1, zx2, zy2, gcol)

    # FPS
    ct = time.time()
    fps = 1.0 / (ct - prev_time) if ct > prev_time else 0
    prev_time = ct

    draw_hud(frame, gesture, fps, fingers, pl, pr, drag_on, debug_mode)
    cv2.imshow("Hand Gesture Virtual Mouse v3.0", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('p'):
        paused = not paused
        print("[PAUSED]" if paused else "[RESUMED]")
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"[DEBUG {'ON' if debug_mode else 'OFF'}]")

# Cleanup
if drag_on:
    pyautogui.mouseUp()
cap.release()
cv2.destroyAllWindows()
detector.close()
print("Goodbye!")
