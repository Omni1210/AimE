import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import tkinter as tk
from tkinter import ttk, messagebox
import requests
from io import BytesIO
from PIL import Image
import random
import threading
import time
import datetime
import os
from collections import deque

# ── Recording dependencies ────────────────────────────────────────────────────
try:
    import mss
    import cv2
    RECORDING_AVAILABLE = True
except ImportError:
    RECORDING_AVAILABLE = False

# ────────────────────────────────────────────────
# Scene scale
# ────────────────────────────────────────────────
EARTH_R      = 160
ASTEROID_R   = 22
IMPACTOR_SZ  = 12
OBS_SZ       = 10

OBS_COLORS  = ["orange", "lime", "magenta"]
OBS_OFFSETS = [
    np.array([ 90,  140,   80]),
    np.array([-110, -130,   90]),
    np.array([ 120,   40, -140]),
]

ASTEROID_START = np.array([3400.0, 420.0, 650.0])
IMPACTOR_START = np.array([-1000.0, 120.0, 100.0])
INTERCEPT_PT   = np.array([500.0, 60.0, 80.0])

RECORD_FPS    = 30
RECORD_WIDTH  = 1400
RECORD_HEIGHT = 800

# Trail settings
TRAIL_LEN     = 80   # how many past positions to keep per object
TRAIL_FADE    = True # opacity fades toward the tail


def _hat(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


EARTH_EXCLUSION_R = EARTH_R + 80

def _clamp_from_earth(pos: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(pos)
    if d < EARTH_EXCLUSION_R:
        return _hat(pos) * EARTH_EXCLUSION_R
    return pos


def _lerp_cam(plotter, target_pos, target_focal, alpha=0.04):
    """Smoothly interpolate camera position and focal point."""
    cam = plotter.camera
    cp  = np.array(cam.position)
    cf  = np.array(cam.focal_point)
    cam.position    = tuple(cp  + (np.array(target_pos)   - cp)  * alpha)
    cam.focal_point = tuple(cf  + (np.array(target_focal) - cf)  * alpha)


# ─────────────────────────────────────────────────────────────────────────────
# Recorder
# ─────────────────────────────────────────────────────────────────────────────
class ScreenRecorder:
    def __init__(self, x, y, width, height):
        self.region  = {"top": y, "left": x, "width": width, "height": height}
        self.running = False
        self._thread = None
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = r"C:\Users\jadda\Downloads\Projects_Creatives\EchoDart\Recording"
        os.makedirs(save_dir, exist_ok=True)
        self.output_path = os.path.join(save_dir, f"echodart_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.output_path, fourcc, RECORD_FPS, (width, height))

    def start(self):
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        self._writer.release()

    def _capture_loop(self):
        interval = 1.0 / RECORD_FPS
        with mss.mss() as sct:
            while self.running:
                t0    = time.perf_counter()
                img   = sct.grab(self.region)
                frame = np.array(img)[:, :, :3]
                frame = cv2.resize(frame, (RECORD_WIDTH, RECORD_HEIGHT))
                self._writer.write(frame)
                wait = interval - (time.perf_counter() - t0)
                if wait > 0:
                    time.sleep(wait)


recorder = None


def run_simulation():
    global recorder

    if not RECORDING_AVAILABLE:
        messagebox.showwarning(
            "Missing packages",
            "Recording requires mss and opencv-python.\n"
            "Run:  pip install mss opencv-python\n\n"
            "Launching simulation without recording."
        )

    try:
        plotter = BackgroundPlotter(
            window_size=(RECORD_WIDTH, RECORD_HEIGHT),
            title="EchoDART – Full Swarm Mission"
        )
        plotter.set_background("black")

        # ── Earth ────────────────────────────────────────────────────────────
        earth_mesh = pv.Sphere(radius=EARTH_R, theta_resolution=64, phi_resolution=32)
        earth_mesh.texture_map_to_sphere(inplace=True)
        texture = None
        try:
            url     = "https://www.solarsystemscope.com/textures/download/2k_earth_daymap.jpg"
            img     = Image.open(BytesIO(requests.get(url, timeout=10).content)).convert("RGB")
            texture = pv.Texture(np.array(img))
        except Exception:
            pass
        earth_actor = plotter.add_mesh(earth_mesh, texture=texture, smooth_shading=True)

        # ── Asteroid ─────────────────────────────────────────────────────────
        base  = pv.Sphere(radius=ASTEROID_R, theta_resolution=140, phi_resolution=90)
        pts   = base.points.copy()
        norms = base.point_normals.copy()
        pts  += np.random.normal(0, 1.0, len(pts))[:, None] * norms * 8.0
        pts  += np.random.normal(0, 1.5, len(pts))[:, None] * norms * 5.0
        crater_depth = 9.0 + np.random.uniform(0, 12.0, len(pts))
        mask = np.random.rand(len(pts)) < 0.025
        pts[mask] -= crater_depth[mask, None] * norms[mask]
        base.points = pts
        asteroid = base.smooth(n_iter=5, relaxation_factor=0.12, feature_angle=55)
        asteroid.compute_normals(inplace=True)
        asteroid_actor = plotter.add_mesh(
            asteroid, color="#3f3328",
            specular=0.25, specular_power=6,
            ambient=0.10, diffuse=0.88, smooth_shading=True
        )
        # Tumble axes — slightly off-axis like a real tumbling rock
        AST_SPIN_AXIS  = _hat(np.array([0.3, 0.8, 0.5]))
        AST_SPIN_RATE  = 0.18   # degrees per frame

        # ── Impactor ─────────────────────────────────────────────────────────
        imp_body_length = IMPACTOR_SZ * 1.6
        imp_body = pv.Box(bounds=[-IMPACTOR_SZ/2, IMPACTOR_SZ/2,
                                   -IMPACTOR_SZ/2, IMPACTOR_SZ/2,
                                   -imp_body_length/2, imp_body_length/2])
        impactor_body_actor = plotter.add_mesh(imp_body, color="crimson",
                                               specular=0.6, smooth_shading=True)

        PANEL_W = IMPACTOR_SZ * 1.1
        PANEL_H = imp_body_length * 0.55
        PANEL_T = 0.35
        imp_panels, imp_panel_offsets = [], []
        for side in [-1, 1]:
            panel  = pv.Box(bounds=[-PANEL_T, PANEL_T, -PANEL_W, PANEL_W, -PANEL_H/2, PANEL_H/2])
            offset = np.array([side * (IMPACTOR_SZ/2 + PANEL_W + 0.5), 0.0, 0.0])
            p_act  = plotter.add_mesh(panel, color="#0a1628", specular=0.5,
                                      specular_power=30, smooth_shading=True)
            imp_panels.append(p_act)
            imp_panel_offsets.append(offset)

        # ── Observers ────────────────────────────────────────────────────────
        obs_actors = []
        for i in range(3):
            body_length = OBS_SZ * 3.2
            body        = pv.Box(bounds=[-OBS_SZ/2, OBS_SZ/2, -OBS_SZ/2, OBS_SZ/2,
                                          -body_length/2, body_length/2])
            body_actor  = plotter.add_mesh(body, color=OBS_COLORS[i],
                                           specular=0.5, specular_power=20, smooth_shading=True)
            OBS_PANEL_W = OBS_SZ * 0.9
            OBS_PANEL_H = body_length * 0.60
            OBS_PANEL_T = 0.4
            panels, panel_offsets = [], []
            for side in [-1, 1]:
                panel  = pv.Box(bounds=[-OBS_PANEL_T, OBS_PANEL_T,
                                        -OBS_PANEL_W, OBS_PANEL_W,
                                        -OBS_PANEL_H/2, OBS_PANEL_H/2])
                offset = np.array([side * (OBS_SZ/2 + OBS_PANEL_W + 0.5), 0.0, 0.0])
                p_actor = plotter.add_mesh(panel, color="#0d1f3c",
                                           specular=0.55, specular_power=40, smooth_shading=True)
                panels.append(p_actor)
                panel_offsets.append(offset)
            obs_actors.append({"body": body_actor, "panels": panels, "offsets": panel_offsets})

        # ── Trails ───────────────────────────────────────────────────────────
        # Each trail: a deque of past positions + one line actor per object
        # We store actors in a list so we can remove+recreate them each frame
        def _make_trail_actor(positions, color, opacity=0.6):
            """Create a spline actor through a list of positions."""
            if len(positions) < 2:
                return None
            pts_arr = np.array(positions)
            try:
                spline = pv.Spline(pts_arr, min(len(pts_arr)*2, 60))
                return plotter.add_mesh(spline, color=color, opacity=opacity,
                                        line_width=1.5, name=None)
            except Exception:
                return None

        trail_imp  = deque(maxlen=TRAIL_LEN)
        trail_obs  = [deque(maxlen=TRAIL_LEN) for _ in range(3)]
        trail_actors_imp = [None]
        trail_actors_obs = [[None] for _ in range(3)]

        # ── Camera setup ─────────────────────────────────────────────────────
        plotter.camera_position = "iso"
        plotter.camera.zoom(0.6)
        # Store a smooth camera target we lerp toward
        cam_target_pos   = [np.array(plotter.camera.position)]
        cam_target_focal = [np.array(plotter.camera.focal_point)]

        # ── State ────────────────────────────────────────────────────────────
        particles        = []
        t                = [0.0]
        dt               = 0.35
        phase            = [0]
        imp_pos          = [IMPACTOR_START.copy().astype(float)]
        ast_pos          = [ASTEROID_START.copy().astype(float)]
        obs_pos          = [IMPACTOR_START + OBS_OFFSETS[i] for i in range(3)]
        impact_time      = [None]
        ast_inbound_vel  = _hat(INTERCEPT_PT - ASTEROID_START) * 3.8
        ast_post_vel     = [np.zeros(3)]

        # ── Auto-start recording ──────────────────────────────────────────────
        def _start_recording_after_delay():
            time.sleep(1.2)
            if not RECORDING_AVAILABLE:
                return
            try:
                win  = plotter.app_window
                x, y = win.x(), win.y()
                global recorder
                recorder = ScreenRecorder(x, y, RECORD_WIDTH, RECORD_HEIGHT)
                recorder.start()
                root.after(0, _set_recording_ui_active)
            except Exception as e:
                print(f"[EchoDART] Recording start failed: {e}")

        threading.Thread(target=_start_recording_after_delay, daemon=True).start()

        # ── Update loop ───────────────────────────────────────────────────────
        def update():
            tv = t[0]
            ph = phase[0]

            earth_actor.RotateZ(0.35)

            # ── Asteroid tumble (every phase) ─────────────────────────────
            asteroid_actor.RotateWXYZ(
                AST_SPIN_RATE,
                AST_SPIN_AXIS[0], AST_SPIN_AXIS[1], AST_SPIN_AXIS[2]
            )

            if ph < 4:
                ast_pos[0] += ast_inbound_vel * dt

            # ── Phase logic ───────────────────────────────────────────────

            if ph == 0:
                plotter.add_text("PHASE 1: LEO CHECKOUT & DEPLOYMENT",
                                 position=(0.01, 0.92), color="cyan",
                                 font_size=12, viewport=True, name="banner")
                ang = tv * 0.014
                ip  = np.array([450*np.cos(ang), 450*np.sin(ang), 80*np.sin(ang*1.6)])
                imp_pos[0] = ip
                for i in range(3):
                    obs_pos[i] = _clamp_from_earth(ip + OBS_OFFSETS[i] * 0.65)
                if tv > 280:
                    phase[0] = 1

                # Camera: wide shot showing Earth + swarm orbiting
                cam_target_pos[0]   = np.array([800, 600, 500])
                cam_target_focal[0] = np.array([0, 0, 0])

            elif ph == 1:
                plotter.add_text("PHASE 2: DETECTION & TRANSFER",
                                 position=(0.01, 0.92), color="cyan",
                                 font_size=12, viewport=True, name="banner")
                d = _hat(ast_pos[0] - imp_pos[0])
                imp_pos[0] += d * 6.5 + np.array([np.sin(tv*0.07)*5,
                                                    np.cos(tv*0.09)*4,
                                                    np.sin(tv*0.08)*4.5])
                for i in range(3):
                    off_hat = _hat(OBS_OFFSETS[i])
                    perp1   = _hat(np.cross(off_hat, np.array([0., 0., 1.])
                                            if abs(off_hat[2]) < 0.9 else np.array([1., 0., 0.])))
                    perp2   = np.cross(off_hat, perp1)
                    phase_i = i * (2 * np.pi / 3)
                    drift   = 6.0 * (np.cos(tv * 0.018 + phase_i) * perp1 +
                                     np.sin(tv * 0.018 + phase_i) * perp2)
                    obs_pos[i] = _clamp_from_earth(imp_pos[0] + OBS_OFFSETS[i] + drift)
                if np.linalg.norm(ast_pos[0] - imp_pos[0]) < 1000:
                    phase[0] = 2

                # Camera: pull back to show both Earth and incoming asteroid
                midpoint = (imp_pos[0] + ast_pos[0]) * 0.5
                cam_target_pos[0]   = midpoint + np.array([400, 300, 600])
                cam_target_focal[0] = midpoint

            elif ph == 2:
                plotter.add_text("PHASE 3: AUTONOMOUS SWARM FORMATION",
                                 position=(0.01, 0.92), color="cyan",
                                 font_size=12, viewport=True, name="banner")
                d = _hat(ast_pos[0] - imp_pos[0])
                imp_pos[0] += d * 7.8
                for i in range(3):
                    off_hat = _hat(OBS_OFFSETS[i])
                    perp1   = _hat(np.cross(off_hat, np.array([0., 0., 1.])
                                            if abs(off_hat[2]) < 0.9 else np.array([1., 0., 0.])))
                    perp2   = np.cross(off_hat, perp1)
                    phase_i = i * (2 * np.pi / 3)
                    drift   = 4.0 * (np.cos(tv * 0.022 + phase_i) * perp1 +
                                     np.sin(tv * 0.022 + phase_i) * perp2)
                    target     = imp_pos[0] + OBS_OFFSETS[i] + drift
                    obs_pos[i] = _clamp_from_earth(obs_pos[i] + (target - obs_pos[i]) * 0.28)
                if np.linalg.norm(ast_pos[0] - imp_pos[0]) < 400:
                    phase[0] = 3

                # Camera: follow impactor from behind-side, asteroid in view
                behind = imp_pos[0] - _hat(ast_pos[0] - imp_pos[0]) * 500 + np.array([0, 200, 200])
                cam_target_pos[0]   = behind
                cam_target_focal[0] = ast_pos[0]

            elif ph == 3:
                plotter.add_text("PHASE 4: TERMINAL GUIDANCE",
                                 position=(0.01, 0.92), color="cyan",
                                 font_size=12, viewport=True, name="banner")
                d = _hat(ast_pos[0] - imp_pos[0])
                imp_pos[0] += d * 10.5
                if np.linalg.norm(imp_pos[0] - ast_pos[0]) < 60:
                    phase[0]        = 4
                    impact_time[0]  = tv
                    ast_post_vel[0] = d * 0.8
                    impactor_body_actor.VisibilityOff()
                    for p in imp_panels:
                        p.VisibilityOff()
                    for _ in range(200):
                        theta = -1.3 + random.random()*2.6
                        phi   = random.random() * (np.pi / 1.3)
                        speed = 28 + random.random()*55
                        vel   = np.array([np.cos(theta)*np.sin(phi),
                                          np.sin(theta)*np.sin(phi),
                                          np.cos(phi)]) * speed + d * 25
                        pm = pv.Sphere(radius=2.5 + random.random()*3)
                        pa = plotter.add_mesh(pm, color="orange", opacity=0.95)
                        pa.SetPosition(ast_pos[0])
                        particles.append({"actor": pa, "pos": ast_pos[0].copy(),
                                          "vel": vel, "opacity": 0.95})

                # Camera: tight close-up chasing impactor toward asteroid
                behind_close = imp_pos[0] - d * 180 + np.array([0, 60, 80])
                cam_target_pos[0]   = behind_close
                cam_target_focal[0] = ast_pos[0]

            elif ph == 4:
                dt_since = tv - impact_time[0]
                if dt_since < 120:
                    plotter.add_text("IMPACT SUCCESS! (\u03b2 > 1)",
                                     position=(0.01, 0.92), color="cyan",
                                     font_size=12, viewport=True, name="banner")
                elif dt_since < 2500:
                    plotter.add_text("Phase 5: Ejecta Complete\nDATA COLLECTION IN PROGRESS...........",
                                     position=(0.01, 0.92), color="cyan",
                                     font_size=12, viewport=True, name="banner")
                else:
                    plotter.add_text("MISSION COMPLETE\nAsteroid deflected",
                                     position=(0.01, 0.92), color="cyan",
                                     font_size=12, viewport=True, name="banner")

                ast_pos[0] += ast_post_vel[0] * dt * 30
                dispersal = 1 + max(0, dt_since) / 4000 * 5
                for i in range(3):
                    offset = OBS_OFFSETS[i] * dispersal + np.array([
                        2.5*np.sin(tv*0.02 + i*2.1),
                        2.0*np.cos(tv*0.025 + i*2.8),
                        1.5*np.sin(tv*0.018 + i)
                    ])
                    target     = ast_pos[0] + offset
                    obs_pos[i] = _clamp_from_earth(obs_pos[i] + (target - obs_pos[i]) * 0.08)

                # Camera: sit between Earth and deflected asteroid so both are visible
                # Lerp the focal point toward a midpoint — Earth (origin) stays in frame
                earth_origin = np.array([0.0, 0.0, 0.0])
                mid = earth_origin * 0.45 + ast_pos[0] * 0.55
                cam_target_pos[0]   = mid + np.array([200, 150, 500])
                cam_target_focal[0] = mid

            # ── Smooth camera lerp ────────────────────────────────────────
            _lerp_cam(plotter, cam_target_pos[0], cam_target_focal[0], alpha=0.03)

            # ── Move actors ───────────────────────────────────────────────
            imp_pos[0] = _clamp_from_earth(imp_pos[0])
            impactor_body_actor.SetPosition(imp_pos[0])
            for p, offset in zip(imp_panels, imp_panel_offsets):
                p.SetPosition(imp_pos[0] + offset)
            for idx, sat in enumerate(obs_actors):
                sat["body"].SetPosition(obs_pos[idx])
                for panel, offset in zip(sat["panels"], sat["offsets"]):
                    panel.SetPosition(obs_pos[idx] + offset)
            asteroid_actor.SetPosition(ast_pos[0])

            # ── Trails ────────────────────────────────────────────────────
            # Only draw trails during phases 1-4 (not LEO checkout)
            if ph >= 1:
                trail_imp.append(imp_pos[0].copy())
                for i in range(3):
                    trail_obs[i].append(obs_pos[i].copy())

                # Impactor trail
                if trail_actors_imp[0] is not None:
                    try:
                        plotter.remove_actor(trail_actors_imp[0])
                    except Exception:
                        pass
                trail_actors_imp[0] = _make_trail_actor(list(trail_imp), "red", opacity=0.55)

                # Observer trails
                obs_trail_colors = ["orange", "lime", "magenta"]
                for i in range(3):
                    if trail_actors_obs[i][0] is not None:
                        try:
                            plotter.remove_actor(trail_actors_obs[i][0])
                        except Exception:
                            pass
                    trail_actors_obs[i][0] = _make_trail_actor(
                        list(trail_obs[i]), obs_trail_colors[i], opacity=0.4
                    )

            # ── Particles ─────────────────────────────────────────────────
            dead = []
            for p in particles:
                p["pos"]     += p["vel"] * dt
                p["opacity"] -= 0.008
                p["actor"].SetPosition(p["pos"])
                p["actor"].GetProperty().SetOpacity(p["opacity"])
                if p["opacity"] <= 0:
                    p["actor"].VisibilityOff()
                    dead.append(p)
            for p in dead:
                particles.remove(p)

            # ── HUD ───────────────────────────────────────────────────────
            dist = np.linalg.norm(imp_pos[0] - ast_pos[0])
            plotter.add_text(f"Impactor \u2192 Asteroid: {dist:.0f} km",
                             position=(0.72, 0.92), color="lime", font_size=10,
                             viewport=True, name="dist_hud")

            t[0] += dt
            plotter.update()

        plotter.add_callback(update, interval=16)
        plotter.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Stop recording
# ─────────────────────────────────────────────────────────────────────────────
def stop_recording():
    global recorder
    if recorder is None:
        messagebox.showinfo("EchoDART", "No recording in progress.")
        return
    recorder.stop()
    path = recorder.output_path
    recorder = None
    _set_recording_ui_idle()
    messagebox.showinfo("Recording saved", f"MP4 saved to:\n{path}")


def _set_recording_ui_active():
    rec_indicator.config(text="● REC", foreground="red")
    stop_btn.config(state="normal")


def _set_recording_ui_idle():
    rec_indicator.config(text="○ idle", foreground="gray")
    stop_btn.config(state="disabled")


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("EchoDART")
root.geometry("420x230")
root.resizable(False, False)

f = ttk.Frame(root, padding=30)
f.pack(expand=True)

ttk.Label(f, text="EchoDART", font=("Helvetica", 14, "bold")).pack(pady=(0, 6))
ttk.Button(f, text="🚀 Launch Simulation", command=run_simulation).pack(pady=(0, 10))

rec_frame = ttk.Frame(f)
rec_frame.pack()

rec_indicator = tk.Label(rec_frame, text="○ idle", foreground="gray",
                         font=("Helvetica", 10, "bold"))
rec_indicator.pack(side="left", padx=(0, 12))

stop_btn = ttk.Button(rec_frame, text="⏹ Stop Recording",
                      command=stop_recording, state="disabled")
stop_btn.pack(side="left")

if not RECORDING_AVAILABLE:
    ttk.Label(f, text="⚠ mss / opencv-python not installed — recording disabled",
              foreground="orange", font=("Helvetica", 8)).pack(pady=(8, 0))

root.mainloop()
