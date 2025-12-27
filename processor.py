import os
import cv2
import mediapipe as mp
import numpy as np
import trimesh
from tqdm import tqdm
import json
from datetime import datetime
from one_euro_filter import OneEuroFilter
from kalman_filter import Kalman3D
import subprocess
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
from pathlib import Path
import shutil

# ================= Camera utils =================
def reprojection_error(object_points, image_points, rvec, tvec, K, dist_coef=None):
    if dist_coef is None:
        dist_coef = np.zeros((4, 1))
    proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coef)
    proj = proj.reshape(-1, 2)
    img = image_points.reshape(-1, 2)
    diff = img - proj
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))), diff


def estimate_camera_from_landmarks(
    landmarks,
    frame_width,
    frame_height,
    fx_grid=(0.6, 1.8, 30),
    refine=True,
    focal_mm=None,
    sensor_width_mm=36,
):
    pts = np.array(landmarks, dtype=np.float64)
    if pts.shape[0] < 6 and focal_mm is None:
        return None
    image_points = pts[:, :2].astype(np.float64)
    object_points = pts.copy().astype(np.float64)

    # Known focal: compute extrinsics
    if focal_mm is not None:
        fx = (focal_mm / sensor_width_mm) * frame_width
        fy = fx
        cx = frame_width / 2.0
        cy = frame_height / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        try:
            ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                rmse, _ = reprojection_error(object_points, image_points, rvec, tvec, K)
                return {"K": K, "rvec": rvec, "tvec": tvec, "rmse": float(rmse)}
        except Exception:
            pass
        return None

    # Grid search on focal
    min_r, max_r, n_steps = fx_grid
    ratios = np.linspace(min_r, max_r, int(n_steps))
    best = None
    best_rmse = 1e9
    for r in ratios:
        fx = frame_width * float(r)
        fy = fx
        cx = frame_width / 2.0
        cy = frame_height / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        try:
            ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            rmse, _ = reprojection_error(object_points, image_points, rvec, tvec, K)
            if rmse < best_rmse:
                best_rmse = rmse
                best = (K.copy(), rvec.copy(), tvec.copy(), rmse)
        except Exception:
            continue
    if best is None:
        return None
    K, rvec, tvec, rmse = best

    if refine:
        try:
            rvec2, tvec2 = cv2.solvePnPRefineLM(object_points, image_points, K, None, rvec, tvec)
            rmse2, _ = reprojection_error(object_points, image_points, rvec2, tvec2, K)
            if rmse2 < rmse:
                rvec, tvec, rmse = rvec2, tvec2, rmse2
        except Exception:
            pass
    return {"K": K, "rvec": rvec, "tvec": tvec, "rmse": float(rmse)}


# ================= Filter helper =================
def apply_filters_to_landmarks(raw_landmarks, w, h, one_euro_filters=None, kalman_filters=None):
    filtered = []
    for pid, lm in enumerate(raw_landmarks):
        x = lm.x * w
        y = lm.y * h
        z = lm.z * w
        if one_euro_filters is not None and pid < len(one_euro_filters):
            x = one_euro_filters[pid][0](x)
            y = one_euro_filters[pid][1](y)
            z = one_euro_filters[pid][2](z)
        if kalman_filters is not None and pid < len(kalman_filters):
            x, y, z = kalman_filters[pid].apply(np.array([x, y, z]))
        filtered.append([x, y, z])
    return np.array(filtered)


# ================= Optical Flow helper =================
def apply_optical_flow(prev_gray, gray, prev_points, mp_points=None, threshold_px=5.0):
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_points, None,
        winSize=(15, 15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    next_points = next_points.reshape(-1, 2)
    mask = status.reshape(-1).astype(bool)
    if mp_points is not None:
        mp_points = mp_points.reshape(-1, 2)
        # replace failed tracks with MediaPipe points
        next_points[~mask] = mp_points[~mask]
        # replace if deviation too large
        diff = np.linalg.norm(next_points - mp_points, axis=1)
        replace = (~mask) | (diff > threshold_px)
        next_points[replace] = mp_points[replace]
    return next_points


# ================= Build triangles from Mediapipe tessellation =================
def build_triangles_from_tesselation(tesselation):
    adjacency = {}
    for (i, j) in tesselation:
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)
    triangles = set()
    for i in adjacency:
        for j in adjacency[i]:
            for k in adjacency[i]:
                if j < k and j in adjacency[k]:
                    tri = tuple(sorted([i, j, k]))
                    triangles.add(tri)
    return np.array(list(triangles), dtype=np.int32)


# ================= Main processor =================
def process_video(
    video_path,
    output_parent_folder,
    progress_callback=None,
    fast_mode=False,
    use_one_euro=False,
    one_euro_min_cutoff=1.0,
    one_euro_beta=0.005,
    use_kalman=False,
    use_optical_flow=False,
    optical_flow_threshold=5.0,
    focal_mm=None,
    # ---- COLMAP options ----
    use_colmap=True,
    colmap_step=1,
    colmap_matcher="sequential",
    SKIP_INITIAL_FRAMES=5,
):
    date_folder = datetime.now().strftime("%Y-%m-%d")
    obj_folder = os.path.join(output_parent_folder, date_folder, "OBJ")
    json_folder = os.path.join(output_parent_folder, date_folder, "JSON")
    os.makedirs(obj_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or (isinstance(fps, float) and (np.isnan(fps) or fps <= 1)):
        fps = 30.0
    prev_msec = None

    results = {}
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    one_euro_filters = None
    if use_one_euro:
        one_euro_filters = [
            [OneEuroFilter(freq=fps, min_cutoff=one_euro_min_cutoff, beta=one_euro_beta) for _ in range(3)]
            for _ in range(468)
        ]
    kalman_filters = [Kalman3D() for _ in range(468)] if use_kalman else None

    prev_gray = None
    prev_points = None

    # Build Mediapipe faces once
    faces_mp = build_triangles_from_tesselation(FACEMESH_TESSELATION)

    # ---- COLMAP (optionnel) : extraction frames + SfM ----
    colmap_poses = {}
    if use_colmap:
        # detect COLMAP binary
        colmap_exec = os.environ.get("COLMAP_PATH", "colmap")
        if not (shutil.which(colmap_exec) or os.path.isfile(colmap_exec)):
            raise RuntimeError("COLMAP introuvable. Définis $COLMAP_PATH vers colmap.bat/exe.")
        from colmap_utils import extract_frames_from_video, run_colmap_pipeline
        frames_folder = os.path.join(output_parent_folder, date_folder, "FRAMES")
        saved = extract_frames_from_video(video_path, frames_folder, step=colmap_step)
        colmap_out = os.path.join(output_parent_folder, date_folder, "COLMAP")
        colmap_poses = run_colmap_pipeline(
            frames_folder,
            colmap_out,
            colmap_path=colmap_exec,
            matcher=colmap_matcher,
        )

    # --- Frame loop ---
    for idx in tqdm(range(num_frames), desc="Traitement vidéo", ncols=80, leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        out = face_mesh.process(rgb)

        # --- dynamic dt based on POS_MSEC (fallback to fps)
        curr_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if prev_msec is not None and np.isfinite(curr_msec):
            dt = max(1e-6, (curr_msec - prev_msec) * 1e-3)
        else:
            curr_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            dt = 1.0 / fps if idx == 0 else max(1e-6, (curr_idx - (idx - 1)) / fps)
        prev_msec = curr_msec

        # --- OneEuro: adjust frequency (1/dt)
        if use_one_euro and one_euro_filters is not None:
            freq_dyn = 1.0 / dt
            for triplet in one_euro_filters:
                triplet[0].freq = freq_dyn
                triplet[1].freq = freq_dyn
                triplet[2].freq = freq_dyn

        # --- Optical flow: dynamic threshold (bounded)
        thresh_dyn = np.clip(optical_flow_threshold * (dt * fps), 0.5 * optical_flow_threshold, 2.0 * optical_flow_threshold)

        results[idx] = []
        if out.multi_face_landmarks:
            face = out.multi_face_landmarks[0]
            landmarks_raw = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in face.landmark], dtype=np.float32)

            # --- Skip initial frames ---
            if idx < SKIP_INITIAL_FRAMES:
                results[idx].append({"landmarks_px": landmarks_raw.tolist(), "camera": None})
                prev_gray = gray.copy()
                prev_points = landmarks_raw.copy()
                if progress_callback:
                    progress_callback((idx + 1) / num_frames * 100)
                continue

            if use_optical_flow and prev_gray is not None and prev_points is not None:
                landmarks_raw[:, :2] = apply_optical_flow(
                    prev_gray,
                    gray,
                    prev_points[:, :2].astype(np.float32),
                    landmarks_raw[:, :2].astype(np.float32),
                    threshold_px=thresh_dyn,
                )
            prev_gray = gray.copy()
            prev_points = landmarks_raw.copy()

            # Normalize for OneEuro (MediaPipe-like)
            class DummyLM:
                def __init__(self, x, y, z, w, h):
                    self.x = x / w
                    self.y = y / h
                    self.z = z / w

            dummy_lms = [DummyLM(pt[0], pt[1], pt[2], w, h) for pt in landmarks_raw]

            landmarks_filtered = apply_filters_to_landmarks(
                dummy_lms,
                w,
                h,
                one_euro_filters if use_one_euro else None,
                kalman_filters if use_kalman else None,
            )

            if fast_mode:
                preview_frame = frame.copy()
                for lm in landmarks_filtered:
                    if not np.all(np.isfinite(lm[:2])):
                        continue
                    cv2.circle(preview_frame, (int(lm[0]), int(lm[1])), 2, (0, 255, 0), -1)
                cv2.imshow("FAST Preview", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                results[idx].append({"landmarks_px": landmarks_filtered.tolist(), "camera": None})
                if progress_callback:
                    progress_callback((idx + 1) / num_frames * 100)
                continue

            # ---- Camera pose: COLMAP first ----
            if use_colmap and idx in colmap_poses:
                pose = colmap_poses[idx]
                cam_data = {
                    "K": pose["K"].tolist(),
                    "R": pose["R"].tolist(),
                    "t": pose["t"].reshape(3).tolist(),
                    "rmse_px": None,
                }
            else:
                cam = estimate_camera_from_landmarks(landmarks_filtered, w, h, focal_mm=focal_mm)
                cam_data = None
                if cam is not None:
                    Rmat, _ = cv2.Rodrigues(cam["rvec"])
                    cam_data = {
                        "K": cam["K"].tolist(),
                        "R": Rmat.tolist(),
                        "t": cam["tvec"].reshape(3).tolist(),
                        "rmse_px": cam["rmse"],
                    }
            results[idx].append({"landmarks_px": landmarks_filtered.tolist(), "camera": cam_data})
        else:
            results[idx].append({"landmarks_px": [], "camera": None})

        if progress_callback:
            progress_callback((idx + 1) / num_frames * 100)

    cap.release()
    cv2.destroyAllWindows()
    if fast_mode:
        return

    # ========== SAVE JSON ==========
    json_path = os.path.join(json_folder, f"{video_name}_landmarks_camera.json")
    with open(json_path, "w") as f:
        json.dump(results, f)

    # ========== GENERATE OBJ (Mediapipe faces) ==========
    for frame_id, frame_data in tqdm(results.items(), desc="Création OBJ", ncols=80):
        rec = next((r for r in reversed(frame_data) if r["landmarks_px"]), None)
        if not rec:
            continue
        pts = np.array(rec["landmarks_px"], dtype=np.float32)
        mesh = trimesh.Trimesh(
            vertices=pts,
            faces=faces_mp,
            process=False,
        )
        fpath = os.path.join(obj_folder, f"{video_name}_frame_{frame_id:04d}.obj")
        mesh.export(fpath)
    print("✓ Traitement terminé et OBJ générés")

    # ========== EXPORT FBX (via Blender) ==========
    try:
        global_scale=0.01,
        apply_unit_scale=True,
        obj_folder = os.path.join(output_parent_folder, date_folder, "OBJ")
        fbx_path = os.path.join(output_parent_folder, date_folder, f"{video_name}_animated.fbx")
        print("→ Export FBX via Blender…")
        blender_exec = os.environ.get("BLENDER_PATH")
        if not blender_exec:
            candidates = [
                r"C:\\Program Files\\Blender Foundation\\Blender 4.4\\blender.exe",
                r"C:\\Program Files\\Blender Foundation\\Blender 4.3\\blender.exe",
                r"C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe",
                r"C:\\Program Files\\Blender Foundation\\Blender 4.1\\blender.exe",
            ]
            blender_exec = next((p for p in candidates if os.path.isfile(p)), "blender")
        script_abs = str(Path(__file__).parent / "export_fbx.py")
        subprocess.run([
            blender_exec,
            "--background",
            "--python", script_abs,
            "--",
            "--obj_folder", obj_folder,
            "--fbx_path", fbx_path,
            "--fps", str(int(fps)),
        ], check=True)
        print("✓ FBX animé exporté :", fbx_path)
    except Exception as e:
        print("⚠ Erreur export FBX :", e)
