# colmap_utils.py (TXT-based, robust)
import os
import cv2
import subprocess
import json
import numpy as np
from typing import Dict, Tuple

# --- Helpers to parse COLMAP TXT outputs ---
# Format reference: cameras.txt / images.txt in COLMAP Output Format docs.
# (IDs, models and parameters as documented)

def _read_cameras_txt(path: str) -> Dict[str, dict]:
    cams = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # cameras.txt: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS...
            cam_id = parts[0]
            model = parts[1]
            width = int(parts[2]); height = int(parts[3])
            params = list(map(float, parts[4:]))
            # Build K depending on model
            if model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
                # fx, fy, cx, cy first
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE", "THIN_PRISM_FISHEYE", "FOV"):
                # first param is focal (fx=fy), then cx, cy
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                # fallback: simple pinhole assumption
                fx = fy = params[0] if params else max(width, height)
                cx, cy = (params[1], params[2]) if len(params) >= 3 else (width/2.0, height/2.0)
            K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]], dtype=float)
            cams[cam_id] = {"model": model, "width": width, "height": height, "params": params, "K": K}
    return cams


def _read_images_txt(path: str) -> Dict[str, dict]:
    imgs = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # images.txt first (pose) line:
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            # followed by points line we ignore here.
            if len(parts) < 10:
                continue
            img_id = parts[0]
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            cam_id = parts[8]
            name = ' '.join(parts[9:])  # file name may contain spaces
            # Convert quaternion to rotation matrix (world->camera)
            q = np.array([qw, qx, qy, qz], dtype=float)
            # Normalize quaternion
            n = np.linalg.norm(q)
            if n > 0: q = q / n
            qw, qx, qy, qz = q
            # Rotation matrix from quaternion (w,x,y,z)
            R = np.array([
                [1-2*(qy*qy+qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)]
            ], dtype=float)
            t = np.array([tx, ty, tz], dtype=float)
            imgs[img_id] = {"cam_id": cam_id, "R": R, "t": t, "name": name}
    return imgs


def extract_frames_from_video(video_path: str, frames_folder: str, step: int = 1) -> int:
    """Extrait des images d'une vidéo pour COLMAP.
    Args:
        video_path: chemin de la vidéo source
        frames_folder: dossier où enregistrer les images extraites
        step: intervalle d'échantillonnage (1 = toutes les frames)
    Returns:
        Nombre d'images enregistrées.
    """
    os.makedirs(frames_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    saved = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % max(1, int(step)) == 0:
                out_path = os.path.join(frames_folder, f"frame_{idx:04d}.png")
                cv2.imwrite(out_path, frame)
                saved += 1
            idx += 1
    finally:
        cap.release()
    return saved


def run_colmap_pipeline(
    frames_folder: str,
    output_folder: str,
    colmap_path: str | None = None,
    matcher: str = "sequential",
    relax_params: bool = True,
) -> dict:
    """Lance COLMAP (SfM) et retourne un dict de poses par frame.
    - Utilise TXT (pas JSON) pour compatibilité modèle_converter.
    - Paramètres 'relaxés' pour vidéos si relax_params=True.
    Returns: {frame_idx: {"K":(3x3), "R":(3x3), "t":(3,)}}
    """
    os.makedirs(output_folder, exist_ok=True)
    db_path = os.path.join(output_folder, "database.db")
    sparse_folder = os.path.join(output_folder, "sparse")
    sparse_txt = os.path.join(output_folder, "sparse_txt")
    os.makedirs(sparse_folder, exist_ok=True)
    os.makedirs(sparse_txt, exist_ok=True)

    if colmap_path is None:
        colmap_path = os.environ.get("COLMAP_PATH", "colmap")

    # 1) Feature extraction (plus de features pour robustesse)
    try:
        cmd = [
            colmap_path, "feature_extractor",
            "--database_path", db_path,
            "--image_path", frames_folder,
            "--SiftExtraction.max_num_features", "8000",
            "--SiftExtraction.max_image_size", "4000",
            "--SiftExtraction.use_gpu", "1",
        ]
        # Vidéo: une seule caméra (intrinsics partagés) peut aider
        if relax_params:
            cmd += ["--ImageReader.single_camera", "1"]
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise RuntimeError("COLMAP introuvable. Définis $COLMAP_PATH vers colmap.bat/exe.")

    # 2) Matching

    if matcher.lower() == "sequential":
        cmd = [colmap_path, "sequential_matcher", "--database_path", db_path]
        subprocess.run(cmd, check=True)
    else:
        cmd = [colmap_path, "exhaustive_matcher", "--database_path", db_path]
        subprocess.run(cmd, check=True)


    # 3) Mapper (Sparse SfM)
    cmd = [
        colmap_path, "mapper",
        "--database_path", db_path,
        "--image_path", frames_folder,
        "--output_path", sparse_folder,
    ]
    if relax_params:
        cmd += [
            "--Mapper.init_min_tri_angle", "2",        # default ~16
            "--Mapper.init_min_num_inliers", "4",      # default 100
            "--Mapper.abs_pose_min_num_inliers", "3",  # default 30
            "--Mapper.abs_pose_max_error", "8",        # default 12
        ]
    subprocess.run(cmd, check=True)

    # 4) Convert model to TXT (not JSON)
    # Choose model directory (usually sparse/0)
    model_root = os.path.join(sparse_folder, "0")
    if not os.path.isdir(model_root):
        subs = [d for d in sorted(os.listdir(sparse_folder)) if os.path.isdir(os.path.join(sparse_folder, d))]
        if not subs:
            # reconstruction failed: return empty
            return {}
        model_root = os.path.join(sparse_folder, subs[0])

    subprocess.run([
        colmap_path, "model_converter",
        "--input_path", model_root,
        "--output_path", sparse_txt,
        "--output_type", "TXT",
    ], check=True)

    # 5) Charger TXT et produire poses
    cameras_path = os.path.join(sparse_txt, "cameras.txt")
    images_path = os.path.join(sparse_txt, "images.txt")
    if not (os.path.isfile(cameras_path) and os.path.isfile(images_path)):
        return {}

    cams = _read_cameras_txt(cameras_path)
    imgs = _read_images_txt(images_path)

    frame_poses: dict[int, dict] = {}
    # map per image using file name pattern frame_XXXX.*
    for img in imgs.values():
        name = os.path.basename(img["name"]).replace('\\', '/').split('/')[-1]
        try:
            base = os.path.splitext(name)[0]
            # look for trailing integer in base (frame_0001)
            if '_' in base:
                idx = int(base.split('_')[-1])
            else:
                # fallback: try full base as integer
                idx = int(base)
        except Exception:
            continue
        cam = cams.get(img["cam_id"]) or next(iter(cams.values()))
        K = cam["K"]
        R = img["R"]
        t = img["t"]
        frame_poses[idx] = {"K": K, "R": R, "t": t}

    return frame_poses
