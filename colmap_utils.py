# colmap_utils.py
import os
import cv2
import subprocess
import json
import numpy as np

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
) -> dict:
    """Lance COLMAP pour générer une reconstruction Sparse et exporter les poses en JSON.
    Args:
        frames_folder: dossier contenant les images (PNG/JPG)
        output_folder: dossier de sortie (database, sparse, json)
        colmap_path: chemin vers l'exécutable COLMAP (par défaut depuis l'env ou 'colmap')
        matcher: 'sequential' (vidéos) ou 'exhaustive'
    Returns:
        dict {frame_idx: {"K": (3x3), "R": (3x3), "t": (3,), }}
    """
    os.makedirs(output_folder, exist_ok=True)
    db_path = os.path.join(output_folder, "database.db")
    sparse_folder = os.path.join(output_folder, "sparse")
    sparse_json = os.path.join(output_folder, "sparse_json")
    os.makedirs(sparse_folder, exist_ok=True)
    os.makedirs(sparse_json, exist_ok=True)

    if colmap_path is None:
        colmap_path = os.environ.get("COLMAP_PATH", "colmap")

    # 1) Feature extraction
    try:
        subprocess.run([
            colmap_path, "feature_extractor",
            "--database_path", db_path,
            "--image_path", frames_folder,
        ], check=True)
    except FileNotFoundError:
        raise RuntimeError("COLMAP introuvable. Définis $COLMAP_PATH vers colmap.bat/exe.")

    # 2) Matching
    if matcher.lower() == "sequential":
        subprocess.run([
            colmap_path, "sequential_matcher",
            "--database_path", db_path,
        ], check=True)
    else:
        subprocess.run([
            colmap_path, "exhaustive_matcher",
            "--database_path", db_path,
        ], check=True)

    # 3) Mapper (Sparse SfM)
    subprocess.run([
        colmap_path, "mapper",
        "--database_path", db_path,
        "--image_path", frames_folder,
        "--output_path", sparse_folder,
    ], check=True)

    # 4) Convert model to JSON (handle model folder)
    model_root = os.path.join(sparse_folder, "0")
    if not os.path.isdir(model_root):
        subs = [d for d in sorted(os.listdir(sparse_folder)) if os.path.isdir(os.path.join(sparse_folder, d))]
        if not subs:
            raise RuntimeError("Aucun modèle COLMAP produit.")
        model_root = os.path.join(sparse_folder, subs[0])

    subprocess.run([
        colmap_path, "model_converter",
        "--input_path", model_root,
        "--output_path", sparse_json,
        "--output_type", "JSON",
    ], check=True)

    # 5) Charger les poses
    cameras_path = os.path.join(sparse_json, "cameras.json")
    images_path = os.path.join(sparse_json, "images.json")

    if not (os.path.isfile(cameras_path) and os.path.isfile(images_path)):
        raise FileNotFoundError("cameras.json / images.json non trouvés — échec export JSON COLMAP")

    with open(images_path, "r") as f:
        images = json.load(f)
    with open(cameras_path, "r") as f:
        cameras = json.load(f)

    frame_poses: dict[int, dict] = {}
    # Selon COLMAP, 'images' est un dict indexé par ID (string) -> {name, camera_id, R, t}
    # On dérive l'index de frame depuis le nom (frame_XXXX.ext)
    for img_id, img in images.items():
        name = img.get("name", "")
        try:
            # Nom attendu: frame_0001.png -> idx=1
            base = os.path.basename(name)
            frame_idx = int(os.path.splitext(base)[0].split("_")[1])
        except Exception:
            # Ignore si nom inattendu
            continue
        cam_id = str(img.get("camera_id"))
        cam = cameras.get(cam_id)
        if cam is None:
            continue
        K = np.array(cam.get("K"), dtype=float)
        R = np.array(img.get("R"), dtype=float)
        t = np.array(img.get("t"), dtype=float).reshape(3)
        frame_poses[frame_idx] = {"K": K, "R": R, "t": t}

    return frame_poses
