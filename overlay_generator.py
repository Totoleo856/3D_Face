import os
import cv2
import json
import numpy as np
from scipy.spatial import Delaunay
from tqdm import tqdm


def generate_overlay(video_path, output_parent_folder):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # --- Trouver le dernier dossier date ---
    date_folders = [
        d for d in os.listdir(output_parent_folder)
        if os.path.isdir(os.path.join(output_parent_folder, d))
    ]
    if not date_folders:
        raise FileNotFoundError("Aucun dossier de date trouvé.")

    date_folder = sorted(date_folders)[-1]
    json_folder = os.path.join(output_parent_folder, date_folder, "JSON")
    json_path = os.path.join(json_folder, f"{video_name}_landmarks_camera.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON non trouvé : {json_path}")

    # --- Video capture / writer ---
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    overlay_path = os.path.join(
        output_parent_folder,
        date_folder,
        f"{video_name}_overlay.mp4"
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(overlay_path, fourcc, fps, (w, h))

    # --- Charger JSON ---
    with open(json_path, "r") as f:
        data = json.load(f)

    tri = None  # cache Delaunay

    # --- Parcourir frames ---
    for idx in tqdm(range(num_frames), desc="Création Overlay", ncols=80):
        ret, frame = cap.read()
        if not ret:
            break

        overlay_frame = frame.copy()
        frame_data = data.get(str(idx), [])

        for face_data in frame_data:
            landmarks = face_data.get("landmarks_px", [])
            if not landmarks:
                continue

            pts = np.array(landmarks, dtype=np.float32)[:, :2]

            # Construire Delaunay UNE seule fois
            if tri is None:
                if len(pts) < 3:
                    continue
                try:
                    tri = Delaunay(pts)
                except Exception:
                    continue

            # Dessiner triangles
            for simplex in tri.simplices:
                pts_tri = pts[simplex].astype(np.int32)
                cv2.polylines(
                    overlay_frame,
                    [pts_tri],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=1
                )

        out_vid.write(overlay_frame)

    cap.release()
    out_vid.release()
    print(f"✔ Overlay généré : {overlay_path}")
    return overlay_path
