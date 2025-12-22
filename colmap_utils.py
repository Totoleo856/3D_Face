# Colmap_utils.py

import os
import cv2
import subprocess
import json
import numpy as np

def extract_frames_from_video(video_path, frames_folder, step=1):
    """Extrait les images d'une vidéo pour COLMAP."""
    os.makedirs(frames_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            cv2.imwrite(os.path.join(frames_folder, f"frame_{idx:04d}.png"), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved

def run_colmap_pipeline(frames_folder, output_folder, colmap_path="colmap"):
    """Lance COLMAP pour générer la reconstruction et les poses JSON."""
    os.makedirs(output_folder, exist_ok=True)
    db_path = os.path.join(output_folder, "database.db")
    sparse_folder = os.path.join(output_folder, "sparse")
    sparse_json = os.path.join(output_folder, "sparse_json")
    
    os.makedirs(sparse_folder, exist_ok=True)
    os.makedirs(sparse_json, exist_ok=True)
    
    # Feature extraction
    subprocess.run([
        colmap_path, "feature_extractor",
        "--database_path", db_path,
        "--image_path", frames_folder
    ], check=True)
    
    # Feature matching
    subprocess.run([
        colmap_path, "exhaustive_matcher",
        "--database_path", db_path
    ], check=True)
    
    # Structure-from-Motion
    subprocess.run([
        colmap_path, "mapper",
        "--database_path", db_path,
        "--image_path", frames_folder,
        "--output_path", sparse_folder
    ], check=True)
    
    # Convert to JSON
    subprocess.run([
        colmap_path, "model_converter",
        "--input_path", os.path.join(sparse_folder, "0"),
        "--output_path", sparse_json,
        "--output_type", "JSON"
    ], check=True)
    
    # Récupération des poses
    frame_poses = {}
    cameras_path = os.path.join(sparse_json, "cameras.json")
    images_path = os.path.join(sparse_json, "images.json")
    
    with open(images_path, "r") as f:
        images = json.load(f)
    with open(cameras_path, "r") as f:
        cameras = json.load(f)
    
    for img in images.values():
        frame_idx = int(img["name"].split("_")[1].split(".")[0])
        cam_id = img["camera_id"]
        K = np.array(cameras[str(cam_id)]["K"])
        R = np.array(img["R"])
        t = np.array(img["t"])
        frame_poses[frame_idx] = {"K": K, "R": R, "t": t}
    
    return frame_poses
