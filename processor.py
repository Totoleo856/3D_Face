import os
import cv2
import mediapipe as mp
import numpy as np
import trimesh
from tqdm import tqdm
import json
from scipy.spatial import Delaunay
from datetime import datetime
from one_euro_filter import OneEuroFilter
from kalman_filter import Kalman3D

# ================= Camera utils =================
def reprojection_error(object_points, image_points, rvec, tvec, K, dist_coef=None):
    if dist_coef is None:
        dist_coef = np.zeros((4,1))
    proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coef)
    proj = proj.reshape(-1,2)
    img = image_points.reshape(-1,2)
    diff = img - proj
    return float(np.sqrt(np.mean(np.sum(diff*diff, axis=1))), diff)

def estimate_camera_from_landmarks(landmarks, frame_width, frame_height, fx_grid=(0.6,1.8,30), refine=True):
    pts = np.array(landmarks, dtype=np.float64)
    if pts.shape[0] < 6:
        return None
    image_points = pts[:, :2].astype(np.float64)
    object_points = pts.copy().astype(np.float64)

    min_r, max_r, n_steps = fx_grid
    ratios = np.linspace(min_r, max_r, int(n_steps))
    best = None
    best_rmse = 1e9

    for r in ratios:
        fx = frame_width * float(r)
        fy = fx
        cx = frame_width/2.0
        cy = frame_height/2.0
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
        try:
            ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            rmse, _ = reprojection_error(object_points, image_points, rvec, tvec, K)
            if rmse < best_rmse:
                best_rmse = rmse
                best = (K.copy(), rvec.copy(), tvec.copy(), rmse)
        except:
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
        except:
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

# ================= Main processor =================
def process_video_with_callback(
    video_path,
    output_parent_folder,
    progress_callback=None,
    fast_mode=False,
    use_one_euro=True,
    one_euro_min_cutoff=1.0,
    one_euro_beta=0.005,
    use_kalman=True
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
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    results = {}
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # --- Initialize filters ---
    one_euro_filters = None
    if use_one_euro:
        one_euro_filters = [
            [OneEuroFilter(freq=fps, min_cutoff=one_euro_min_cutoff, beta=one_euro_beta) for _ in range(3)]
            for _ in range(468)
        ]

    kalman_filters = [Kalman3D() for _ in range(468)] if use_kalman else None

    # --- Frame loop ---
    for idx in tqdm(range(num_frames), desc="Traitement vidéo", ncols=80, leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        out = face_mesh.process(rgb)
        results[idx] = []

        if out.multi_face_landmarks:
            face = out.multi_face_landmarks[0]
            landmarks = apply_filters_to_landmarks(face.landmark, w, h, one_euro_filters, kalman_filters)

            if fast_mode:
                preview_frame = frame.copy()
                for lm in landmarks:
                    cv2.circle(preview_frame, (int(lm[0]), int(lm[1])), 2, (0,255,0), -1)
                cv2.imshow("FAST Preview", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                results[idx].append({"landmarks_px": landmarks.tolist(), "camera": None})
                if progress_callback:
                    progress_callback((idx+1)/num_frames*100)
                continue  # skip OBJ/JSON in fast mode

            # Normal mode
            pts = np.array(landmarks)
            if pts.shape[0] >= 3:
                xy = pts[:, :2]
                faces = Delaunay(xy).simplices
                mesh = trimesh.Trimesh(vertices=pts, faces=faces, process=True)
                mesh.invert()
                fpath = os.path.join(obj_folder, f"{video_name}_frame_{idx:04d}.obj")
                mesh.export(fpath)

            cam = estimate_camera_from_landmarks(landmarks, w, h)
            cam_data = None
            if cam is not None:
                Rmat, _ = cv2.Rodrigues(cam["rvec"])
                cam_data = {
                    "K": cam["K"].tolist(),
                    "R": Rmat.tolist(),
                    "t": cam["tvec"].reshape(3).tolist(),
                    "rmse_px": cam["rmse"]
                }
            results[idx].append({"landmarks_px": landmarks.tolist(), "camera": cam_data})
        else:
            results[idx].append({"landmarks_px": [], "camera": None})

        if progress_callback:
            progress_callback((idx+1)/num_frames*100)

    cap.release()
    if fast_mode:
        cv2.destroyAllWindows()

    # Save JSON & Blender script only in normal mode
    if not fast_mode:
        json_path = os.path.join(json_folder, f"{video_name}_landmarks_camera.json")
        with open(json_path, "w") as f:
            json.dump(results, f)

        obj_files = sorted([f for f in os.listdir(obj_folder) if f.endswith(".obj")])
        blender_script = os.path.join(output_parent_folder, date_folder, f"{video_name}_blender_anim.py")
        blender_script_content = f"""
import bpy
import os

obj_folder = r"{obj_folder}"
output_fbx = r"{os.path.join(output_parent_folder, date_folder, video_name + '.fbx')}"

bpy.ops.wm.read_factory_settings(use_empty=True)

obj_files = {obj_files}

for i, obj_file in enumerate(obj_files):
    bpy.ops.import_scene.obj(filepath=os.path.join(obj_folder, obj_file))
    obj = bpy.context.selected_objects[0]
    obj.keyframe_insert(data_path="location", frame=i+1)
    obj.keyframe_insert(data_path="scale", frame=i+1)
    obj.keyframe_insert(data_path="rotation_euler", frame=i+1)

bpy.ops.export_scene.fbx(filepath=output_fbx, bake_space_transform=True)
"""
        with open(blender_script, "w") as f:
            f.write(blender_script_content)
