# 3D_Face
Video analyse for 3D model extraction.

📘 Documentation Technique – Pipeline de Reconstruction Faciale 3D & Track Camera
📌 1. Objectif du script

Ce script permet :

Le suivi temporel de 468 landmarks faciaux (MediaPipe FaceMesh).

Le filtrage temporel (One Euro, Kalman 3D).

L’amélioration du suivi via optical flow.

L’estimation de la pose caméra (intrinsèques + extrinsèques) par solvePnP.

La sauvegarde des données sous forme :

JSON (landmarks + caméra pour chaque frame)

OBJ (nuage de points triangulé par Delaunay)

Optionnellement :

Fast mode (affichage en temps réel, sans export)

Fusion de la séquence (désactivé dans ton code)

📦 2. Dépendances principales
🎥 Vision & géométrie

OpenCV : optical flow (Lucas-Kanade), conversion couleurs, solvePnP.

MediaPipe FaceMesh : 468 landmarks 3D.

SciPy (Delaunay) : triangulation 2D pour générer la topologie 3D.

trimesh : export des OBJ.

📉 Filtrage des signaux

OneEuroFilter : lissage adaptatif.

Kalman3D : filtre Kalman vectoriel pour stabiliser x,y,z.

🗂 Gestion de données

json, os, datetime, tqdm.

⚙️ 3. Structure du script
3.1 Fonctions principales
✔ reprojection_error()

Calcule l’erreur de reprojection entre :

landmarks 3D (object_points)

projections 2D (image_points)

caméra (K, rvec, tvec)

→ utilise cv2.projectPoints.

✔ estimate_camera_from_landmarks()

Objectif : estimer la matrice de la caméra :

K = [[fx, 0, cx],
     [0, fy, cy],
     [0, 0, 1]]


Deux modes :

Mode 1 – focale imposée (focal_mm)

fx = (focal_mm / sensor_width_mm) * frame_width

Pas de solvePnP, retourne rvec=tvec=0.

Mode 2 – estimation automatique

Balaye une grille de focales (fx_grid).

Pour chaque valeur → solvePnP.

Sélectionne la focale donnant la plus faible erreur de reprojection.

Option : raffinement via solvePnPRefineLM.

Sortie :

{
 "K": [...],
 "rvec": [...],
 "tvec": [...],
 "rmse": ...
}

✔ apply_filters_to_landmarks()

Applique pour chaque landmark :

OneEuroFilter → stabilise mouvements rapides

Kalman3D → stabilise tremblements + bruit

Sortie : array 468×3 filtré.

✔ apply_optical_flow()

Combine MediaPipe + Optical Flow :

Lucas-Kanade calcule la position suivante.

Compare avec la prédiction MediaPipe.

Si trop différent → remplace par MediaPipe.

→ Corrige les pertes de tracking + jitter.

✔ process_video()

Le cœur du pipeline.
Responsable de :

capture vidéo

FaceMesh

optical flow + filtering

estimation caméra

export JSON + OBJ

3.2 Pipeline général
Étape 1 : Chargement vidéo

Ouverture via cv2.VideoCapture

Lecture du FPS et nombre de frames

Étape 2 : Initialisation

MediaPipe FaceMesh (1 seul visage)

Filtres (si activés)

Variables Optical Flow

Étape 3 : Boucle de traitement frame par frame

Pour chaque frame :

1 ⟶ Détection MediaPipe

Si visage trouvé → 468 points 3D (x,y,z).

2 ⟶ Correction par Optical Flow (optionnelle)

Améliore la stabilité.

3 ⟶ Filtrage temporel (optionnel)

OneEuroFilter

Kalman3D

Résultat : landmarks fiables + stabilisés.

4 ⟶ Mode Fast

Affiche preview

Pas d’enregistrement

5 ⟶ Estimation de la caméra

solvePnP / solvePnPRefineLM

retourne K, R, t, rmse

6 ⟶ Stockage des résultats

Dans une structure Python :

results[frame_id] = {
    "landmarks_px": [...],
    "camera": {...}
}

Étape 4 : Export JSON

Format :

{
  "0": [
    {
      "landmarks_px": [[x,y,z], ...],
      "camera": {
        "K": [...],
        "R": [...],
        "t": [...],
        "rmse_px": ...
      }
    }
  ]
}

Étape 5 : Export OBJ (mesh 3D)

Pour chaque frame :

Prend les landmarks filtrés

Triangule le plan 2D (x,y) → Delaunay

Utilise les z comme profondeur

Export .obj via trimesh

→ Génère un mesh par frame.

🎚 4. Paramètres du script
Paramètre	Description
video_path	Vidéo en entrée
output_parent_folder	Dossier parent pour JSON + OBJ
fast_mode	Bypass de l’export, preview temps réel
use_one_euro	Active filtre One Euro
one_euro_min_cutoff	Cutoff du One Euro
one_euro_beta	Beta du One Euro
use_kalman	Active filtre Kalman
use_optical_flow	Active Optical Flow
optical_flow_threshold	Distance max MediaPipe vs OF
focal_mm	Focale réelle du capteur ; désactive l’estimation auto
📁 5. Organisation des fichiers générés
output/
   └── YYYY-MM-DD/
        ├── JSON/
        │     └── <video_name>_landmarks_camera.json
        └── OBJ/
              └── <video_name>_frame_0000.obj
              └── <video_name>_frame_0001.obj
              └── ...

📌 6. Points forts
✔ Très robuste :

Optical Flow + MediaPipe

OneEuro + Kalman

solvePnP + refinement

✔ Sorties complètes :

Landmarks 3D stabilisés

Pose caméra

Mesh 3D frame-par-frame

JSON structuré

✔ Architecture claire, modulaire et maintenable
📌 7. Points d’amélioration possibles

Si tu veux, je peux t’aider à :

⭐ Optimiser les performances
⭐ Ajouter un mesh template réanimé (morph targets)
⭐ Faire une fusion 3D plus propre
⭐ Générer une vidéo overlay
⭐ Export .fbx ou .gltf