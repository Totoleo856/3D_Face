# 📘 Documentation Technique – Pipeline de Reconstruction Faciale 3D \& Track Camera



## 📌 1. Objectif du script



### Ce script permet :



* Le suivi temporel de 468 landmarks faciaux (MediaPipe FaceMesh).
* Le filtrage temporel (One Euro, Kalman 3D).
* L’amélioration du suivi via optical flow.
* L’estimation de la pose caméra (intrinsèques + extrinsèques) par solvePnP.
* La sauvegarde des données sous forme :

&nbsp;	- JSON (landmarks + caméra pour chaque frame)

&nbsp;	- OBJ (nuage de points triangulé par Delaunay)

* Optionnellement :

&nbsp;	- Fast mode (affichage en temps réel, sans export)

&nbsp;	- Fusion de la séquence (désactivé dans ton code)



## 📦 2. Dépendances principals



### 🎥 Vision \& géométrie



* OpenCV : optical flow (Lucas-Kanade), conversion couleurs, solvePnP.
* MediaPipe FaceMesh : 468 landmarks 3D.
* SciPy (Delaunay) : triangulation 2D pour générer la topologie 3D.
* trimesh : export des OBJ.



### 📉 Filtrage des signaux



* OneEuroFilter : lissage adaptatif.
* Kalman3D : filtre Kalman vectoriel pour stabiliser x,y,z.



### 🗂 Gestion de données



* json, os, datetime, tqdm.



### ⚙️ 3. Structure du script



#### 3.1 Fonctions principals



##### ✔ reprojection\_error()



Calcule l’erreur de reprojection entre :

* landmarks 3D (object\_points)
* projections 2D (image\_points)
* caméra (K, rvec, tvec)



→ utilise cv2.projectPoints.



##### ✔ estimate\_camera\_from\_landmarks()



Objectif : estimer la matrice de la caméra :



```

K = \[\[fx, 0, cx], \[0, fy, cy], \[0, 0, 1]]

```



Deux modes :



Mode 1 – focale imposée (focal\_mm)



fx = (focal\_mm / sensor\_width\_mm) \* frame\_width



Pas de solvePnP, retourne rvec=tvec=0.



Mode 2 – estimation automatique



Balaye une grille de focales (fx\_grid).



Pour chaque valeur → solvePnP.



Sélectionne la focale donnant la plus faible erreur de reprojection.



Option : raffinement via solvePnPRefineLM.



Sortie :



{

&nbsp;"K": \[...],

&nbsp;"rvec": \[...],

&nbsp;"tvec": \[...],

&nbsp;"rmse": ...

}



✔ apply\_filters\_to\_landmarks()



Applique pour chaque landmark :



OneEuroFilter → stabilise mouvements rapides



Kalman3D → stabilise tremblements + bruit



Sortie : array 468×3 filtré.



✔ apply\_optical\_flow()



Combine MediaPipe + Optical Flow :



Lucas-Kanade calcule la position suivante.



Compare avec la prédiction MediaPipe.



Si trop différent → remplace par MediaPipe.



→ Corrige les pertes de tracking + jitter.



✔ process\_video()



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



results\[frame\_id] = {

&nbsp;   "landmarks\_px": \[...],

&nbsp;   "camera": {...}

}



Étape 4 : Export JSON



Format :



{

&nbsp; "0": \[

&nbsp;   {

&nbsp;     "landmarks\_px": \[\[x,y,z], ...],

&nbsp;     "camera": {

&nbsp;       "K": \[...],

&nbsp;       "R": \[...],

&nbsp;       "t": \[...],

&nbsp;       "rmse\_px": ...

&nbsp;     }

&nbsp;   }

&nbsp; ]

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

video\_path	Vidéo en entrée

output\_parent\_folder	Dossier parent pour JSON + OBJ

fast\_mode	Bypass de l’export, preview temps réel

use\_one\_euro	Active filtre One Euro

one\_euro\_min\_cutoff	Cutoff du One Euro

one\_euro\_beta	Beta du One Euro

use\_kalman	Active filtre Kalman

use\_optical\_flow	Active Optical Flow

optical\_flow\_threshold	Distance max MediaPipe vs OF

focal\_mm	Focale réelle du capteur ; désactive l’estimation auto

📁 5. Organisation des fichiers générés

output/

&nbsp;  └── YYYY-MM-DD/

&nbsp;       ├── JSON/

&nbsp;       │     └── <video\_name>\_landmarks\_camera.json

&nbsp;       └── OBJ/

&nbsp;             └── <video\_name>\_frame\_0000.obj

&nbsp;             └── <video\_name>\_frame\_0001.obj

&nbsp;             └── ...



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

