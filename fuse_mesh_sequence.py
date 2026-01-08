import open3d as o3d
import numpy as np
import glob
import os

def fuse_obj_sequence(
    input_folder,
    output_path,
    samples_per_mesh=5000,
    icp_distance=5.0,
    bpa_radius=2.0,
    verbose=True
):
    """
    Fusionne une séquence d'OBJ en un mesh précis.
    - Chargement des OBJ
    - Conversion en nuages de points
    - Alignement ICP
    - Fusion intelligente
    - Reconstruction via Ball Pivoting
    """

    # ===== 1. Trouver les OBJ =====
    obj_paths = sorted(glob.glob(os.path.join(input_folder, "*.obj")))
    if len(obj_paths) == 0:
        raise FileNotFoundError("Aucun OBJ trouvé dans le dossier fourni.")

    if verbose:
        print(f"✓ {len(obj_paths)} OBJ trouvés")

    # ===== 2. Charger les meshes =====
    meshes = []
    for p in obj_paths:
        mesh = o3d.io.read_triangle_mesh(p)
        if mesh.is_empty():
            if verbose:
                print(f"⚠ Mesh vide ignoré : {p}")
            continue
        meshes.append(mesh)

    if verbose:
        print(f"✓ {len(meshes)} meshes valides chargés")

    # ===== 3. Convertir en nuages de points =====
    pcs = [m.sample_points_uniformly(samples_per_mesh) for m in meshes]

    # Normales nécessaires pour Ball Pivoting
    for i, pc in enumerate(pcs):
        pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))

    # ===== 4. Alignement ICP (incremental) =====
aligned_pcs = [pcs[0]]
prev_pc = pcs[0]

for i, pc in enumerate(pcs[1:], start=1):
    if verbose:
        print(f"→ ICP incrémental {i}/{len(pcs)-1}")

    reg = o3d.pipelines.registration.registration_icp(
        pc, prev_pc,
        max_correspondence_distance=icp_distance,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    pc.transform(reg.transformation)
    aligned_pcs.append(pc)
    prev_pc = pc

    # ===== 5. Fusionner tous les nuages =====
    merged_points = np.vstack([np.asarray(pc.points) for pc in aligned_pcs])
    merged_pc = o3d.geometry.PointCloud()
    merged_pc.points = o3d.utility.Vector3dVector(merged_points)

    if verbose:
        print(f"✓ Fusion des {len(aligned_pcs)} nuages terminée, {len(merged_points)} points au total")

    # ===== 6. Estimer les normales pour le nuage fusionné =====
    if verbose:
        print("→ Estimation des normales du nuage fusionné…")
    merged_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))

    # ===== 7. Reconstruction Ball Pivoting =====
    if verbose:
        print("→ Reconstruction du mesh via Ball Pivoting…")
    distances = merged_pc.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radii = [bpa_radius * avg_dist, bpa_radius * 2 * avg_dist]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        merged_pc, o3d.utility.DoubleVector(radii)
    )

    # ===== 8. Nettoyage =====
    if verbose:
        print("→ Nettoyage du mesh…")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # ===== 9. Sauvegarde =====
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o3d.io.write_triangle_mesh(output_path, mesh)

    if verbose:
        print(f"✓ Mesh final sauvegardé : {output_path}")

    return output_path


# ================== Test ==================
if __name__ == "__main__":
    fuse_obj_sequence(
        input_folder="samples/objs/",
        output_path="output/fused_face_bpa.obj",
        samples_per_mesh=5000,
        icp_distance=5.0,
        bpa_radius=2.0,
        verbose=True
    )
