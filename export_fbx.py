import bpy
import sys
import os
import re

# ------------------------------
# Parse arguments passed after "--"
# ------------------------------
avrg = sys.argv
argv = avrg[avrg.index("--") + 1:] if "--" in avrg else []
obj_folder = None
fbx_path = None
fps = 30

# Parse parameters
for i in range(len(argv)):
    if argv[i] == "--obj_folder":
        obj_folder = argv[i + 1]
    elif argv[i] == "--fbx_path":
        fbx_path = argv[i + 1]
    elif argv[i] == "--fps":
        fps = int(argv[i + 1])

if obj_folder is None or fbx_path is None:
    print("❌ ERROR: Missing arguments --obj_folder or --fbx_path")
    sys.exit(1)
print("OBJ folder :", obj_folder)
print("Output FBX :", fbx_path)

# Ensure output directory exists
out_dir = os.path.dirname(fbx_path)
if out_dir and not os.path.isdir(out_dir):
    os.makedirs(out_dir, exist_ok=True)

# ------------------------------
# Clean scene
# ------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)

# ------------------------------
# Collect OBJ files
# ------------------------------

def _natkey(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]

obj_files = [f for f in os.listdir(obj_folder) if f.lower().endswith(".obj")]
obj_files = sorted(obj_files, key=_natkey)
obj_files = [os.path.join(obj_folder, f) for f in obj_files]
if not obj_files:
    print("❌ ERROR: No OBJ files found.")
    sys.exit(1)
print(f"Found {len(obj_files)} OBJ frames")

# ------------------------------
# Import first OBJ (rest pose)
# ------------------------------
print("Importing base mesh:", obj_files[0])
bpy.ops.wm.obj_import(filepath=obj_files[0])
sel = bpy.context.selected_objects
if not sel:
    print(f"❌ No object imported from: {obj_files[0]}")
    sys.exit(1)
obj = sel[0]
obj.name = "FaceMesh"

# ------------------------------
# Create shape keys
# ------------------------------
obj.shape_key_add(name="Basis")  # base
bpy.context.view_layer.objects.active = obj

for i, filepath in enumerate(obj_files[1:], start=1):
    print(f"Importing shape {i}: {filepath}")
    # Import frame
    bpy.ops.wm.obj_import(filepath=filepath)
    imported = bpy.context.selected_objects[0]
    # Safety: same vertex count
    if len(imported.data.vertices) != len(obj.data.vertices):
        print(f"❌ ERROR: Vertex count mismatch on {filepath}")
        bpy.data.objects.remove(imported, do_unlink=True)
        sys.exit(1)
    # Copy vertices into a new shape key
    key = obj.shape_key_add(name=f"frame_{i:04d}")
    for v_main, v_imp in zip(obj.data.vertices, imported.data.vertices):
        key.data[v_main.index].co = v_imp.co
    # Delete imported temp mesh
    bpy.data.objects.remove(imported, do_unlink=True)
print("✔ Shape keys generated.")

# ------------------------------
# Animate shape keys (1 key per frame)
# ------------------------------
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = len(obj_files)
scene.render.fps = fps

# Reset all keys
for kb in obj.data.shape_keys.key_blocks:
    kb.value = 0.0

for i in range(1, len(obj_files)):
    curr = obj.data.shape_keys.key_blocks[f"frame_{i:04d}"]
    curr.value = 1.0
    curr.keyframe_insert("value", frame=i + 1)

    if i > 1:
        prev = obj.data.shape_keys.key_blocks[f"frame_{i-1:04d}"]
        prev.value = 0.0
        prev.keyframe_insert("value", frame=i + 1)

    prev_name = key_name
print("✔ Animation created.")

# ------------------------------
# Export FBX
# ------------------------------
print("Exporting FBX…")
bpy.ops.export_scene.fbx(
    filepath=fbx_path,
    use_selection=False,
    add_leaf_bones=False,
    bake_space_transform=False,
    object_types={'MESH'},
    bake_anim=True,
    bake_anim_use_all_bones=False,
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False,
)
print("✔ FBX export complete:", fbx_path)
sys.exit(0)
