import bpy
import sys
import os

# ------------------------------------------------------------
# Parse arguments passed after "--"
# ------------------------------------------------------------
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

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

# ------------------------------------------------------------
# Clean scene
# ------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)

# ------------------------------------------------------------
# Collect OBJ files
# ------------------------------------------------------------
obj_files = sorted([
    os.path.join(obj_folder, f)
    for f in os.listdir(obj_folder)
    if f.lower().endswith(".obj")
])

if not obj_files:
    print("❌ ERROR: No OBJ files found.")
    sys.exit(1)

print(f"Found {len(obj_files)} OBJ frames")

# ------------------------------------------------------------
# Import first OBJ (rest pose)
# ------------------------------------------------------------
print("Importing base mesh:", obj_files[0])
bpy.ops.wm.obj_import(filepath=obj_files[0])

# Get imported object
obj = bpy.context.selected_objects[0]
obj.name = "FaceMesh"

# ------------------------------------------------------------
# Create shape keys
# ------------------------------------------------------------
obj.shape_key_add(name="Basis")  # base
bpy.context.view_layer.objects.active = obj

for i, filepath in enumerate(obj_files[1:], start=1):
    print(f"Importing shape {i}: {filepath}")

    # Import frame
    bpy.ops.wm.obj_import(filepath=filepath)
    imported = bpy.context.selected_objects[0]

    # Copy vertices into a new shape key
    key = obj.shape_key_add(name=f"frame_{i:04d}")

    for v_main, v_imp in zip(obj.data.vertices, imported.data.vertices):
        key.data[v_main.index].co = v_imp.co

    # Delete imported temp mesh
    bpy.data.objects.remove(imported, do_unlink=True)

print("✔ Shape keys generated.")

# ------------------------------------------------------------
# Animate shape keys (1 key per frame)
# ------------------------------------------------------------
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = len(obj_files)
scene.render.fps = fps

for i in range(len(obj_files)):
    key_name = f"frame_{i:04d}" if i != 0 else "Basis"
    if key_name in obj.data.shape_keys.key_blocks:

        for k in obj.data.shape_keys.key_blocks:
            k.value = 0
            k.keyframe_insert("value", frame=i + 1)

        obj.data.shape_keys.key_blocks[key_name].value = 1.0
        obj.data.shape_keys.key_blocks[key_name].keyframe_insert("value", frame=i + 1)

print("✔ Animation created.")

# ------------------------------------------------------------
# Export FBX
# ------------------------------------------------------------
print("Exporting FBX…")
bpy.ops.export_scene.fbx(
    filepath=fbx_path,
    use_selection=False,
    add_leaf_bones=False,
    bake_space_transform=False,
    apply_scale_options='FBX_SCALE_ALL',
    object_types={'MESH'},
    bake_anim=True,
    bake_anim_use_all_bones=False,
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False
)

print("✔ FBX export complete:", fbx_path)

sys.exit(0)
