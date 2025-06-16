import bpy
import os
import sys
import math
import mathutils
import ezdxf
import bmesh

def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.film_transparent = False
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.0

    world = bpy.data.worlds.new("World") if not bpy.data.worlds else bpy.data.worlds["World"]
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes.clear()
    bg_node = world.node_tree.nodes.new(type='ShaderNodeBackground')
    bg_node.inputs[0].default_value = (1, 1, 1, 1)
    output_node = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
    world.node_tree.links.new(bg_node.outputs[0], output_node.inputs[0])

    scene.render.use_freestyle = True
    freestyle_settings = scene.view_layers["ViewLayer"].freestyle_settings
    freestyle_settings.use_culling = True
    for lineset in freestyle_settings.linesets:
        freestyle_settings.linesets.remove(lineset)

    linestyle = bpy.data.linestyles.get("LineStyle")
    if linestyle is None:
        linestyle = bpy.data.linestyles.new("LineStyle")

    lineset = freestyle_settings.linesets.new("LineSet")
    lineset.linestyle = linestyle
    lineset.linestyle.color = (0, 0, 0)
    lineset.linestyle.thickness = 1.0

    if not bpy.data.objects.get('Camera'):
        bpy.ops.object.camera_add(location=(0, 0, 10))
    scene.camera = bpy.data.objects['Camera']

    if not bpy.data.objects.get('Light'):
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 20))

def import_model(model_path):
    ext = os.path.splitext(model_path)[1].lower()
    if ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=model_path)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=model_path)
    elif ext == '.obj':
        bpy.ops.import_scene.obj(filepath=model_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    obj = next((o for o in bpy.context.scene.objects if o.type == 'MESH'), None)
    if not obj:
        raise RuntimeError("No mesh found in model.")

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.location = (0, 0, 0)
    max_dim = max(obj.dimensions)
    if max_dim > 0:
        obj.scale = [2.0 / max_dim] * 3

    return obj

def frame_object(camera, obj, direction_vector, ortho=False):
    bbox = [mathutils.Vector(corner) for corner in obj.bound_box]
    center = sum(bbox, mathutils.Vector()) / 8
    dims = obj.dimensions
    if ortho:
        camera.data.type = 'ORTHO'
        camera.data.ortho_scale = max(dims) * 1.2
        dist = max(dims) * 2
    else:
        camera.data.type = 'PERSP'
        dist = max(dims) * 1.5
    direction = direction_vector.normalized()
    camera.location = center + direction * dist
    direction = center - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

def render_view(output_path, direction_vector, ortho=False):
    obj = next((o for o in bpy.context.scene.objects if o.type == 'MESH'), None)
    frame_object(bpy.data.objects['Camera'], obj, direction_vector, ortho)
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    if not os.path.exists(output_path):
        raise RuntimeError(f"Render failed: {output_path}")

def generate_section_dxf(obj, axis='Z', position=0.0, output_path='Section.dxf'):
    bpy.ops.mesh.primitive_plane_add(size=10)
    cutter = bpy.context.active_object
    cutter.name = "SectionCutter"

    if axis == 'X':
        cutter.rotation_euler[1] = math.radians(90)
        cutter.location.x = position
    elif axis == 'Y':
        cutter.rotation_euler[0] = math.radians(90)
        cutter.location.y = position
    else:
        cutter.location.z = position

    copy = obj.copy()
    copy.data = obj.data.copy()
    bpy.context.collection.objects.link(copy)

    mod = copy.modifiers.new("BooleanCut", type='BOOLEAN')
    mod.operation = 'INTERSECT'
    mod.object = cutter
    mod.solver = 'EXACT'
    bpy.context.view_layer.objects.active = copy
    bpy.ops.object.modifier_apply(modifier="BooleanCut")

    bm = bmesh.new()
    bm.from_mesh(copy.data)

    lines = []
    for edge in bm.edges:
        v1, v2 = edge.verts
        if axis == 'X':
            lines.append(((v1.co.y, v1.co.z), (v2.co.y, v2.co.z)))
        elif axis == 'Y':
            lines.append(((v1.co.x, v1.co.z), (v2.co.x, v2.co.z)))
        else:
            lines.append(((v1.co.x, v1.co.y), (v2.co.x, v2.co.y)))

    bm.free()

    doc = ezdxf.new()
    msp = doc.modelspace()
    for p1, p2 in lines:
        msp.add_line(p1, p2)
    doc.saveas(output_path)

    bpy.data.objects.remove(cutter, do_unlink=True)
    bpy.data.objects.remove(copy, do_unlink=True)

def generate_drawings(model_path, output_dir):
    setup_scene()
    obj = import_model(model_path)

    views = [
        ('X-Section', mathutils.Vector((5, 0, 0)), True),
        ('Y-Section', mathutils.Vector((0, -1, 0)), True),
        ('Z-Section', mathutils.Vector((0, 0, 1)), True),
        ('Axonometric', mathutils.Vector((1, -0.7, 0.7)), False)
    ]

    for name, direction, ortho in views:
        render_view(os.path.join(output_dir, f"{name}.png"), direction, ortho)

    generate_section_dxf(obj, axis='X', position=0.0, output_path=os.path.join(output_dir, "Section_X.dxf"))
    generate_section_dxf(obj, axis='Y', position=0.0, output_path=os.path.join(output_dir, "Section_Y.dxf"))
    generate_section_dxf(obj, axis='Z', position=0.0, output_path=os.path.join(output_dir, "Section_Z.dxf"))

    return True

if __name__ == "__main__":
    try:
        args = sys.argv[sys.argv.index("--") + 1:]
        if len(args) != 2:
            print("Usage: blender --background --python generate_drawings.py -- input_model output_dir", file=sys.stderr)
            sys.exit(1)
        model_path, output_dir = args
        os.makedirs(output_dir, exist_ok=True)
        if not generate_drawings(model_path, output_dir):
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
