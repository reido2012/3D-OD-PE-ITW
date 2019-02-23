# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# blender --background --python mytest.py -- --views 1 

# blender --background --python blender_render.py -- --output_folder /tmp /notebooks/selerio/datasets/pascal3d/PASCAL3D+_release1.1/CAD_OBJ/aeroplane/01.obj
#
import argparse, sys, os

def rotation_tuple(s):
    try:
        x, y, z = map(float, s.split(','))
        return (x, y, z)
    except:
        raise argparse.ArgumentTypeError("rotation values must be x,y,z")

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')

parser.add_argument('--views', type=int, default=10,
                    help='number of views to be rendered')
parser.add_argument('--obj_id', type=str, help='ID of Obj')
parser.add_argument('obj', type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--output_folder', type=str, default='/tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--cad_index', type=str, default="",
                    help='The ID of depth rendered CAD model')
parser.add_argument('--scale', type=float, default=1,
                    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--radians', type=bool, default=False,
                    help='Tells us if viewpoint is given in radians')
parser.add_argument('--specific_viewpoint', type=bool, default=False,
                    help='True when you want to use to render a single depth map from a particulary viewpoint .')
parser.add_argument('--viewpoint', help="Tuple of XYZ rotation", dest="rotation_tuple", default=(0,0,0), type=rotation_tuple, nargs='+')
parser.add_argument('--depth_scale', type=float, default=1,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='8',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='PNG',
                    help='Format of files generated. Either PNG or OPEN_EXR')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

import bpy

# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

bpy.context.scene.render.engine = 'CYCLES'
# Add passes for additionally dumping albedo and normals.
bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
bpy.context.scene.render.image_settings.file_format = args.format
bpy.context.scene.render.image_settings.color_depth = args.color_depth

# Clear default nodes
for n in tree.nodes:
    tree.nodes.remove(n)

# Create input render layer node.
render_layers = tree.nodes.new('CompositorNodeRLayers')

depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = 'Depth Output'
if args.format == 'OPEN_EXR':
    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
else:
    # Remap as other types can not represent the full range of depth.
    map = tree.nodes.new(type="CompositorNodeMapValue")
    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    map.offset = [-0.7]
    map.size = [args.depth_scale]
    map.use_min = True
    map.min = [0]
    map.use_max = True
    map.max = [1]

    links.new(render_layers.outputs['Depth'], map.inputs[0])
    links.new(map.outputs[0], depth_file_output.inputs[0])
    
    # New
    invert = tree.nodes.new(type="CompositorNodeInvert")
    links.new(map.outputs[0], invert.inputs[1])

    # The viewer can come in handy for inspecting the results in the GUI
    depthViewer = tree.nodes.new(type="CompositorNodeViewer")
    links.new(invert.outputs[0], depthViewer.inputs[0])
    # Use alpha from input.
    links.new(render_layers.outputs[1], depthViewer.inputs[1])
    
    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    links.new(invert.outputs[0], depth_file_output.inputs[0])

# Delete default cube
bpy.data.objects['Cube'].select = True
bpy.ops.object.delete()

# Add New Object
bpy.ops.import_scene.obj(filepath=args.obj, axis_forward='-Y', axis_up='Z')
current_obj = None

for obj in bpy.context.scene.objects:
    if obj.name in ['Camera', 'Lamp']:
        continue
    obj.name = '3d_model'
    bpy.context.scene.objects.active = obj
    current_obj = obj
    if args.scale != 1:
        bpy.ops.transform.resize(value=(args.scale,args.scale,args.scale))
        bpy.ops.object.transform_apply(scale=True)
    if args.remove_doubles:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')
    if args.edge_split:
        bpy.ops.object.modifier_add(type='EDGE_SPLIT')
        bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier="EdgeSplit")


def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty


rotation_mode = 'XYZ'
scene = bpy.context.scene
scene.render.resolution_x = 700
scene.render.resolution_y = 700
scene.render.resolution_percentage = 100
scene.render.alpha_mode = 'TRANSPARENT'
cam = scene.objects['Camera']
cam.rotation_mode = rotation_mode
cam.location = (0, -1.4, 0)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
cam_constraint.target = current_obj

model_id = os.path.split(args.obj)[-1].split(".")[0]
print("Model ID: ", end="")
print(model_id)

model_type = os.path.split(os.path.split(args.obj)[0])[1] 

print("Model Type: ", end="")
print(model_type)

fp = os.path.join(args.output_folder, model_type, model_id)
print("Filepath: ", end="")
print(fp)


scene.render.image_settings.file_format = 'PNG'  # set output format to .png

from math import radians

current_obj.rotation_mode = rotation_mode


for output_node in [depth_file_output]:
    output_node.base_path = ''

    
def render_image(directory_path, img_name):
    scene.render.filepath = directory_path + "/" + img_name
    print("Scene Render FP: ", end="")
    print(scene.render.filepath)
    depth_file_output.file_slots[0].path = scene.render.filepath + "_"
    bpy.ops.render.render(write_still=True)  # render still


def rotate_full_pose_space():
    """
    Render depth image for model in full pose space - discretized to 30 degree intervals
    """
#     print("Original - Rotation Euler")
    print("Rendering Full Pose Space")
    current_obj.rotation_euler = (0, 0, 0)
    current_obj.location = (0, 0, 0)
    
    for rot_x  in range(-180, 180, 30):
        current_obj.rotation_euler[0] = radians(rot_x)
        
        for rot_z in range(0, 360, 30):
            #Anti-Clockwize about Z
            current_obj.rotation_euler[2] = radians(rot_z)
            
            for rot_y in range(0, 360, 30):
                current_obj.rotation_euler[1] = radians(rot_y)
                img_id = "x_" + str(rot_x) + "_z_"+str(rot_z)+ "_y_" + str(rot_y)
                render_image(fp, img_id)


def render_at_viewpoint():
    # Render to correct place - we want to do this for each model type
    current_obj.rotation_euler = (0, 0, 0)
    current_obj.location = (0, 0, 0)
    print("Cam Location:")
    print(cam.location)
    print("Obj Location:")
    print(current_obj.location)
    print(args.radians)
    print("Rotation Tuple")
    print(args.rotation_tuple[0])
    # OBJ files are not front facing at first we are correcting for this
    base_tuple = (radians(90), 0, radians(0))

    if not args.radians:
        print("Converting from Degrees to Radians")
        actual_rotation_tuple = []

        for counter, angle in enumerate(args.rotation_tuple[0]):
            actual_rotation_tuple.append(radians(angle) + base_tuple[counter])
        print(actual_rotation_tuple)

        current_obj.rotation_euler = tuple(actual_rotation_tuple)
    else:
        new_rotation = []

        for counter, angle in enumerate(args.rotation_tuple[0]):
            new_rotation.append((angle) + base_tuple[counter])

        new_rotation = tuple(new_rotation)
        print(new_rotation)
        current_obj.rotation_euler[0] = new_rotation[0]
        current_obj.rotation_euler[1] = new_rotation[1]
        current_obj.rotation_euler[2] = new_rotation[2]

        # current_obj.rotation_euler = new_rotation

    directory_path = args.output_folder
    render_image(directory_path, args.obj_id + "_" + args.cad_index)
    print(current_obj.rotation_euler)


if args.specific_viewpoint:
    render_at_viewpoint()
else:
    rotate_full_pose_space()
