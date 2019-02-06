import glob
import subprocess
import os
import logging
import time

logging.basicConfig(filename='pose_space.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


def render_pose_space():
    command = "nvidia-docker run -v /home/omarreid/selerio/:/workdir peterlauri/blender-python:latest blender -noaudio --background --python /workdir/pix3d/blender_render.py -- --output_folder /workdir/pix3d/depth_renderings "
    
    start = time.time()
    for object_path in glob.glob("/home/omarreid/selerio/datasets/pascal3d/PASCAL3D+_release1.1/CAD_OBJ/*/*.obj"):
        
        print(object_path)
#         object_path = "/home/omarreid/selerio/datasets/pascal3d/PASCAL3D+_release1.1/CAD_OBJ/car/05.obj"
#         if output_folder_exists(object_path):
#             continue
            
        docker_object_path = "/workdir/" + "/".join(object_path.split("/")[4:])
        full_command = command + docker_object_path
        process = subprocess.run(full_command.split(), check=True)
        logging.info(object_path)
    end = time.time()
    
    print("Time Taken: ")
    print(end-start)
    logging.info(end-start)


def output_folder_exists(object_path):
    output_folder = "/home/omarreid/selerio/pix3d/depth_renderings/" + object_path.split("/")[-2] + "/" + object_path.split("/")[-1].split(".")[0]
    print(output_folder)
    return os.path.isdir(output_folder)


def render_for_dataset(image_class, gt_index, rotation_xyz, record_id):
    x_rotation, y_rotation, z_rotation = rotation_xyz
    print(int(round(x_rotation)))
    print(round(y_rotation))
    print(round(z_rotation))
    
    print(f"x rotation: {round(x_rotation)}")
    print(f"y rotation: {round(y_rotation)}")
    print(f"z rotation: {round(z_rotation)}")

    #render for each object in CAD_OBJ in image_class
    # return path to depth image and we want - gt_index
    for object_path in glob.glob("/home/omarreid/selerio/datasets/pascal3d/PASCAL3D+_release1.1/CAD_OBJ/" + image_class + "/*.obj"):
        
        curr_obj_cad_index = object_path.split("/")[-1].split(".")[0]
        
        command = "nvidia-docker run -v /home/omarreid/selerio/:/workdir peterlauri/blender-python:latest blender -noaudio --background --python /workdir/pix3d/blender_render.py --  --specific_viewpoint True --cad_index " +  curr_obj_cad_index + " --viewpoint=" + str(x_rotation) + "," + str(y_rotation) + "," + str(z_rotation) + " --output_folder /workdir/pix3d/synth_renderings/" + str(record_id) + " "
        
        print("Object Path: " + object_path)
        print("Curr CAD INDEX: " + str(curr_obj_cad_index) )
        print("Command: " + command)
        
        docker_object_path = "/workdir/" + "/".join(object_path.split("/")[4:])
        full_command = command + docker_object_path
        process = subprocess.run(full_command.split(), check=True)
#         logging.info(object_path)
        break
    
    return "/home/omarreid/selerio/pix3d/synth_renderings/" + str(record_id) + "/" + gt_index + "_0001.png"
                                     
        
if __name__ == '__main__':
    render_for_dataset('aeroplane', '01', (-30, -30, -30), 1234)