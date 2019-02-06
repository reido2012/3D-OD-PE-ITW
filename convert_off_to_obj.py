import glob
import argparse
import sys
import subprocess
import os

parser = argparse.ArgumentParser(description='Converts .off to .obj files for PASCAL3D+')

parser.add_argument('--path_to_dataset', type=str, default='./',
                    help='Path to root folder of the PASCAL3D+ dataset')

args = parser.parse_args()


def convert_models(path_to_models):
    for off_model_path in glob.glob(path_to_models + "CAD/*/*.off"):
        print("Off Model Path: " + off_model_path)

        new_obj_path = off_model_path.replace("CAD", "OBJ").replace(".off", ".obj")
        print("OBJ Model Path: " + new_obj_path)

        new_dir = "/".join(new_obj_path.split("/")[:-1]) + "/"
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)

        assimp_command = "/usr/local/bin/assimp export " + off_model_path + " " + new_obj_path
        subprocess.run(assimp_command.split(), check=True)


convert_models(args.path_to_dataset)
