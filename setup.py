from distutils.core import setup

setup(
    name='3D Pose Estimation',
    version='0.1.0',
    author='Omar Reid',
    author_email='oreid52@googlemail.com',
    description='Code to generate datasets, and train models for 3D pose estimation',
    install_requires=[
        "tqdm",
        "matplotlib",
        "scikit-image",
        "numpy",
        "scipy",
        "scikit-learn",
        "opencv-python",
        "opencv-contrib-python",
        "argparse"
    ],
)