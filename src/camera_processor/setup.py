from setuptools import find_packages, setup
import glob
import os

package_name = 'camera_processor'

# Find all files in models and launch folders
models_files = glob.glob(os.path.join('models', '*'))
launch_files = [f for f in glob.glob(os.path.join('launch', '*')) if not os.path.basename(f).startswith('__')]

# Build the data_files list
data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/models', models_files),
    ('share/' + package_name + '/launch', launch_files)
]


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools', 'opencv-python', 'cv-bridge'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='marianadsr.2001@gmail.com',
    description='ROS2 Camera Processor with YOLO and ReID',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'color_processor = camera_processor.nodes.color_processor:main',
            'person_processor = camera_processor.nodes.person_processor:main',
            'heuristic_pose = camera_processor.nodes.heuristic_pose:main',
            'ai_pose = camera_processor.nodes.ai_pose:main',
            'pose_processor = camera_processor.nodes.pose_processor:main'
        ],
    },
)
