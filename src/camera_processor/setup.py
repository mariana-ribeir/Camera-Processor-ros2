from setuptools import find_packages, setup
import glob
import os

package_name = 'camera_processor'


def package_data_files(source_dir, install_base):
    """Collect files recursively, preserving the directory structure."""
    data = []
    for root, _, files in os.walk(source_dir):
        if not files:
            continue
        relative = os.path.relpath(root, source_dir)
        destination = install_base if relative == '.' else os.path.join(install_base, relative)
        data.append((destination, [os.path.join(root, f) for f in files]))
    return data


# Find all files in models and launch folders
launch_files = [f for f in glob.glob(os.path.join('launch', '*')) if not os.path.basename(f).startswith('__')]

# Build the data_files list
data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/launch', launch_files)
]
data_files += package_data_files('models', 'share/' + package_name + '/models')


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=[
        'setuptools',
        'opencv-python',
        'cv-bridge',
        'boto3',
        'numpy',
        'pandas',
        'scikit-learn==1.6.1',
        'scipy',
        'antropy',
        'onnxruntime',
    ],
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
            'pose_processor = camera_processor.nodes.pose_processor:main',
            'empatica_gmm_monitor = camera_processor.nodes.empatica_gmm_node:main',
            'polarh10_monitor = camera_processor.nodes.polarh10_monitor_node:main',
        ],
    },
)
