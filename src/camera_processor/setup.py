from setuptools import find_packages, setup
import glob
import os

package_name = 'camera_processor'
models_files = glob.glob(os.path.join('models', '*'))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + ([('share/' + package_name + '/models', models_files)] if models_files else []),
    install_requires=['setuptools', 'opencv-python', 'cv-bridge'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='marianadsr.2001@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'color_processor = camera_processor.color_processor:main',
            'person_processor = camera_processor.person_processor:main',
        ],
    },
)
