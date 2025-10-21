from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'faint_deployment'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'topomap_generator_node = faint_deployment.topomap_generator_node:main',
            'place_recognition_node = faint_deployment.place_recognition_node:main',
            'goal_reaching_policy_node = faint_deployment.goal_reaching_policy_node:main',
            'visualization_node = faint_deployment.visualization_node:main',
            'disk_writer_node = faint_deployment.disk_writer_node:main',
        ],
    },
)
