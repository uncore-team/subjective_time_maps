from setuptools import find_packages, setup

package_name = 'tyrell_tb3_time_map'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anita',
    maintainer_email='anita@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
                'tyrell_tb3_time_map = tyrell_tb3_time_map.tyrell_tb3_time_map_node:main',
        ],
    },
)
