from setuptools import find_packages, setup

package_name = 'cuttlebot_nodes'

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
    maintainer='cuttlebot',
    maintainer_email='cuttlebot@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'location_awareness = cuttlebot_nodes.location_awareness:main',
            'brain_node = cuttlebot_nodes.brain_node:main',
            'state_manager = cuttlebot_nodes.state_manager:main',
            'plot_results = cuttlebot_nodes.plot_results:main',
            'nav_to_pose = cuttlebot_nodes.nav_to_pose:main',
        ],
    },
)
