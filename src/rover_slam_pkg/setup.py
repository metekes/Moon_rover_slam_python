from setuptools import find_packages, setup

package_name = 'rover_slam_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy'
        'opencv-python',
        'g2o',
        'helper'],
    zip_safe=True,
    maintainer='mete',
    maintainer_email='metekesici@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rover_slam = rover_slam_pkg.rover_slam:main',
            'rover_direct_slam = rover_slam_pkg.rover_direct_slam:main'
        ],
    },
)
