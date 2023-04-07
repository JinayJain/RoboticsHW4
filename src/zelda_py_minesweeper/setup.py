from setuptools import setup

package_name = 'zelda_py_minesweeper'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jinay',
    maintainer_email='jinaybjain@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'zelda_py_minesweeper = zelda_py_minesweeper.zelda_py_minesweeper:main'
        ],
    },
)
