from setuptools import find_packages, setup

package_name = 'rb2301_tutorial'

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
    maintainer='lujianyi',
    maintainer_email='jianyilu13@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'tut = rb2301_tutorial.tutorial:main',
            'fake = rb2301_tutorial.fake:main',
            'logger = rb2301_tutorial.logger:main',
            'recorder = rb2301_tutorial.recorder:main',
        ],
    },
)
